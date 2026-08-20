# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import pytest

from dflash_mlx.engine.ddtree import (
    build_flat_ddtree,
    build_flat_tree_attention_mask,
    build_flat_tree_inputs,
    branch_positions,
    candidate_token_ids,
    clone_cache_for_batch,
    copy_selected_cache,
    follow_verified_tree,
    flat_tree_depths,
    flat_tree_path_token_ids,
    flat_tree_positions,
    flat_tree_token_ids,
    restore_cache,
    select_captured,
    select_tree_slots,
    snapshot_cache,
    top_ids_and_values_desc,
)
from dflash_mlx.model import ContextOnlyDraftKVCache
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache


def test_clone_cache_for_batch_repeats_batched_arrays_only():
    cache = SimpleNamespace(
        keys=mx.array([[[1, 2], [3, 4]]]),
        positions=mx.array([7, 8]),
        nested=[mx.array([[10, 20]])],
        offset=2,
    )

    clone = clone_cache_for_batch([cache], 3)[0]

    assert clone is not cache
    assert clone.offset == 2
    assert clone.positions.tolist() == [7, 8]
    assert clone.keys.shape == (3, 2, 2)
    assert clone.keys[2].tolist() == [[1, 2], [3, 4]]
    assert clone.nested[0].shape == (3, 2)
    assert clone.nested[0][1].tolist() == [10, 20]


def test_copy_selected_cache_extracts_batch_row_and_preserves_metadata():
    dst = SimpleNamespace(keys=None, offset=0)
    src = SimpleNamespace(
        keys=mx.array(
            [
                [[1, 2]],
                [[3, 4]],
                [[5, 6]],
            ]
        ),
        positions=mx.array([11]),
        state=[mx.array([[9], [8], [7]])],
        offset=12,
    )

    copy_selected_cache(dst_entries=[dst], src_entries=[src], batch_index=1)

    assert dst.offset == 12
    assert dst.positions.tolist() == [11]
    assert dst.keys.shape == (1, 1, 2)
    assert dst.keys.tolist() == [[[3, 4]]]
    assert dst.state[0].shape == (1, 1)
    assert dst.state[0].tolist() == [[8]]


def test_clone_cache_for_batch_handles_context_draft_cache():
    cache = ContextOnlyDraftKVCache(sink_size=1, window_size=2)
    cache.append_context(
        mx.array([[[[1.0], [2.0]]]], dtype=mx.float32),
        mx.array([[[[3.0], [4.0]]]], dtype=mx.float32),
        2,
    )

    clone = clone_cache_for_batch([cache], 3)[0]

    assert isinstance(clone, ContextOnlyDraftKVCache)
    assert clone is not cache
    assert clone.sink_size == 1
    assert clone.window_size == 2
    assert clone.offset == 2
    assert clone.keys.shape == (3, 1, 2, 1)
    assert clone.values.shape == (3, 1, 2, 1)
    assert clone.positions.tolist() == [0, 1]


def test_copy_selected_cache_handles_recurrent_rollback_cache():
    dst = RecurrentRollbackCache(2, conv_kernel_size=4)
    src = RecurrentRollbackCache(2, conv_kernel_size=4)
    src.cache = [
        mx.array(
            [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
            ],
            dtype=mx.float32,
        ),
        mx.array(
            [
                [[5.0]],
                [[6.0]],
            ],
            dtype=mx.float32,
        ),
    ]
    src._armed = True
    src._snapshot = list(src.cache)

    copy_selected_cache(dst_entries=[dst], src_entries=[src], batch_index=1)

    assert isinstance(dst, RecurrentRollbackCache)
    assert dst.conv_kernel_size == 4
    assert dst.cache[0].shape == (1, 2, 1)
    assert dst.cache[0].tolist() == [[[3.0], [4.0]]]
    assert dst.cache[1].shape == (1, 1, 1)
    assert dst.cache[1].tolist() == [[[6.0]]]
    assert dst._armed is True
    assert dst._snapshot[0].shape == (1, 2, 1)


def test_snapshot_restore_cache_restores_list_identity_snapshot():
    cache = SimpleNamespace(values=[mx.array([[1]])], offset=1)
    snap = snapshot_cache([cache])
    cache.values.append(mx.array([[2]]))
    cache.offset = 3

    restore_cache(snap)

    assert len(cache.values) == 1
    assert cache.values[0].tolist() == [[1]]
    assert cache.offset == 1


def test_select_captured_handles_dict_and_list():
    captured_dict = {2: mx.array([[[1]], [[2]]])}
    selected_dict = select_captured(captured_dict, 1)
    assert selected_dict[2].shape == (1, 1, 1)
    assert selected_dict[2].tolist() == [[[2]]]

    captured_list = [mx.array([[[3]], [[4]]])]
    selected_list = select_captured(captured_list, 0)
    assert selected_list[0].shape == (1, 1, 1)
    assert selected_list[0].tolist() == [[[3]]]


def test_select_tree_slots_gathers_non_prefix_path_order():
    captured_dict = {2: mx.array([[[10], [11], [12], [13]]])}
    selected_dict = select_tree_slots(captured_dict, [0, 1, 3])

    assert selected_dict[2].shape == (1, 3, 1)
    assert selected_dict[2].tolist() == [[[10], [11], [13]]]

    captured_list = [mx.array([[[20], [21], [22], [23]]])]
    selected_list = select_tree_slots(captured_list, [0, 2])

    assert selected_list[0].shape == (1, 2, 1)
    assert selected_list[0].tolist() == [[[20], [22]]]


def test_branch_positions_first_and_margin():
    top_values = [
        [9.0, 8.0],
        [7.0, 6.9],
        [10.0, 4.0],
        [5.0, 4.8],
    ]

    assert branch_positions(
        top_values_desc=top_values,
        block_len=5,
        max_branch_positions=2,
        strategy="first",
    ) == [0, 1]
    assert branch_positions(
        top_values_desc=top_values,
        block_len=5,
        max_branch_positions=2,
        strategy="margin",
    ) == [1, 3]


def test_candidate_token_ids_truncates_or_fills_suffix():
    prefix = mx.array([1, 2], dtype=mx.uint32)
    suffix = mx.array([3, 4, 5], dtype=mx.uint32)

    assert candidate_token_ids(
        prefix_tokens=prefix,
        suffix_tokens=suffix,
        block_len=4,
    ).tolist() == [1, 2, 3, 4]
    assert candidate_token_ids(
        prefix_tokens=mx.array([1, 2, 3, 4], dtype=mx.uint32),
        suffix_tokens=suffix,
        block_len=3,
    ).tolist() == [1, 2, 3]


def test_top_ids_and_values_desc_returns_descending_scores():
    logits = mx.array(
        [
            [0.1, 0.8, 0.3, 0.2],
            [4.0, 1.0, 5.0, 3.0],
        ]
    )

    ids, values = top_ids_and_values_desc(logits, width=2)

    assert ids == [[1, 2], [2, 0]]
    row0_lse = math.log(sum(math.exp(x) for x in [0.1, 0.8, 0.3, 0.2]))
    row1_lse = math.log(sum(math.exp(x) for x in [4.0, 1.0, 5.0, 3.0]))
    assert values[0] == pytest.approx([0.8 - row0_lse, 0.3 - row0_lse])
    assert values[1] == pytest.approx([5.0 - row1_lse, 4.0 - row1_lse])


def test_build_flat_ddtree_chain_seed_expands_best_first_branch():
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        top_scores_desc=[
            [-0.1, -2.0],
            [-0.2, -0.3],
            [-0.3, -4.0],
        ],
        budget=5,
    )

    assert tree.parents == [-1, 0, 1, 2, 1, 4]
    assert tree.depths == [1, 2, 3, 2, 3]
    assert tree.token_ids == [10, 20, 30, 21, 30]
    assert tree.child_maps[0] == {10: 1}
    assert tree.child_maps[1] == {20: 2, 21: 4}
    assert tree.visibility[5][0] is True
    assert tree.visibility[5][1] is True
    assert tree.visibility[5][2] is False
    assert tree.visibility[5][4] is True
    assert tree.visibility[5][5] is True


def test_follow_verified_tree_walks_sibling_path():
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        top_scores_desc=[
            [-0.1, -2.0],
            [-0.2, -0.3],
            [-0.3, -4.0],
        ],
        budget=5,
    )

    accepted, next_token = follow_verified_tree(tree, [10, 21, 0, 0, 30, 99])

    assert accepted == [0, 1, 4, 5]
    assert next_token == 99


def test_flat_tree_path_token_ids_returns_root_to_slot_paths():
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        top_scores_desc=[
            [-0.1, -2.0],
            [-0.2, -0.3],
            [-0.3, -4.0],
        ],
        budget=5,
    )

    paths = flat_tree_path_token_ids(tree, root_token_id=7)

    assert paths == [
        [7],
        [7, 10],
        [7, 10, 20],
        [7, 10, 20, 30],
        [7, 10, 21],
        [7, 10, 21, 30],
    ]


def test_flat_tree_inputs_match_lucebox_tree_contract():
    tree = build_flat_ddtree(
        top_token_ids_desc=[
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        top_scores_desc=[
            [-0.1, -2.0],
            [-0.2, -0.3],
            [-0.3, -4.0],
        ],
        budget=5,
    )

    assert flat_tree_token_ids(tree, root_token_id=7) == [7, 10, 20, 30, 21, 30]
    assert flat_tree_depths(tree) == [0, 1, 2, 3, 2, 3]
    assert flat_tree_positions(tree, prefix_len=100) == [100, 101, 102, 103, 102, 103]

    mask = build_flat_tree_attention_mask(tree, prefix_len=2)

    assert mask.shape == (6, 8)
    assert mask[:, :2].tolist() == [[True, True]] * 6
    assert mask[0, 2:].tolist() == [True, False, False, False, False, False]
    assert mask[3, 2:].tolist() == [True, True, True, True, False, False]
    assert mask[5, 2:].tolist() == [True, True, False, False, True, True]

    inputs = build_flat_tree_inputs(tree, root_token_id=7, prefix_len=100)

    assert inputs.size == 6
    assert inputs.token_ids.tolist() == [7, 10, 20, 30, 21, 30]
    assert inputs.depths.tolist() == [0, 1, 2, 3, 2, 3]
    assert inputs.parent_ids.tolist() == [-1, 0, 1, 2, 1, 4]
    assert inputs.positions.tolist() == [100, 101, 102, 103, 102, 103]
    assert inputs.attention_mask.tolist() == build_flat_tree_attention_mask(
        tree,
        prefix_len=100,
    ).tolist()


def test_build_flat_ddtree_handles_empty_budget():
    tree = build_flat_ddtree(
        top_token_ids_desc=[[1, 2]],
        top_scores_desc=[[-0.1, -0.2]],
        budget=0,
    )

    assert tree.token_ids == []
    assert tree.depths == []
    assert tree.parents == [-1]
    assert tree.child_maps == [{}]
    assert tree.visibility == [[True]]


def test_follow_verified_tree_requires_full_posterior():
    tree = build_flat_ddtree(
        top_token_ids_desc=[[1]],
        top_scores_desc=[[-0.1]],
        budget=1,
    )

    with pytest.raises(ValueError, match="posterior_token_ids"):
        follow_verified_tree(tree, [1])

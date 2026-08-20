# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import copy
import heapq
import time
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx

from dflash_mlx.draft_backend import _astype_if_needed, _draft_compute_dtype
from dflash_mlx.engine.acceptance import match_acceptance_length
from dflash_mlx.engine.sampling import (
    eval_logits_and_captured,
    greedy_tokens_with_mask,
)


BranchPositionStrategy = Literal["first", "margin"]


@dataclass
class DDTreeCandidateResult:
    source: str
    ids: mx.array
    posterior: mx.array
    logits: mx.array
    hidden_states: list[mx.array] | dict[int, mx.array]
    acceptance_len: int
    verify_us: float

    @property
    def commit_count(self) -> int:
        return 1 + int(self.acceptance_len)


@dataclass
class FlatDDTree:
    token_ids: list[int]
    depths: list[int]
    parents: list[int]
    child_maps: list[dict[int, int]]
    visibility: list[list[bool]]

    @property
    def n_nodes(self) -> int:
        return len(self.token_ids)

    @property
    def size(self) -> int:
        return len(self.parents)


@dataclass(frozen=True)
class FlatDDTreeInputs:
    token_ids: mx.array
    depths: mx.array
    parent_ids: mx.array
    positions: mx.array
    attention_mask: mx.array

    @property
    def size(self) -> int:
        return int(self.token_ids.shape[0])


def build_flat_ddtree(
    *,
    top_token_ids_desc: list[list[int]],
    top_scores_desc: list[list[float]],
    budget: int,
    chain_seed: bool = True,
) -> FlatDDTree:
    """Build a DFS-flat DDTree skeleton from per-position top-k scores.

    Slot 0 is the root. Non-root tree nodes live in slots 1..N, so
    ``token_ids[i - 1]`` and ``depths[i - 1]`` describe flat node ``i``.
    Scores are expected to be log-prob-like values when available; raw logits
    are acceptable for diagnostics but not for calibrated product decisions.
    """
    if len(top_token_ids_desc) != len(top_scores_desc):
        raise ValueError("top_token_ids_desc and top_scores_desc must have equal length")
    for ids, scores in zip(top_token_ids_desc, top_scores_desc, strict=True):
        if len(ids) != len(scores):
            raise ValueError("top_token_ids_desc and top_scores_desc rows must have equal length")
    max_nodes = max(0, int(budget))
    token_ids: list[int] = []
    depths: list[int] = []
    parents = [-1]
    child_maps: list[dict[int, int]] = [{}]
    heap: list[tuple[float, int, int, int, float]] = []

    def score_at(depth: int, rank: int) -> float:
        return float(top_scores_desc[depth - 1][rank])

    def token_at(depth: int, rank: int) -> int:
        return int(top_token_ids_desc[depth - 1][rank])

    def append_node(parent_index: int, depth: int, token_id: int) -> int | None:
        if token_id in child_maps[parent_index]:
            return None
        node_index = len(parents)
        token_ids.append(int(token_id))
        depths.append(int(depth))
        parents.append(int(parent_index))
        child_maps.append({})
        child_maps[parent_index][int(token_id)] = node_index
        return node_index

    def push_candidate(parent_index: int, depth: int, rank: int, score: float) -> None:
        if depth < 1 or depth > len(top_token_ids_desc):
            return
        if rank < 0 or rank >= len(top_token_ids_desc[depth - 1]):
            return
        if rank >= len(top_scores_desc[depth - 1]):
            return
        heapq.heappush(heap, (-float(score), int(parent_index), int(depth), int(rank), float(score)))

    if max_nodes > 0 and top_token_ids_desc:
        if chain_seed:
            cumulative_score = 0.0
            parent_index = 0
            for depth in range(1, min(len(top_token_ids_desc), max_nodes) + 1):
                if not top_token_ids_desc[depth - 1]:
                    break
                rank0_score = score_at(depth, 0)
                cumulative_score += rank0_score
                current_parent = parent_index
                node_index = append_node(current_parent, depth, token_at(depth, 0))
                if node_index is None:
                    break
                if len(top_token_ids_desc[depth - 1]) > 1 and len(top_scores_desc[depth - 1]) > 1:
                    push_candidate(
                        current_parent,
                        depth,
                        1,
                        cumulative_score - rank0_score + score_at(depth, 1),
                    )
                parent_index = node_index
                if len(token_ids) >= max_nodes:
                    break
        else:
            if top_token_ids_desc[0]:
                push_candidate(0, 1, 0, score_at(1, 0))

    while heap and len(token_ids) < max_nodes:
        _neg_score, parent_index, depth, rank, score = heapq.heappop(heap)
        node_index = append_node(parent_index, depth, token_at(depth, rank))
        if rank + 1 < len(top_token_ids_desc[depth - 1]) and rank + 1 < len(top_scores_desc[depth - 1]):
            push_candidate(
                parent_index,
                depth,
                rank + 1,
                score - score_at(depth, rank) + score_at(depth, rank + 1),
            )
        if node_index is None:
            continue
        if depth < len(top_token_ids_desc) and top_token_ids_desc[depth]:
            push_candidate(node_index, depth + 1, 0, score + score_at(depth + 1, 0))

    visibility = build_tree_visibility(parents)
    return FlatDDTree(
        token_ids=token_ids,
        depths=depths,
        parents=parents,
        child_maps=child_maps,
        visibility=visibility,
    )


def build_tree_visibility(parents: list[int]) -> list[list[bool]]:
    size = len(parents)
    visibility = [[False for _ in range(size)] for _ in range(size)]
    for node_index in range(size):
        cursor = node_index
        while cursor >= 0:
            visibility[node_index][cursor] = True
            cursor = int(parents[cursor])
    return visibility


def flat_tree_token_ids(
    tree: FlatDDTree,
    *,
    root_token_id: int,
) -> list[int]:
    return [int(root_token_id), *[int(token_id) for token_id in tree.token_ids]]


def flat_tree_depths(tree: FlatDDTree) -> list[int]:
    return [0, *[int(depth) for depth in tree.depths]]


def flat_tree_positions(
    tree: FlatDDTree,
    *,
    prefix_len: int,
) -> list[int]:
    base = int(prefix_len)
    return [base + depth for depth in flat_tree_depths(tree)]


def build_flat_tree_attention_mask(
    tree: FlatDDTree,
    *,
    prefix_len: int,
) -> mx.array:
    prefix = max(0, int(prefix_len))
    rows: list[list[bool]] = []
    for visibility_row in tree.visibility:
        rows.append([True] * prefix + [bool(value) for value in visibility_row])
    return mx.array(rows, dtype=mx.bool_)


def build_flat_tree_inputs(
    tree: FlatDDTree,
    *,
    root_token_id: int,
    prefix_len: int,
) -> FlatDDTreeInputs:
    token_ids = mx.array(flat_tree_token_ids(tree, root_token_id=root_token_id), dtype=mx.uint32)
    depths = mx.array(flat_tree_depths(tree), dtype=mx.int32)
    parent_ids = mx.array(tree.parents, dtype=mx.int32)
    positions = mx.array(flat_tree_positions(tree, prefix_len=prefix_len), dtype=mx.int32)
    attention_mask = build_flat_tree_attention_mask(tree, prefix_len=prefix_len)
    return FlatDDTreeInputs(
        token_ids=token_ids,
        depths=depths,
        parent_ids=parent_ids,
        positions=positions,
        attention_mask=attention_mask,
    )


def flat_tree_path_token_ids(
    tree: FlatDDTree,
    *,
    root_token_id: int,
) -> list[list[int]]:
    paths: list[list[int]] = []
    for slot_index in range(tree.size):
        ancestry: list[int] = []
        cursor = slot_index
        while cursor > 0:
            ancestry.append(cursor)
            cursor = int(tree.parents[cursor])
        node_tokens = [
            int(tree.token_ids[node_index - 1])
            for node_index in reversed(ancestry)
        ]
        paths.append([int(root_token_id), *node_tokens])
    return paths


def follow_verified_tree(
    tree: FlatDDTree,
    posterior_token_ids: list[int],
) -> tuple[list[int], int]:
    if len(posterior_token_ids) < tree.size:
        raise ValueError("posterior_token_ids must cover every flat tree slot")
    accepted = [0]
    current_index = 0
    next_token = int(posterior_token_ids[current_index])
    while True:
        child_index = tree.child_maps[current_index].get(next_token)
        if child_index is None:
            return accepted, next_token
        current_index = int(child_index)
        accepted.append(current_index)
        next_token = int(posterior_token_ids[current_index])


def clone_attr_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def snapshot_cache(cache_entries: list[Any]) -> list[tuple[Any, dict[str, Any]]]:
    return [
        (entry, {key: clone_attr_value(value) for key, value in vars(entry).items()})
        for entry in cache_entries
    ]


def restore_cache(snapshot: list[tuple[Any, dict[str, Any]]]) -> None:
    for entry, state in snapshot:
        vars(entry).clear()
        vars(entry).update({key: clone_attr_value(value) for key, value in state.items()})


def repeat_batch_value(value: Any, batch_size: int) -> Any:
    if isinstance(value, mx.array):
        if value.ndim >= 2 and int(value.shape[0]) == 1 and int(batch_size) > 1:
            return mx.repeat(value, int(batch_size), axis=0)
        return value
    if isinstance(value, list):
        return [repeat_batch_value(item, batch_size) for item in value]
    if isinstance(value, tuple):
        return tuple(repeat_batch_value(item, batch_size) for item in value)
    if isinstance(value, dict):
        return {key: repeat_batch_value(item, batch_size) for key, item in value.items()}
    return value


def select_batch_value(value: Any, batch_index: int) -> Any:
    if isinstance(value, mx.array):
        if value.ndim >= 2 and int(value.shape[0]) > int(batch_index):
            return value[int(batch_index) : int(batch_index) + 1]
        return value
    if isinstance(value, list):
        return [select_batch_value(item, batch_index) for item in value]
    if isinstance(value, tuple):
        return tuple(select_batch_value(item, batch_index) for item in value)
    if isinstance(value, dict):
        return {key: select_batch_value(item, batch_index) for key, item in value.items()}
    return value


def clone_cache_for_batch(cache_entries: list[Any], batch_size: int) -> list[Any]:
    clones: list[Any] = []
    for entry in cache_entries:
        clone = copy.copy(entry)
        vars(clone).clear()
        vars(clone).update(
            {
                key: repeat_batch_value(value, batch_size)
                for key, value in vars(entry).items()
            }
        )
        clones.append(clone)
    return clones


def copy_selected_cache(
    *,
    dst_entries: list[Any],
    src_entries: list[Any],
    batch_index: int,
) -> None:
    for dst, src in zip(dst_entries, src_entries, strict=True):
        vars(dst).clear()
        vars(dst).update(
            {
                key: select_batch_value(value, batch_index)
                for key, value in vars(src).items()
            }
        )


def select_captured(
    captured: list[mx.array] | dict[int, mx.array],
    batch_index: int,
) -> list[mx.array] | dict[int, mx.array]:
    if isinstance(captured, dict):
        return {
            key: select_batch_value(value, batch_index)
            for key, value in captured.items()
        }
    return [select_batch_value(value, batch_index) for value in captured]


def select_tree_slots(
    captured: list[mx.array] | dict[int, mx.array],
    slot_indices: list[int],
) -> list[mx.array] | dict[int, mx.array]:
    indices = mx.array([int(slot_index) for slot_index in slot_indices], dtype=mx.int32)

    def select(value: mx.array) -> mx.array:
        return mx.take(value, indices, axis=1)

    if isinstance(captured, dict):
        return {key: select(value) for key, value in captured.items()}
    return [select(value) for value in captured]


def branch_positions(
    *,
    top_values_desc: list[list[float]],
    block_len: int,
    max_branch_positions: int,
    strategy: BranchPositionStrategy,
) -> list[int]:
    limit = min(
        len(top_values_desc),
        max(0, int(max_branch_positions)),
        max(0, int(block_len) - 1),
    )
    if limit <= 0:
        return []
    if strategy == "first":
        return list(range(limit))
    if strategy != "margin":
        raise ValueError("DDTree branch position strategy must be first or margin")
    scored_positions: list[tuple[float, int]] = []
    for pos in range(min(len(top_values_desc), max(0, int(block_len) - 1))):
        values = top_values_desc[pos]
        margin = float("inf") if len(values) < 2 else float(values[0]) - float(values[1])
        scored_positions.append((margin, pos))
    return [pos for _, pos in sorted(scored_positions)[:limit]]


def candidate_token_ids(
    *,
    prefix_tokens: mx.array,
    suffix_tokens: mx.array,
    block_len: int,
) -> mx.array:
    if int(prefix_tokens.shape[0]) >= int(block_len):
        return prefix_tokens[:block_len]
    need = int(block_len) - int(prefix_tokens.shape[0])
    return mx.concatenate([prefix_tokens, suffix_tokens[:need]], axis=0)


def top_ids_and_values_desc(
    logits_2d: mx.array,
    *,
    width: int,
) -> tuple[list[list[int]], list[list[float]]]:
    top_width = int(width)
    if top_width <= 0:
        raise ValueError("width must be positive")
    top = mx.argpartition(logits_2d, kth=-top_width, axis=-1)[:, -top_width:]
    top_logits = mx.take_along_axis(logits_2d, top, axis=-1)
    order = mx.argsort(top_logits, axis=-1)[:, ::-1]
    top = mx.take_along_axis(top, order, axis=-1)
    log_probs = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
    values = mx.take_along_axis(log_probs, top, axis=-1)
    mx.eval(top, values)
    return (
        [[int(token) for token in row] for row in top.tolist()],
        [[float(value) for value in row] for row in values.tolist()],
    )


def draft_block_with_topk(
    *,
    target_model: Any,
    target_ops: Any,
    draft_model: Any,
    draft_cache: list[Any],
    prefix_tokens: mx.array,
    draft_context: mx.array,
    block_len: int,
    suppress_token_mask: mx.array | None,
    top_width: int,
) -> tuple[mx.array, list[list[int]], list[list[float]], float]:
    if int(prefix_tokens.shape[0]) > int(block_len):
        raise ValueError("prefix_tokens cannot be longer than block_len")
    draft_start = time.perf_counter_ns()
    pad_len = int(block_len) - int(prefix_tokens.shape[0])
    if pad_len > 0:
        mask_tail = mx.full((pad_len,), int(draft_model.mask_token_id), dtype=mx.uint32)
        block_token_ids = mx.concatenate([prefix_tokens, mask_tail], axis=0)
    else:
        block_token_ids = prefix_tokens

    draft_dtype = _draft_compute_dtype(draft_model)
    noise_embedding = target_ops.embed_tokens(target_model)(block_token_ids[None])
    if draft_dtype is not None:
        noise_embedding = _astype_if_needed(noise_embedding, draft_dtype)
        draft_context = _astype_if_needed(draft_context, draft_dtype)
    draft_hidden = draft_model.forward_projected_context(
        noise_embedding=noise_embedding,
        draft_context=draft_context,
        cache=draft_cache,
    )
    draft_logits = target_ops.logits_from_hidden(target_model, draft_hidden[:, 1:, :])
    if int(draft_logits.shape[1]) <= 0:
        raise RuntimeError(
            "draft logits are empty: "
            f"block_len={int(block_len)} "
            f"prefix_len={int(prefix_tokens.shape[0])} "
            f"block_token_shape={tuple(int(x) for x in block_token_ids.shape)} "
            f"draft_hidden_shape={tuple(int(x) for x in draft_hidden.shape)}"
        )
    masked_logits = draft_logits
    if suppress_token_mask is not None:
        floor = mx.array(-1e9, dtype=draft_logits.dtype)
        masked_logits = mx.where(suppress_token_mask, floor, draft_logits)
    drafted_all = greedy_tokens_with_mask(draft_logits, suppress_token_mask).squeeze(0)
    top_ids, top_values = top_ids_and_values_desc(
        masked_logits.squeeze(0),
        width=top_width,
    )
    mx.eval(drafted_all)
    return (
        drafted_all,
        top_ids,
        top_values,
        (time.perf_counter_ns() - draft_start) / 1_000.0,
    )


def draft_branch_blocks_batch(
    *,
    target_model: Any,
    target_ops: Any,
    draft_model: Any,
    draft_cache: list[Any],
    branch_prefixes: list[mx.array],
    draft_context: mx.array,
    block_len: int,
    suppress_token_mask: mx.array | None,
) -> tuple[list[mx.array], float]:
    if not branch_prefixes:
        return [], 0.0
    draft_start = time.perf_counter_ns()
    rows: list[mx.array] = []
    for prefix_tokens in branch_prefixes:
        pad_len = int(block_len) - int(prefix_tokens.shape[0])
        if pad_len > 0:
            mask_tail = mx.full((pad_len,), int(draft_model.mask_token_id), dtype=mx.uint32)
            rows.append(mx.concatenate([prefix_tokens, mask_tail], axis=0))
        else:
            rows.append(prefix_tokens[:block_len])
    block_token_ids = mx.stack(rows, axis=0)
    batch_size = int(block_token_ids.shape[0])
    draft_dtype = _draft_compute_dtype(draft_model)
    noise_embedding = target_ops.embed_tokens(target_model)(block_token_ids)
    batch_context = mx.repeat(draft_context, batch_size, axis=0)
    if draft_dtype is not None:
        noise_embedding = _astype_if_needed(noise_embedding, draft_dtype)
        batch_context = _astype_if_needed(batch_context, draft_dtype)
    draft_hidden = draft_model.forward_projected_context(
        noise_embedding=noise_embedding,
        draft_context=batch_context,
        cache=draft_cache,
    )
    draft_logits = target_ops.logits_from_hidden(target_model, draft_hidden[:, 1:, :])
    drafted_all = greedy_tokens_with_mask(draft_logits, suppress_token_mask)
    mx.eval(drafted_all)
    candidate_ids: list[mx.array] = []
    for row_index, prefix_tokens in enumerate(branch_prefixes):
        suffix_start = int(prefix_tokens.shape[0]) - 1
        candidate_ids.append(
            candidate_token_ids(
                prefix_tokens=prefix_tokens,
                suffix_tokens=drafted_all[row_index, suffix_start:],
                block_len=block_len,
            )
        )
    return candidate_ids, (time.perf_counter_ns() - draft_start) / 1_000.0


def verify_candidates_batch(
    *,
    target_model: Any,
    target_ops: Any,
    target_cache: list[Any],
    capture_layer_ids: set[int],
    candidate_ids: list[mx.array],
    candidate_sources: list[str],
    suppress_token_mask: mx.array | None,
    prefix_len: int,
) -> tuple[list[DDTreeCandidateResult], float]:
    if not candidate_ids:
        raise ValueError("candidate_ids must not be empty")
    if len(candidate_ids) != len(candidate_sources):
        raise ValueError("candidate_ids and candidate_sources must have equal length")
    verify_ids = mx.stack(candidate_ids, axis=0)
    target_ops.arm_rollback(target_cache, prefix_len=int(prefix_len))
    verify_start = time.perf_counter_ns()
    logits, hidden_states = target_ops.verify_block(
        target_model=target_model,
        verify_ids=verify_ids,
        target_cache=target_cache,
        capture_layer_ids=capture_layer_ids,
    )
    eval_logits_and_captured(logits, hidden_states)
    posterior = greedy_tokens_with_mask(logits, suppress_token_mask)
    mx.eval(posterior)
    verify_us = (time.perf_counter_ns() - verify_start) / 1_000.0
    results: list[DDTreeCandidateResult] = []
    for index, ids in enumerate(candidate_ids):
        row_posterior = posterior[index]
        acceptance_len = int(
            match_acceptance_length(ids[1:], row_posterior[:-1]).item()
        )
        results.append(
            DDTreeCandidateResult(
                source=candidate_sources[index],
                ids=ids,
                posterior=row_posterior,
                logits=logits[index : index + 1],
                hidden_states=select_captured(hidden_states, index),
                acceptance_len=acceptance_len,
                verify_us=verify_us,
            )
        )
    return results, verify_us

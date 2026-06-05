# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from dflash_mlx.cache.codecs import PrefixSnapshotBuilder
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import TargetHiddenChunks
from dflash_mlx.engine.target_features import TargetFeatureStore
from dflash_mlx.model import ContextOnlyDraftKVCache, DFlashAttention


def _key() -> DFlashPrefixKey:
    return DFlashPrefixKey(
        target_model_id="target",
        draft_model_id="draft",
        capture_layer_ids=(1,),
        draft_sink_size=2,
        draft_window_size=4,
        template_hash="template",
        prompt_policy_hash="policy",
        target_fa_window=0,
    )


def _snapshot(prompt_len: int = 12):
    sink = mx.ones((1, 2, 3), dtype=mx.float32)
    tail = mx.ones((1, 4, 3), dtype=mx.float32) * 2
    return SimpleNamespace(
        target_hidden_chunks=(sink, tail),
        target_hidden_chunk_spans=((0, 2), (prompt_len - 4, prompt_len)),
    )


def test_exact_hit_generation_snapshot_stays_chunked_without_materialize():
    prompt_len = 12
    store = TargetFeatureStore(prompt_len=prompt_len)
    hidden = store.hydrate_from_snapshot(
        _snapshot(prompt_len),
        snap_prefix_len=prompt_len,
    )
    assert isinstance(hidden, TargetHiddenChunks)

    with patch.object(
        TargetHiddenChunks,
        "materialize",
        side_effect=AssertionError("exact-hit path materialized target hidden"),
    ):
        store.freeze_prefill_for_snapshot(enabled=True)
        generated = mx.ones((1, 3, 3), dtype=mx.float32) * 3
        store.commit_generation(generated, collect_snapshot=True)
        snapshot_hidden = store.generation_snapshot_hidden()
        assert isinstance(snapshot_hidden, TargetHiddenChunks)

        built = PrefixSnapshotBuilder(
            key=_key(),
            draft_sink_size=2,
            draft_window_size=4,
        ).build(
            token_ids=list(range(prompt_len + 3)),
            target_cache=[],
            target_hidden=snapshot_hidden,
            last_logits=None,
            kind="generation",
        )

    assert built.target_hidden_chunk_spans == ((0, 2), (prompt_len - 1, prompt_len + 3))
    np.testing.assert_array_equal(
        np.array(built.target_hidden_chunks[0]),
        np.ones((1, 2, 3), dtype=np.float32),
    )
    expected_tail = np.concatenate(
        [
            np.full((1, 1, 3), 2, dtype=np.float32),
            np.full((1, 3, 3), 3, dtype=np.float32),
        ],
        axis=1,
    )
    np.testing.assert_array_equal(
        np.array(built.target_hidden_chunks[1]),
        expected_tail,
    )


def test_chunked_context_feeds_sink_and_tail_without_materialize():
    prompt_len = 12
    hidden = TargetHiddenChunks(
        total_len=prompt_len,
        chunks=(
            mx.ones((1, 2, 3), dtype=mx.float32),
            mx.ones((1, 4, 3), dtype=mx.float32) * 2,
        ),
        spans=((0, 2), (8, 12)),
    )
    cache = ContextOnlyDraftKVCache(sink_size=2, window_size=4)

    with patch.object(
        TargetHiddenChunks,
        "materialize",
        side_effect=AssertionError("draft context materialized target hidden"),
    ):
        selected, spans = DFlashAttention._context_segments_for_cache(
            None,
            hidden,
            cache,
        )

    assert spans == [(0, 2), (8, 12)]
    assert selected.shape == (1, 6, 3)
    np.testing.assert_array_equal(
        np.array(selected[:, :2, :]),
        np.ones((1, 2, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.array(selected[:, 2:, :]),
        np.full((1, 4, 3), 2, dtype=np.float32),
    )


def test_partial_hit_keeps_existing_dense_restore_behavior():
    store = TargetFeatureStore(prompt_len=14)
    restored = store.hydrate_from_snapshot(_snapshot(12), snap_prefix_len=12)
    assert isinstance(restored, mx.array)
    assert restored.shape == (1, 14, 3)

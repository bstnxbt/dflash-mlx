# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Gemma4 sliding-window support for positional sparse prefill.

Gemma4 alternates sliding-window and full-attention layers; the sliding mask
must use the selected tokens' *true* positions, not their compacted indices.
These tests build a tiny real mlx-lm gemma4 model in-memory (no download), so
they run in CI: select-all sparse prefill must be bitwise-identical to dense,
and the sliding mask must exclude keys outside the true-position window.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.base import create_causal_mask

from dflash_mlx.engine import target_gemma4
from dflash_mlx.engine.sparse_rope import (
    clear_sparse_positions,
    install_position_mapped_rope,
    restore_ropes,
    set_sparse_positions,
)
from dflash_mlx.engine.target_ops import resolve_target_ops


def _tiny_gemma4():
    from mlx_lm.models.gemma4_text import Model, ModelArgs

    args = ModelArgs(
        model_type="gemma4_text", hidden_size=64, num_hidden_layers=6,
        intermediate_size=128, num_attention_heads=2, head_dim=32,
        global_head_dim=64, global_partial_rotary_factor=0.25,
        num_key_value_heads=1, num_kv_shared_layers=2,
        hidden_size_per_layer_input=16, vocab_size=128,
        vocab_size_per_layer_input=128, sliding_window=4, sliding_window_pattern=3,
        partial_rotary_factor=1.0, max_position_embeddings=4096,
        final_logit_softcapping=30.0, tie_word_embeddings=True,
        use_double_wide_mlp=False,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _forward_logits(ops, model, ids):
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    logits, _ = ops.forward_with_hidden_capture(
        model, input_ids=ids, cache=cache, logits_last_only=True
    )
    mx.eval(logits)
    return logits


class TestSparseLayerMasks:
    def test_select_all_sliding_mask_matches_dense(self):
        model = _tiny_gemma4()
        inner = resolve_target_ops(model).text_model(model)
        n = 10
        h = mx.zeros((1, n, inner.layers[0].self_attn.head_dim))
        positions = mx.arange(0, n, dtype=mx.int32)
        masks = target_gemma4._sparse_layer_masks(inner, h, positions, [None] * len(inner.layers))
        sliding = next(
            m for layer, m in zip(inner.layers, masks)
            if layer.layer_type == "sliding_attention"
        )
        dense = create_causal_mask(n, window_size=inner.window_size)
        assert mx.array_equal(sliding, dense)

    def test_sliding_mask_excludes_far_true_positions(self):
        model = _tiny_gemma4()
        inner = resolve_target_ops(model).text_model(model)
        window = inner.window_size  # 4
        # two chunks with a gap: indices 3..5 are at positions 7,8,9
        positions = mx.array([0, 1, 2, 7, 8, 9], dtype=mx.int32)
        h = mx.zeros((1, 6, 8))
        masks = target_gemma4._sparse_layer_masks(inner, h, positions, [None] * len(inner.layers))
        sliding = next(
            m for layer, m in zip(inner.layers, masks)
            if layer.layer_type == "sliding_attention"
        )
        row = sliding[3].tolist()  # query at true position 7, window 4
        # pos 7 attends keys with 7 - kpos < 4  ->  kpos in {7} among selected
        # (positions 0,1,2 are >4 away and must be masked out)
        assert row == [False, False, False, True, False, False]


class TestGemma4ForwardParity:
    def test_select_all_forward_matches_dense(self):
        model = _tiny_gemma4()
        ops = resolve_target_ops(model)
        inner = ops.text_model(model)
        ids = mx.array([[3, 9, 15, 22, 30, 41, 55, 60, 71, 88]], dtype=mx.uint32)
        n = ids.shape[1]

        dense = _forward_logits(ops, model, ids)
        positions = mx.arange(0, n, dtype=mx.int32)
        saved = install_position_mapped_rope(inner, positions)
        set_sparse_positions(inner, positions)
        try:
            sparse = _forward_logits(ops, model, ids)
        finally:
            restore_ropes(saved)
            clear_sparse_positions(inner)

        diff = float(mx.abs(dense.astype(mx.float32) - sparse.astype(mx.float32)).max())
        assert diff == 0.0, f"gemma4 select-all diverged from dense: {diff}"

    def test_noncontiguous_forward_runs_and_is_finite(self):
        model = _tiny_gemma4()
        ops = resolve_target_ops(model)
        inner = ops.text_model(model)
        keep = [0, 1, 2, 7, 8, 9]
        ids = mx.array([[5, 11, 19, 44, 51, 63]], dtype=mx.uint32)
        positions = mx.array(keep, dtype=mx.int32)
        saved = install_position_mapped_rope(inner, positions)
        set_sparse_positions(inner, positions)
        try:
            logits = _forward_logits(ops, model, ids)
        finally:
            restore_ropes(saved)
            clear_sparse_positions(inner)
        assert bool(mx.all(mx.isfinite(logits.astype(mx.float32))))

    def test_restore_returns_native_trajectory(self):
        model = _tiny_gemma4()
        ops = resolve_target_ops(model)
        inner = ops.text_model(model)
        ids = mx.array([[3, 9, 15, 22, 30, 41]], dtype=mx.uint32)
        before = _forward_logits(ops, model, ids)
        positions = mx.arange(0, ids.shape[1], dtype=mx.int32)
        saved = install_position_mapped_rope(inner, positions)
        set_sparse_positions(inner, positions)
        restore_ropes(saved)
        clear_sparse_positions(inner)
        after = _forward_logits(ops, model, ids)
        assert float(mx.abs(before.astype(mx.float32) - after.astype(mx.float32)).max()) == 0.0

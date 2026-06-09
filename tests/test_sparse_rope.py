# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file

"""Numerical parity tests for the sparse-prefill RoPE helpers.

These validate that applying RoPE at explicit positions matches mlx's built-in
``nn.RoPE`` for contiguous positions (the convention anchor) and composes
correctly for non-contiguous positions. Model-free and fast.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from dflash_mlx.engine.sparse_rope import (
    OffsetAdjustedRoPE,
    PositionMappedRoPE,
    decode_position_adjustment,
    install_position_mapped_rope,
    iter_attention_modules,
    manual_rope,
    restore_ropes,
    switch_to_offset_adjusted_rope,
)

_ATOL = 2e-4


def _rope(dims=128, base=1_000_000.0):
    return nn.RoPE(dims=dims, traditional=False, base=base)


class TestManualRope:
    def test_matches_real_rope_contiguous(self):
        # The convention anchor: explicit contiguous positions == nn.RoPE(offset).
        dims, base = 128, 1_000_000.0
        rope = _rope(dims, base)
        x = mx.random.normal((1, 4, 6, dims))
        offset = 3
        real = rope(x, offset=offset)
        manual = manual_rope(x, mx.arange(offset, offset + 6), dims=dims, base=base)
        assert mx.allclose(real, manual, atol=_ATOL)

    def test_single_token_at_arbitrary_position(self):
        dims, base = 64, 10_000.0
        rope = _rope(dims, base)
        x = mx.random.normal((1, 2, 1, dims))
        # Large positions accumulate float32 trig range-reduction error
        # (angle ~= position radians); mlx's own RoPE shares this float32 limit
        # and the downstream bf16 attention tolerates ~8e-3. Scale tolerance with
        # position rather than pretending float32 trig is exact at 4096 rad.
        for p in (0, 5, 37, 128, 4096):
            real = rope(x, offset=p)
            manual = manual_rope(x, mx.array([p]), dims=dims, base=base)
            atol = _ATOL if p <= 1024 else 2e-3
            assert mx.allclose(real, manual, atol=atol), f"mismatch at position {p}"

    def test_partial_rotary_dims_passthrough(self):
        # dims < head_dim: the tail dimensions must pass through untouched.
        head_dim, dims, base = 96, 64, 10_000.0
        rope = _rope(dims, base)
        x = mx.random.normal((1, 2, 5, head_dim))
        real = rope(x, offset=2)
        manual = manual_rope(x, mx.arange(2, 7), dims=dims, base=base)
        assert mx.allclose(real, manual, atol=_ATOL)

    def test_preserves_bfloat16_dtype(self):
        # The downstream SDPA contract requires bf16/fp16 in == out.
        dims, base = 128, 1_000_000.0
        rope = _rope(dims, base)
        x = mx.random.normal((1, 4, 6, dims)).astype(mx.bfloat16)
        manual = manual_rope(x, mx.arange(0, 6), dims=dims, base=base)
        assert manual.dtype == mx.bfloat16
        real = rope(x, offset=0)
        assert mx.allclose(real.astype(mx.float32), manual.astype(mx.float32), atol=8e-3)


class _FakeAttn:
    def __init__(self, rope):
        self.rope = rope


class _FakeLayer:
    def __init__(self, *, is_linear, rope=None):
        self.is_linear = is_linear
        if not is_linear:
            self.self_attn = _FakeAttn(rope)


class _FakeTextModel:
    def __init__(self, layers):
        self.layers = layers


class TestRopeLifecycle:
    def _model(self):
        rope = _rope()
        # 2 full-attention layers (have rope), 1 GDN layer (no rope), interleaved.
        return _FakeTextModel(
            [
                _FakeLayer(is_linear=False, rope=_rope()),
                _FakeLayer(is_linear=True),
                _FakeLayer(is_linear=False, rope=_rope()),
            ]
        ), rope

    def test_iter_attention_modules_skips_gdn_layers(self):
        model, _ = self._model()
        attns = list(iter_attention_modules(model))
        assert len(attns) == 2  # the GDN layer is skipped

    def test_install_switch_restore_roundtrip(self):
        model, _ = self._model()
        originals = [layer.self_attn.rope for layer in model.layers if not layer.is_linear]

        saved = install_position_mapped_rope(model, mx.array([0, 1, 2]), cache_start=0)
        installed = [a.rope for a in iter_attention_modules(model)]
        assert all(isinstance(r, PositionMappedRoPE) for r in installed)

        switch_to_offset_adjusted_rope(saved, adjustment=5)
        switched = [a.rope for a in iter_attention_modules(model)]
        assert all(isinstance(r, OffsetAdjustedRoPE) for r in switched)

        restore_ropes(saved)
        restored = [a.rope for a in iter_attention_modules(model)]
        assert restored == originals  # exact original objects, not copies

    def test_switch_wraps_original_not_prefill_wrapper(self):
        # OffsetAdjustedRoPE must compose over the original rope, not over the
        # PositionMappedRoPE installed during prefill.
        model, _ = self._model()
        originals = [layer.self_attn.rope for layer in model.layers if not layer.is_linear]
        saved = install_position_mapped_rope(model, mx.array([0, 1, 2]))
        switch_to_offset_adjusted_rope(saved, adjustment=3)
        for attn, original in zip(iter_attention_modules(model), originals):
            assert attn.rope._original is original

    def test_install_failure_restores_already_swapped_ropes(self):
        first = _rope()
        unsupported = nn.RoPE(dims=64, traditional=True, base=10_000.0)
        model = _FakeTextModel(
            [
                _FakeLayer(is_linear=False, rope=first),
                _FakeLayer(is_linear=False, rope=unsupported),
            ]
        )

        with pytest.raises(NotImplementedError, match="traditional"):
            install_position_mapped_rope(model, mx.array([0, 1, 2]))

        assert model.layers[0].self_attn.rope is first
        assert model.layers[1].self_attn.rope is unsupported


class TestDecodePositionAdjustment:
    def test_dense_select_all_is_zero(self):
        # Select-all (dense) must make the decode wrapper a no-op.
        assert decode_position_adjustment(tuple(range(10))) == 0

    def test_sparse_subset_shift(self):
        # 4 selected tokens, last at original position 99 -> first decode at 100,
        # cache holds 4 entries -> adjustment 100 - 4 = 96.
        assert decode_position_adjustment((0, 40, 70, 99)) == 96

    def test_empty(self):
        assert decode_position_adjustment(()) == 0


class TestPositionMappedRoPE:
    def test_applies_each_token_at_its_sparse_position(self):
        dims, base = 128, 1_000_000.0
        rope = _rope(dims, base)
        positions = mx.array([0, 5, 9, 20])
        wrapped = PositionMappedRoPE(rope, positions, cache_start=0)
        x = mx.random.normal((1, 4, 4, dims))
        out = wrapped(x, offset=0)
        for i, p in enumerate((0, 5, 9, 20)):
            ref = rope(x[:, :, i : i + 1, :], offset=int(p))
            assert mx.allclose(out[:, :, i : i + 1, :], ref, atol=_ATOL)

    def test_offset_indexes_into_position_array(self):
        # A later chunk: offset=2 selects positions[2:4].
        dims, base = 64, 10_000.0
        rope = _rope(dims, base)
        positions = mx.array([0, 3, 11, 40])
        wrapped = PositionMappedRoPE(rope, positions, cache_start=0)
        x = mx.random.normal((1, 2, 2, dims))
        out = wrapped(x, offset=2)
        for i, p in enumerate((11, 40)):
            ref = rope(x[:, :, i : i + 1, :], offset=int(p))
            assert mx.allclose(out[:, :, i : i + 1, :], ref, atol=_ATOL)


class TestOffsetAdjustedRoPE:
    def test_adds_constant_adjustment(self):
        dims, base = 64, 10_000.0
        rope = _rope(dims, base)
        wrapped = OffsetAdjustedRoPE(rope, adjustment=7)
        x = mx.random.normal((1, 2, 3, dims))
        out = wrapped(x, offset=2)
        ref = rope(x, offset=2 + 7)
        assert mx.allclose(out, ref, atol=_ATOL)


class TestTraditionalRopeRejected:
    def test_manual_rope_rejects_unsupported_via_wrapper(self):
        # Interleaved (traditional) RoPE is a different convention; reject it
        # rather than silently producing wrong rotations.
        trad = nn.RoPE(dims=64, traditional=True, base=10_000.0)
        with pytest.raises(NotImplementedError, match="traditional"):
            PositionMappedRoPE(trad, mx.array([0, 1]), cache_start=0)

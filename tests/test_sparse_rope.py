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
    manual_rope,
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

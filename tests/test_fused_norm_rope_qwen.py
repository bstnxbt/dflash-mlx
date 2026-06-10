# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Parity tests for fused_norm_rope_qwen kernel vs reference path."""

from __future__ import annotations

import pytest

import mlx.core as mx
import mlx.nn as nn

from dflash_mlx.engine.fused_norm_rope_qwen import (
    fused_norm_rope_qwen,
    is_fused_norm_rope_qwen_eligible,
    make_qwen_cos_sin,
)

D_HEAD = 256
D_ROPE = 64
EPS = 1e-6
ROPE_THETA_QWEN36 = 10000000.0

requires_metal = pytest.mark.skipif(
    not mx.metal.is_available(),
    reason="executes Metal kernels; requires a Metal GPU",
)


def _reference_norm_rope(
    x: mx.array, weight: mx.array, offset: int, rope_theta: float
) -> mx.array:
    """nn.RMSNorm + transpose + nn.RoPE reference (matches Qwen3NextAttention)."""
    norm = nn.RMSNorm(D_HEAD, eps=EPS)
    norm.weight = weight
    rope = nn.RoPE(D_ROPE, traditional=False, base=rope_theta)
    return rope(norm(x).transpose(0, 2, 1, 3), offset=offset)


def _reference_norm_rope_positions(
    x: mx.array,
    weight: mx.array,
    positions: list[int],
    rope_theta: float,
) -> mx.array:
    """nn.RMSNorm + transpose + per-token nn.RoPE for non-uniform positions."""
    norm = nn.RMSNorm(D_HEAD, eps=EPS)
    norm.weight = weight
    rope = nn.RoPE(D_ROPE, traditional=False, base=rope_theta)
    normed = norm(x).transpose(0, 2, 1, 3)
    if all(p == positions[0] for p in positions):
        return rope(normed, offset=positions[0])
    chunks = [
        rope(normed[:, :, i : i + 1, :], offset=int(positions[i]))
        for i in range(len(positions))
    ]
    return mx.concatenate(chunks, axis=2)


def _build_inputs(B: int, L: int, H: int, *, dtype: mx.Dtype = mx.bfloat16, seed: int = 0):
    mx.random.seed(seed)
    x = (mx.random.normal((B, L, H, D_HEAD)) * 0.5).astype(dtype)
    weight = (mx.random.normal((D_HEAD,)) * 0.1 + 1.0).astype(dtype)
    mx.eval(x, weight)
    return x, weight


@requires_metal
@pytest.mark.parametrize(
    "B,L,H,offset",
    [
        (1, 16, 24, 0),
        (1, 16, 4, 0),
        (1, 16, 24, 4096),
        (1, 16, 4, 8192),
        (1, 4, 24, 1024),
        (1, 4, 4, 0),
        (2, 16, 24, 256),
    ],
)
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_parity_uniform_offset(B: int, L: int, H: int, offset: int, dtype: mx.Dtype) -> None:
    x, weight = _build_inputs(B, L, H, dtype=dtype)
    positions = mx.arange(offset, offset + L, dtype=mx.int32)
    cos, sin = make_qwen_cos_sin(positions, rope_theta=ROPE_THETA_QWEN36)
    mx.eval(cos, sin)

    ref = _reference_norm_rope(x, weight, offset, rope_theta=ROPE_THETA_QWEN36)
    out = fused_norm_rope_qwen(x, weight, cos, sin, eps=EPS)
    mx.eval(ref, out)

    rf = ref.astype(mx.float32)
    of = out.astype(mx.float32)
    diff = mx.abs(rf - of)
    rmax = float(mx.max(mx.abs(rf)).item())
    max_rel = float(mx.max(diff).item()) / (rmax + 1e-6)
    assert max_rel < 0.01, (
        f"parity drift: max_rel={max_rel:.3e}"
        f" (B={B}, L={L}, H={H}, off={offset}, dtype={dtype})"
    )


@requires_metal
@pytest.mark.parametrize(
    "positions_list,H",
    [
        ([100, 101, 102, 103, 104, 105, 106, 107], 24),
        ([0, 5, 10, 15, 20, 25, 30, 35], 4),
        ([8192, 8193, 8194, 8195], 24),
    ],
)
def test_parity_nonuniform_positions(positions_list: list[int], H: int) -> None:
    L = len(positions_list)
    x, weight = _build_inputs(1, L, H, dtype=mx.bfloat16)
    positions = mx.array(positions_list, dtype=mx.int32)
    cos, sin = make_qwen_cos_sin(positions, rope_theta=ROPE_THETA_QWEN36)
    mx.eval(cos, sin)

    ref = _reference_norm_rope_positions(
        x, weight, positions_list, rope_theta=ROPE_THETA_QWEN36
    )
    out = fused_norm_rope_qwen(x, weight, cos, sin, eps=EPS)
    mx.eval(ref, out)

    rf = ref.astype(mx.float32)
    of = out.astype(mx.float32)
    diff = mx.abs(rf - of)
    rmax = float(mx.max(mx.abs(rf)).item())
    max_rel = float(mx.max(diff).item()) / (rmax + 1e-6)
    assert max_rel < 0.01, f"parity drift on non-uniform positions: max_rel={max_rel:.3e}"


def test_eligibility_gates() -> None:
    """The gate should accept the Qwen3.x prefill shape and reject everything else."""
    assert is_fused_norm_rope_qwen_eligible(
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_traditional=False,
        q_len=16,
        dtype=mx.bfloat16,
    )
    assert is_fused_norm_rope_qwen_eligible(
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_traditional=False,
        q_len=4,
        dtype=mx.float16,
    )
    assert not is_fused_norm_rope_qwen_eligible(
        head_dim=128,
        partial_rotary_factor=0.25,
        rope_traditional=False,
        q_len=16,
        dtype=mx.bfloat16,
    )
    assert not is_fused_norm_rope_qwen_eligible(
        head_dim=256,
        partial_rotary_factor=0.5,
        rope_traditional=False,
        q_len=16,
        dtype=mx.bfloat16,
    )
    assert not is_fused_norm_rope_qwen_eligible(
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_traditional=True,
        q_len=16,
        dtype=mx.bfloat16,
    )
    assert not is_fused_norm_rope_qwen_eligible(
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_traditional=False,
        q_len=1,
        dtype=mx.bfloat16,
    )
    assert not is_fused_norm_rope_qwen_eligible(
        head_dim=256,
        partial_rotary_factor=0.25,
        rope_traditional=False,
        q_len=16,
        dtype=mx.float32,
    )


@requires_metal
def test_rejects_wrong_shape() -> None:
    """The kernel call must validate D and cos/sin shapes."""
    x, weight = _build_inputs(1, 16, 24)
    cos = mx.zeros((16, 32), dtype=mx.float32)
    sin = mx.zeros((16, 32), dtype=mx.float32)

    bad_x = x.reshape(1, 16, 48, 128)
    with pytest.raises(ValueError, match="D=256"):
        fused_norm_rope_qwen(bad_x, weight, cos, sin)

    with pytest.raises(ValueError, match="weight must be"):
        fused_norm_rope_qwen(x, weight[:128], cos, sin)

    with pytest.raises(ValueError, match="cos/sin"):
        fused_norm_rope_qwen(x, weight, cos[:8], sin)

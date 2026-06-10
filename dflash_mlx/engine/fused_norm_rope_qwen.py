# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Fused RMSNorm + partial-RoPE + transpose kernel for Qwen3.x attention.

Replaces the three-call sequence ``q_norm(x).transpose(0, 2, 1, 3)`` ->
``rope(., offset=...)`` in :class:`Qwen3NextAttention` with a single Metal
dispatch. Hard-coded to ``head_dim=256`` and ``partial_rotary_factor=0.25``
(``D_ROPE=64``) which are the Qwen3.x text-model defaults.

The kernel uses a strided lane layout — lane ``t`` holds elements
``{t, t+32, ..., t+224}`` of one row. The RoPE pair ``(t, t+32)`` lives in
the same lane (``v[0]``, ``v[1]``), eliminating the cross-lane tg-mem
shuffle that earlier blocked-layout drafts required.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

_D_HEAD = 256
_D_ROPE = 64
_HALF = 32
_RMS_NORM_EPS_DEFAULT = 1e-6

_KERNEL_CACHE: dict = {}


def _build_kernel(dtype: mx.Dtype, eps: float):
    key = ("v2_strided", dtype, eps)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    source = """
        using namespace metal;
        constexpr int D     = 256;
        constexpr int HALF  = 32;
        constexpr int THR   = 32;
        constexpr int ELTS  = 8;

        uint tg_b = threadgroup_position_in_grid.y;
        uint tg_l = threadgroup_position_in_grid.z;
        uint lane = thread_position_in_threadgroup.x;

        int H = int(H_size);
        int L = int(L_size);
        int b = int(tg_b) / H;
        int h = int(tg_b) % H;
        int l = int(tg_l);

        int in_off  = ((b * L + l) * H + h) * D;
        int out_off = ((b * H + h) * L + l) * D;

        float v[ELTS];
        float w[ELTS];
        for (int i = 0; i < ELTS; ++i) {
            v[i] = float(x[in_off + i * THR + lane]);
            w[i] = float(weight[i * THR + lane]);
        }

        float local_sq = 0.0f;
        for (int i = 0; i < ELTS; ++i) {
            local_sq += v[i] * v[i];
        }
        float total_sq = simd_sum(local_sq);
        float inv_rms = rsqrt(total_sq / float(D) + EPS_CONST);

        for (int i = 0; i < ELTS; ++i) {
            v[i] = v[i] * inv_rms * w[i];
        }

        float c = cos[l * HALF + lane];
        float s = sin[l * HALF + lane];
        float low  = v[0];
        float high = v[1];
        v[0] = low * c - high * s;
        v[1] = low * s + high * c;

        for (int i = 0; i < ELTS; ++i) {
            y[out_off + i * THR + lane] = T(v[i]);
        }
    """
    source = source.replace("EPS_CONST", f"{eps:.7e}f")
    dtype_tag = "bf16" if dtype == mx.bfloat16 else "fp16" if dtype == mx.float16 else "unk"
    kernel = mx.fast.metal_kernel(
        name=f"fused_norm_rope_qwen_{dtype_tag}_eps{int(eps * 1e9)}",
        input_names=["x", "weight", "cos", "sin", "H_size", "L_size"],
        output_names=["y"],
        source=source,
    )
    _KERNEL_CACHE[key] = kernel
    return kernel


def fused_norm_rope_qwen(
    x: mx.array,
    weight: mx.array,
    cos: mx.array,
    sin: mx.array,
    *,
    eps: float = _RMS_NORM_EPS_DEFAULT,
) -> mx.array:
    """Apply RMSNorm + partial RoPE + transpose in one Metal dispatch.

    Args:
        x: ``(B, L, H, 256)`` activations after ``q_proj`` or ``k_proj``.
        weight: ``(256,)`` RMSNorm ``gamma``.
        cos: ``(L, 32)`` per-row RoPE cosines (one row per token position).
        sin: ``(L, 32)`` per-row RoPE sines.
        eps: RMSNorm epsilon. Defaults to Qwen3.x ``1e-6``.

    Returns:
        ``(B, H, L, 256)`` — RMSNorm'd + partial-RoPE'd + transposed.

    Raises:
        ValueError: when ``D != 256`` or shapes mismatch.
    """
    if x.ndim != 4:
        raise ValueError(f"x must be 4-D (B, L, H, D), got shape {x.shape}")
    B, L, H, D = x.shape
    if D != _D_HEAD:
        raise ValueError(f"fused_norm_rope_qwen requires D={_D_HEAD}, got {D}")
    if int(weight.shape[0]) != _D_HEAD:
        raise ValueError(f"weight must be (256,), got {weight.shape}")
    if cos.shape != (L, _HALF) or sin.shape != (L, _HALF):
        raise ValueError(
            f"cos/sin must be (L={L}, {_HALF}); got cos={cos.shape}, sin={sin.shape}"
        )

    kernel = _build_kernel(x.dtype, eps)
    # mx.fast.metal_kernel grid is TOTAL THREADS (dispatchThreads semantics),
    # not threadgroup count: (32, B*H, L) with threadgroup (32, 1, 1) yields
    # exactly one 32-lane simdgroup per (b, h, l) row. A (B*H, L, 1) grid
    # silently runs lane 0 only and zero-fills the rest.
    (y,) = kernel(
        inputs=[x, weight, cos, sin, H, L],
        template=[("T", x.dtype)],
        grid=(32, B * H, L),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, H, L, D)],
        output_dtypes=[x.dtype],
    )
    return y


def make_qwen_cos_sin(
    positions: mx.array,
    rope_theta: float,
    *,
    dtype: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array]:
    """Build ``(cos, sin)`` of shape ``(L, 32)`` for the given positions.

    Args:
        positions: ``(L,)`` int array of per-token positions.
        rope_theta: base ``theta`` (``rope_parameters.rope_theta`` in HF config).
        dtype: dtype of returned cos/sin (kernel reads as float).
    """
    # Matches mx.fast.rope freqs for dims=64: theta ** (-2i/64) == theta ** (-i/32).
    freqs = mx.array(
        [rope_theta ** (-i / _HALF) for i in range(_HALF)],
        dtype=mx.float32,
    )
    pos = positions.astype(mx.float32)[:, None]
    angles = pos * freqs[None, :]
    return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def is_fused_norm_rope_qwen_eligible(
    head_dim: int,
    partial_rotary_factor: float,
    rope_traditional: bool,
    q_len: int,
    dtype: mx.Dtype,
) -> bool:
    """Gate for the fused kernel — keep the reference path for everything else.

    Returns ``True`` only when the kernel's hard-coded contract matches:
    ``head_dim=256``, ``D_ROPE=64`` (factor 0.25), non-traditional split RoPE,
    a prefill-sized ``q_len >= 4`` window, and a supported dtype.
    """
    if head_dim != _D_HEAD:
        return False
    if int(round(head_dim * partial_rotary_factor)) != _D_ROPE:
        return False
    if rope_traditional:
        return False
    if q_len < 4:
        return False
    return dtype in (mx.bfloat16, mx.float16)

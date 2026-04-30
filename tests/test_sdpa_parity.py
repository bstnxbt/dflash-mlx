# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import math

import mlx.core as mx
import pytest
from mlx_lm.models.base import scaled_dot_product_attention

from dflash_mlx.kernels import batched_sdpa_2pass_exact

SHAPES = [
    ("qwen36_27b", 24, 4, 256),
    ("qwen36_35b_a3b", 16, 2, 256),
]
KV_LENGTHS = [1024, 4096, 16384]
DTYPES = [mx.bfloat16, mx.float16]
MASK_KINDS = ["implicit_causal", "explicit_causal", "swa"]

def _sync() -> None:
    if hasattr(mx, "synchronize"):
        mx.synchronize()

def _rand(shape: tuple[int, ...], dtype: mx.Dtype) -> mx.array:
    return (mx.random.normal(shape) * 0.02).astype(dtype)

def _additive_mask(mask: mx.array, dtype: mx.Dtype) -> mx.array:
    zero = mx.zeros(mask.shape, dtype=dtype)
    neg = mx.full(mask.shape, mx.finfo(dtype).min, dtype=dtype)
    return mx.where(mask, zero, neg)[None, None, :, :]

def _causal_tail_mask(q_len: int, kv_len: int, dtype: mx.Dtype) -> mx.array:
    q_pos = mx.arange(kv_len - q_len, kv_len)[:, None]
    k_pos = mx.arange(kv_len)[None, :]
    return _additive_mask(k_pos <= q_pos, dtype)

def _swa_tail_mask(q_len: int, kv_len: int, dtype: mx.Dtype, window_size: int) -> mx.array:
    q_pos = mx.arange(kv_len - q_len, kv_len)[:, None]
    k_pos = mx.arange(kv_len)[None, :]
    allowed = (k_pos <= q_pos) & (q_pos < k_pos + window_size)
    return _additive_mask(allowed, dtype)

@pytest.mark.parametrize("name,hq,hk,head_dim", SHAPES)
@pytest.mark.parametrize("kv_len", KV_LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask_kind", MASK_KINDS)
def test_batched_sdpa_2pass_matches_native_tail_masks(
    name: str,
    hq: int,
    hk: int,
    head_dim: int,
    kv_len: int,
    dtype: mx.Dtype,
    mask_kind: str,
) -> None:
    q_len = 16
    q = _rand((1, hq, q_len, head_dim), dtype)
    k = _rand((1, hk, kv_len, head_dim), dtype)
    v = _rand((1, hk, kv_len, head_dim), dtype)
    scale = 1.0 / math.sqrt(head_dim)

    if mask_kind == "swa":
        native_mask = _swa_tail_mask(q_len, kv_len, dtype, window_size=2048)
        custom_mask = native_mask
    else:
        native_mask = _causal_tail_mask(q_len, kv_len, dtype)
        custom_mask = None if mask_kind == "implicit_causal" else native_mask

    native = scaled_dot_product_attention(
        q,
        k,
        v,
        cache=None,
        scale=scale,
        mask=native_mask,
    )
    custom = batched_sdpa_2pass_exact(
        q,
        k,
        v,
        scale=scale,
        mask=custom_mask,
    )
    assert custom is not None, f"{name} {dtype} kv={kv_len} mask={mask_kind} returned None"
    mx.eval(native, custom)
    _sync()

    native_f = native.astype(mx.float32)
    custom_f = custom.astype(mx.float32)
    max_abs = float(mx.max(mx.abs(native_f - custom_f)).item())
    ref_max = float(mx.max(mx.abs(native_f)).item())
    max_rel = max_abs / (ref_max + 1e-6)

    assert max_abs <= 1e-2, (
        f"{name} {dtype} kv={kv_len} mask={mask_kind} max_abs={max_abs:.4g}"
    )
    assert max_rel <= 1e-2, (
        f"{name} {dtype} kv={kv_len} mask={mask_kind} max_rel={max_rel:.4g}"
    )

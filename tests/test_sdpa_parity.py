# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import math

import mlx.core as mx
import pytest
from mlx_lm.models.base import scaled_dot_product_attention

from dflash_mlx.engine import target_qwen_gdn as qwen_gdn
from dflash_mlx.engine.target_gemma4 import _gemma4_full_gqa_sdpa
from dflash_mlx.engine.target_qwen_gdn import _gqa_reshape_sdpa

SHAPES = [
    ("qwen36_27b", 24, 4, 256),
    ("qwen36_35b_a3b", 16, 2, 256),
]
GEMMA4_FULL_SHAPES = [
    ("gemma4_26b_full", 16, 2, 512),
    ("gemma4_31b_full", 32, 4, 512),
]
DTYPES = [mx.bfloat16, mx.float16]


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

def _swa_tail_bool_mask(q_len: int, kv_len: int, window_size: int) -> mx.array:
    q_pos = mx.arange(kv_len - q_len, kv_len)[:, None]
    k_pos = mx.arange(kv_len)[None, :]
    return (k_pos <= q_pos) & (q_pos < k_pos + window_size)

@pytest.mark.parametrize("name,hq,hk,head_dim", SHAPES)
@pytest.mark.parametrize("q_len", [1, 4, 16])
@pytest.mark.parametrize("kv_len", [1024, 4096, 8192, 16384, 32768])
@pytest.mark.parametrize("mask_kind", ["causal_string", "explicit_causal", "swa_bool"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gqa_reshape_sdpa_matches_native_masks(
    name: str,
    hq: int,
    hk: int,
    head_dim: int,
    q_len: int,
    kv_len: int,
    mask_kind: str,
    dtype: mx.Dtype,
) -> None:
    q = _rand((1, hq, q_len, head_dim), dtype)
    k = _rand((1, hk, kv_len, head_dim), dtype)
    v = _rand((1, hk, kv_len, head_dim), dtype)
    scale = 1.0 / math.sqrt(head_dim)

    if mask_kind == "causal_string":
        mask = "causal"
    elif mask_kind == "explicit_causal":
        mask = _causal_tail_mask(q_len, kv_len, dtype)
    else:
        mask = _swa_tail_bool_mask(q_len, kv_len, window_size=2048)

    native = scaled_dot_product_attention(
        q,
        k,
        v,
        cache=None,
        scale=scale,
        mask=mask,
    )
    custom = _gqa_reshape_sdpa(
        q,
        k,
        v,
        cache=None,
        scale=scale,
        mask=mask,
    )
    mx.eval(native, custom)
    _sync()

    native_f = native.astype(mx.float32)
    custom_f = custom.astype(mx.float32)
    max_abs = float(mx.max(mx.abs(native_f - custom_f)).item())
    ref_max = float(mx.max(mx.abs(native_f)).item())
    max_rel = max_abs / (ref_max + 1e-6)

    assert max_abs <= 1e-2, (
        f"{name} dtype={dtype} kv={kv_len} mask={mask_kind} max_abs={max_abs:.4g}"
    )
    assert max_rel <= 1e-2, (
        f"{name} dtype={dtype} kv={kv_len} mask={mask_kind} max_rel={max_rel:.4g}"
    )


@pytest.mark.parametrize("name,hq,hk,head_dim", GEMMA4_FULL_SHAPES)
@pytest.mark.parametrize("q_len", [1, 4, 16])
@pytest.mark.parametrize("kv_len", [1024, 8192, 16384, 32768])
@pytest.mark.parametrize("mask_kind", ["causal_string", "explicit_causal"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_gemma4_full_gqa_sdpa_matches_native_masks(
    name: str,
    hq: int,
    hk: int,
    head_dim: int,
    q_len: int,
    kv_len: int,
    mask_kind: str,
    dtype: mx.Dtype,
) -> None:
    q = _rand((1, hq, q_len, head_dim), dtype)
    k = _rand((1, hk, kv_len, head_dim), dtype)
    v = _rand((1, hk, kv_len, head_dim), dtype)
    scale = 1.0 / math.sqrt(head_dim)
    mask = (
        "causal"
        if mask_kind == "causal_string"
        else _causal_tail_mask(q_len, kv_len, dtype)
    )

    native = scaled_dot_product_attention(
        q,
        k,
        v,
        cache=None,
        scale=scale,
        mask=mask,
    )
    custom = _gemma4_full_gqa_sdpa(
        q,
        k,
        v,
        cache=None,
        scale=scale,
        mask=mask,
    )
    mx.eval(native, custom)
    _sync()

    native_f = native.astype(mx.float32)
    custom_f = custom.astype(mx.float32)
    max_abs = float(mx.max(mx.abs(native_f - custom_f)).item())
    ref_max = float(mx.max(mx.abs(native_f)).item())
    max_rel = max_abs / (ref_max + 1e-6)

    assert max_abs <= 1e-2, (
        f"{name} dtype={dtype} q={q_len} kv={kv_len} mask={mask_kind} max_abs={max_abs:.4g}"
    )
    assert max_rel <= 1e-2, (
        f"{name} dtype={dtype} q={q_len} kv={kv_len} mask={mask_kind} max_rel={max_rel:.4g}"
    )


@pytest.mark.parametrize("name,hq,hk,head_dim", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("q_len,kv_len", [(1, 65536), (4, 65536), (16, 65536)])
def test_qwen_gqa_large_kv_routes_match_native(
    name: str,
    hq: int,
    hk: int,
    head_dim: int,
    dtype: mx.Dtype,
    q_len: int,
    kv_len: int,
) -> None:
    q = _rand((1, hq, q_len, head_dim), dtype)
    k = _rand((1, hk, kv_len, head_dim), dtype)
    v = _rand((1, hk, kv_len, head_dim), dtype)
    scale = 1.0 / math.sqrt(head_dim)

    native = scaled_dot_product_attention(
        q,
        k,
        v,
        cache=None,
        scale=scale,
        mask="causal",
    )
    custom = _gqa_reshape_sdpa(
        q,
        k,
        v,
        cache=None,
        scale=scale,
        mask="causal",
    )
    mx.eval(native, custom)
    _sync()

    native_f = native.astype(mx.float32)
    custom_f = custom.astype(mx.float32)
    max_abs = float(mx.max(mx.abs(native_f - custom_f)).item())
    ref_max = float(mx.max(mx.abs(native_f)).item())
    max_rel = max_abs / (ref_max + 1e-6)

    assert max_abs <= 1e-2, (
        f"{name} dtype={dtype} q={q_len} kv={kv_len} max_abs={max_abs:.4g}"
    )
    assert max_rel <= 1e-2, (
        f"{name} dtype={dtype} q={q_len} kv={kv_len} max_rel={max_rel:.4g}"
    )


@pytest.mark.parametrize(
    "kv_len,expected_route,route_name",
    [
        (8192, "grouped", "grouped_gqa_sdpa"),
        (16384, "async", "async_per_head_gqa_sdpa"),
        (65536, "per_head", "per_head_gqa_sdpa"),
    ],
)
def test_qwen_hk2_q16_long_context_route_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    kv_len: int,
    expected_route: str,
    route_name: str,
) -> None:
    calls: list[str] = []

    def _route(queries, *_args, **_kwargs):
        calls.append(expected_route)
        return mx.zeros(queries.shape, dtype=queries.dtype)

    monkeypatch.setattr(qwen_gdn, route_name, _route)
    q = mx.zeros((1, 16, 16, 256), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, kv_len, 256), dtype=mx.bfloat16)
    v = mx.zeros((1, 2, kv_len, 256), dtype=mx.bfloat16)

    mx.eval(_gqa_reshape_sdpa(q, k, v, cache=None, scale=1.0, mask="causal"))

    assert calls == [expected_route]

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file

from __future__ import annotations

import argparse
import math

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

from dflash_mlx.kernels import quartet_gqa_sdpa_exact


def _dense_prefix_causal_mask(
    *,
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    dtype: mx.Dtype,
    pad_multiple: int = 0,
) -> mx.array:
    floor = float(mx.finfo(dtype).min)
    base = [[[[
        0.0 if k <= (kv_len - q_len + q) else floor
        for k in range(kv_len)
    ] for q in range(q_len)] for _ in range(heads)] for _ in range(batch)]

    if pad_multiple > 0:
        pad_start = max(0, kv_len - pad_multiple)
        for b in range(batch):
            for h in range(heads):
                for q in range(q_len):
                    for k in range(pad_start, kv_len):
                        base[b][h][q][k] = floor

    return mx.array(base, dtype=dtype)


def _validate_case(
    *,
    kv_len: int,
    with_mask: bool,
    rtol: float,
    atol: float,
) -> dict[str, float | int | bool]:
    batch = 1
    hq = 32
    hk = 8
    q_len = 16
    dim = 128
    dtype = mx.bfloat16
    scale = dim**-0.5

    queries = mx.random.normal((batch, hq, q_len, dim), dtype=mx.float32).astype(dtype)
    keys = mx.random.normal((batch, hk, kv_len, dim), dtype=mx.float32).astype(dtype)
    values = mx.random.normal((batch, hk, kv_len, dim), dtype=mx.float32).astype(dtype)

    causal_mask = _dense_prefix_causal_mask(
        batch=batch,
        heads=hq,
        q_len=q_len,
        kv_len=kv_len,
        dtype=dtype,
        pad_multiple=64 if with_mask and kv_len >= 256 else 0,
    )
    custom_mask = causal_mask if with_mask else None

    ref = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=scale,
        mask=causal_mask,
    )
    out = quartet_gqa_sdpa_exact(
        queries=queries,
        keys=keys,
        values=values,
        scale=scale,
        mask=custom_mask,
    )
    if out is None:
        raise RuntimeError(f"quartet_gqa_sdpa_exact returned None for N={kv_len} mask={with_mask}")

    mx.eval(ref, out)
    diff = mx.abs(ref.astype(mx.float32) - out.astype(mx.float32))
    denom = mx.maximum(mx.abs(ref.astype(mx.float32)), mx.array(1e-6, dtype=mx.float32))
    rel = diff / denom
    max_abs_diff = float(mx.max(diff).item())
    max_rel_diff = float(mx.max(rel).item())
    allclose = bool(mx.allclose(ref, out, rtol=rtol, atol=atol).item())
    return {
        "kv_len": kv_len,
        "with_mask": with_mask,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "allclose": allclose,
    }


def _validate_fallback() -> dict[str, bool]:
    queries = mx.zeros((1, 32, 1, 128), dtype=mx.bfloat16)
    keys = mx.zeros((1, 8, 32, 128), dtype=mx.bfloat16)
    values = mx.zeros((1, 8, 32, 128), dtype=mx.bfloat16)
    wrong_m = quartet_gqa_sdpa_exact(queries, keys, values, scale=128**-0.5)
    wrong_dtype = quartet_gqa_sdpa_exact(
        queries.astype(mx.float16),
        keys.astype(mx.float16),
        values.astype(mx.float16),
        scale=128**-0.5,
    )
    return {
        "fallback_m1": wrong_m is None,
        "fallback_fp16": wrong_dtype is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate quartet-GQA SDPA numerics.")
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()

    results = []
    for kv_len in (256, 1024, 4096, 8192):
        results.append(_validate_case(kv_len=kv_len, with_mask=False, rtol=args.rtol, atol=args.atol))
        results.append(_validate_case(kv_len=kv_len, with_mask=True, rtol=args.rtol, atol=args.atol))

    fallback = _validate_fallback()

    failed = [entry for entry in results if not entry["allclose"]]
    for entry in results:
        print(entry)
    print(fallback)

    if failed or not all(fallback.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

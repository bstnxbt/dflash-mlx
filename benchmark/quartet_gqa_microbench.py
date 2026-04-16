# Copyright 2026 bstnxbt
# MIT License — see LICENSE file

from __future__ import annotations

import argparse
import time

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
) -> mx.array:
    floor = float(mx.finfo(dtype).min)
    base = [[[[
        0.0 if k <= (kv_len - q_len + q) else floor
        for k in range(kv_len)
    ] for q in range(q_len)] for _ in range(heads)] for _ in range(batch)]
    return mx.array(base, dtype=dtype)


def _time_callable(fn, *, warmup: int, measure: int) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    elapsed_us = 0.0
    for _ in range(measure):
        start = time.perf_counter_ns()
        out = fn()
        mx.eval(out)
        elapsed_us += (time.perf_counter_ns() - start) / 1000.0
    return elapsed_us / measure


def _bench_case(
    *,
    kv_len: int,
    warmup: int,
    measure: int,
    quartet_explicit_mask: bool,
) -> dict[str, float | int]:
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
    )

    mx.eval(queries, keys, values, causal_mask)

    def stock_call():
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=None,
            scale=scale,
            mask=causal_mask,
        )

    quartet_mask = causal_mask if quartet_explicit_mask else None

    def quartet_call():
        out = quartet_gqa_sdpa_exact(
            queries=queries,
            keys=keys,
            values=values,
            scale=scale,
            mask=quartet_mask,
        )
        if out is None:
            raise RuntimeError(f"quartet_gqa_sdpa_exact returned None for kv_len={kv_len}")
        return out

    stock_us = _time_callable(stock_call, warmup=warmup, measure=measure)
    quartet_us = _time_callable(quartet_call, warmup=warmup, measure=measure)
    return {
        "kv_len": kv_len,
        "stock_us": stock_us,
        "quartet_us": quartet_us,
        "ratio_stock_over_quartet": stock_us / quartet_us if quartet_us else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbench stock SDPA vs quartet-GQA SDPA.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument(
        "--quartet-explicit-mask",
        action="store_true",
        help="Pass the dense additive mask to quartet too. Default mimics runtime with implicit prefix-causal masking.",
    )
    args = parser.parse_args()

    for kv_len in (1024, 2048, 4096, 8192):
        result = _bench_case(
            kv_len=kv_len,
            warmup=args.warmup,
            measure=args.measure,
            quartet_explicit_mask=args.quartet_explicit_mask,
        )
        print(
            {
                "kv_len": result["kv_len"],
                "stock_us": round(float(result["stock_us"]), 2),
                "quartet_us": round(float(result["quartet_us"]), 2),
                "ratio_stock_over_quartet": round(float(result["ratio_stock_over_quartet"]), 4),
            }
        )


if __name__ == "__main__":
    main()

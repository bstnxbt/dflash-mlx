# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import gc
import resource
import sys
import tempfile
import time
from collections.abc import Sequence

import mlx.core as mx

from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l2 import (
    DFlashPrefixL2Cache,
    _format_filename,
    _token_hash,
)
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot

def rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024

def mlx_active_mb() -> float:
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        return 0.0

def mlx_peak_mb() -> float:
    try:
        return mx.get_peak_memory() / (1024 * 1024)
    except Exception:
        return 0.0

def make_key(model_id: str = "bench-target") -> DFlashPrefixKey:
    return DFlashPrefixKey(
        target_model_id=model_id,
        draft_model_id="bench-draft",
        capture_layer_ids=tuple(range(8)),
        draft_sink_size=128,
        draft_window_size=2048,
        target_fa_window=0,
        format_version=1,
    )

def make_synthetic_snapshot(
    target_mb: int, key: DFlashPrefixKey, *, n_fa_layers: int = 32, kind: str = "prefill"
) -> DFlashPrefixSnapshot:
    target_bytes = int(target_mb * 1024 * 1024)
    head, head_dim = 8, 128
    bytes_per_token_per_layer_kv = 2 * 2 * head * head_dim
    seq = max(1, target_bytes // (n_fa_layers * bytes_per_token_per_layer_kv))
    fa_shape = (1, head, seq, head_dim)
    fa_states: list = []
    for _ in range(n_fa_layers):
        k = mx.zeros(fa_shape, dtype=mx.bfloat16)
        v = mx.zeros(fa_shape, dtype=mx.bfloat16)
        fa_states.append((k, v, 0))
    last_logits = mx.zeros((1, 152064), dtype=mx.float32)

    mx.eval(*[a for fa in fa_states for a in (fa[0], fa[1])], last_logits)
    return DFlashPrefixSnapshot(
        token_ids=tuple(range(seq)),
        fa_states=tuple(fa_states),
        gdn_states=tuple([None] * n_fa_layers),
        target_hidden_chunks=tuple(),
        target_hidden_chunk_spans=tuple(),
        target_hidden_total_len=0,
        last_logits=last_logits,
        key=key,
        kind=kind,
    )

def wait_for_writes(l2: DFlashPrefixL2Cache, target: int, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if l2.stats()["writes"] >= target:
            return
        time.sleep(0.01)
    raise RuntimeError(f"writer didn't finish: writes={l2.stats()['writes']} target={target}")

def time_block(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, out

def bench_size(target_mb: int) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Snapshot target: {target_mb} MB")
    print(f"{'=' * 60}")
    key = make_key()

    rss0 = rss_mb()
    mlx0 = mlx_active_mb()
    print(f"baseline             RSS={rss0:7.0f} MB   MLX active={mlx0:7.0f} MB")

    build_ms, snap = time_block("build", lambda: make_synthetic_snapshot(target_mb, key))
    actual_mb = snap.nbytes / (1024 * 1024)
    rss1 = rss_mb()
    mlx1 = mlx_active_mb()
    print(
        f"after build          RSS={rss1:7.0f} MB   MLX active={mlx1:7.0f} MB   "
        f"(actual {actual_mb:.0f} MB, build {build_ms:.0f} ms)"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_dir, max_bytes=10 * 1024 * 1024 * 1024)
        try:

            insert_ms, _ = time_block("insert_async", lambda: l2.insert_async(snap))
            rss_post_insert = rss_mb()
            mlx_post_insert = mlx_active_mb()

            t0 = time.perf_counter()
            wait_for_writes(l2, target=1)
            write_complete_ms = (time.perf_counter() - t0) * 1000
            rss_post_write = rss_mb()
            mlx_post_write = mlx_active_mb()

            print(
                f"after insert_async   RSS={rss_post_insert:7.0f} MB   "
                f"MLX active={mlx_post_insert:7.0f} MB   "
                f"(insert_async {insert_ms:.0f} ms = mx.eval + np.array(copy))"
            )
            print(
                f"after writer drain   RSS={rss_post_write:7.0f} MB   "
                f"MLX active={mlx_post_write:7.0f} MB   "
                f"(writer {write_complete_ms:.0f} ms = safetensors save + rename)"
            )

            token_len = len(snap.token_ids)
            del snap
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass
            rss_post_gc = rss_mb()
            mlx_post_gc = mlx_active_mb()
            print(
                f"after del+gc+clear   RSS={rss_post_gc:7.0f} MB   "
                f"MLX active={mlx_post_gc:7.0f} MB   "
                f"(MLX freed {mlx_post_write - mlx_post_gc:.0f} MB)"
            )

            disk_mb = l2.stats()["current_bytes"] / (1024 * 1024)
            print(f"on disk              {disk_mb:7.0f} MB")

            other_key = make_key(model_id="OTHER")
            tokens_short = tuple(range(min(token_len, 100)))
            t0 = time.perf_counter()
            iters = 1000
            for _ in range(iters):
                res = l2.lookup(tokens_short, other_key)
                assert res is None
            miss_other_us = (time.perf_counter() - t0) * 1e6 / iters
            print(f"lookup miss (other key, no bucket): {miss_other_us:.1f} us/op")

            from dflash_mlx.cache.prefix_l2 import _key_hash, _runtime_layout_hash, L2_LAYOUT_ROOT
            from pathlib import Path
            import secrets
            kh = _key_hash(key)
            bucket = Path(tmp_dir) / L2_LAYOUT_ROOT / _runtime_layout_hash() / kh[:2] / kh
            for _ in range(100):
                name = _format_filename(
                    token_len=token_len,
                    token_hash=secrets.token_hex(8),
                    kind="prefill",
                    fp_short=secrets.token_hex(8),
                )
                (bucket / name).write_bytes(b"\x00")
            wrong_tokens = tuple([t + 1 for t in range(token_len)])
            t0 = time.perf_counter()
            iters_filt = 200
            for _ in range(iters_filt):
                res = l2.lookup(wrong_tokens, key)
                assert res is None
            miss_filtered_us = (time.perf_counter() - t0) * 1e6 / iters_filt
            print(
                f"lookup miss (same key, hash-filtered, 100 bogus): "
                f"{miss_filtered_us:.1f} us/op"
            )
            stats_mid = l2.stats()
            print(
                f"   lookup_hash_filtered={stats_mid['lookup_hash_filtered']} "
                f"lookup_loads={stats_mid['lookup_loads']}"
            )

            assert stats_mid["lookup_loads"] == 0, "filter regressed: mx.load was called"

            req_tokens = tuple(range(token_len))
            mlx_pre_hit = mlx_active_mb()
            rss_pre_hit = rss_mb()
            t0 = time.perf_counter()
            hit = l2.lookup(req_tokens, key)
            hit_cold_ms = (time.perf_counter() - t0) * 1000
            assert hit is not None
            assert hit.token_ids == req_tokens
            mlx_post_hit = mlx_active_mb()
            rss_post_hit = rss_mb()
            print(
                f"lookup hit (cold):   {hit_cold_ms:.0f} ms   "
                f"(MLX active +{mlx_post_hit - mlx_pre_hit:.0f} MB, "
                f"RSS +{rss_post_hit - rss_pre_hit:.0f} MB)"
            )
            del hit
            gc.collect()
            try:
                mx.clear_cache()
            except Exception:
                pass

            warm_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                hit = l2.lookup(req_tokens, key)
                warm_times.append((time.perf_counter() - t0) * 1000)
                assert hit is not None
                del hit
                gc.collect()
                try:
                    mx.clear_cache()
                except Exception:
                    pass
            warm_avg_ms = sum(warm_times) / len(warm_times)
            print(f"lookup hit (warm):   {warm_avg_ms:.0f} ms avg ({warm_times})")

            return {
                "target_mb": target_mb,
                "actual_mb": actual_mb,
                "insert_async_ms": insert_ms,
                "write_complete_ms": write_complete_ms,
                "lookup_miss_other_us": miss_other_us,
                "lookup_miss_filtered_us": miss_filtered_us,
                "lookup_hit_cold_ms": hit_cold_ms,
                "lookup_hit_warm_ms": warm_avg_ms,
                "rss_baseline_mb": rss0,
                "rss_after_build_mb": rss1,
                "rss_after_write_mb": rss_post_write,
                "rss_after_gc_mb": rss_post_gc,
                "mlx_baseline_mb": mlx0,
                "mlx_after_build_mb": mlx1,
                "mlx_after_write_mb": mlx_post_write,
                "mlx_after_gc_mb": mlx_post_gc,
            }
        finally:
            l2.shutdown()

def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="256,1400", help="comma-separated MB")
    args = parser.parse_args(list(argv) if argv is not None else None)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print(f"Sizes: {sizes} MB")
    results = []
    for size in sizes:
        results.append(bench_size(size))
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    print(
        f"{'size_MB':>8s} {'insert_ms':>10s} {'write_ms':>10s} "
        f"{'miss_other_us':>14s} {'miss_filt_us':>13s} "
        f"{'hit_cold_ms':>12s} {'hit_warm_ms':>12s}"
    )
    for r in results:
        print(
            f"{r['target_mb']:8d} {r['insert_async_ms']:10.0f} "
            f"{r['write_complete_ms']:10.0f} "
            f"{r['lookup_miss_other_us']:14.1f} "
            f"{r['lookup_miss_filtered_us']:13.1f} "
            f"{r['lookup_hit_cold_ms']:12.0f} "
            f"{r['lookup_hit_warm_ms']:12.0f}"
        )

if __name__ == "__main__":
    main()

"""Aggregate POST/cycle/cache JSONL events into a summary.json.

Usage:
    python benchmark/aggregate_replay.py <run_dir>

Reads <run_dir>/{post_events.jsonl,cycle_events.jsonl,cache_events.jsonl}
and writes <run_dir>/summary.json with totals and per-quartile decomposition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def aggregate(run_dir: Path) -> dict[str, Any]:
    posts = _read_jsonl(run_dir / "post_events.jsonl")
    cycles = _read_jsonl(run_dir / "cycle_events.jsonl")
    caches = _read_jsonl(run_dir / "cache_events.jsonl")

    n_posts = len(posts)
    dflash_posts = [p for p in posts if p.get("mode_used", "").startswith("dflash")]
    ar_posts = [p for p in posts if p.get("mode_used") == "ar_fastpath"]
    fallback_posts = [p for p in posts if p.get("mode_used") == "dflash_fallback"]

    wall_ms_total = sum(_safe_float(p.get("wall_ms")) for p in posts)
    prefill_ms_total = sum(_safe_float(p.get("prefill_ms")) for p in dflash_posts)
    decode_ms_total = sum(_safe_float(p.get("decode_ms")) for p in dflash_posts)
    cache_lookup_ms_total = sum(_safe_float(p.get("cache_lookup_ms")) for p in dflash_posts)
    cache_insert_ms_total = sum(_safe_float(p.get("cache_insert_ms")) for p in dflash_posts)
    prompt_tokens_total = sum(_safe_int(p.get("prompt_tokens")) for p in dflash_posts)
    generated_tokens_total = sum(_safe_int(p.get("generated_tokens")) for p in dflash_posts)
    cache_hit_tokens_total = sum(_safe_int(p.get("cache_hit_tokens")) for p in dflash_posts)

    ttft_ms_values = [
        _safe_float(p.get("ttft_ms"))
        for p in dflash_posts
        if p.get("ttft_ms") is not None
    ]
    ttft_avg = sum(ttft_ms_values) / len(ttft_ms_values) if ttft_ms_values else 0.0

    decode_tps_session = (
        (generated_tokens_total / (decode_ms_total / 1000.0))
        if decode_ms_total > 0
        else 0.0
    )

    n_cycles = len(cycles)
    commit_total = sum(_safe_int(c.get("commit_count")) for c in cycles)
    acceptance_total = sum(_safe_int(c.get("acceptance_len")) for c in cycles)
    block_len_avg = (
        sum(_safe_int(c.get("block_len")) for c in cycles) / n_cycles
        if n_cycles
        else 0.0
    )
    avg_tokens_per_cycle = (commit_total / n_cycles) if n_cycles else 0.0

    if n_cycles >= 4:
        q = max(1, n_cycles // 4)
        first_q = cycles[:q]
        last_q = cycles[-q:]
        first_quarter_tpc = sum(_safe_int(c.get("commit_count")) for c in first_q) / len(first_q)
        last_quarter_tpc = sum(_safe_int(c.get("commit_count")) for c in last_q) / len(last_q)
    else:
        first_quarter_tpc = avg_tokens_per_cycle
        last_quarter_tpc = avg_tokens_per_cycle

    cycle_us_keys = (
        "draft_us",
        "verify_us",
        "acceptance_us",
        "hidden_extraction_us",
        "rollback_us",
        "other_us",
        "cycle_total_us",
    )
    cycle_us_totals = {
        key: sum(_safe_float(c.get(key)) for c in cycles) for key in cycle_us_keys
    }

    cache_ops: dict[str, int] = {}
    for c in caches:
        op = str(c.get("op", "?"))
        result = str(c.get("result", "")) if c.get("result") else ""
        bucket = f"{op}:{result}" if result else op
        cache_ops[bucket] = cache_ops.get(bucket, 0) + 1

    summary = {
        "run_dir": str(run_dir),
        "posts": n_posts,
        "posts_dflash": len(dflash_posts),
        "posts_ar_fastpath": len(ar_posts),
        "posts_dflash_fallback": len(fallback_posts),
        "wall_ms_total": wall_ms_total,
        "prefill_ms_total": prefill_ms_total,
        "decode_ms_total": decode_ms_total,
        "cache_lookup_ms_total": cache_lookup_ms_total,
        "cache_insert_ms_total": cache_insert_ms_total,
        "prompt_tokens_total": prompt_tokens_total,
        "generated_tokens_total": generated_tokens_total,
        "cache_hit_tokens_total": cache_hit_tokens_total,
        "ttft_ms_avg": ttft_avg,
        "decode_tps_session": decode_tps_session,
        "dflash_cycles": n_cycles,
        "commit_total": commit_total,
        "acceptance_total": acceptance_total,
        "block_len_avg": block_len_avg,
        "avg_tokens_per_cycle": avg_tokens_per_cycle,
        "first_quarter_tpc": first_quarter_tpc,
        "last_quarter_tpc": last_quarter_tpc,
        "cycle_us_totals": cycle_us_totals,
        "cache_ops": cache_ops,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--print", dest="print_only", action="store_true")
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"Not a directory: {args.run_dir}", file=sys.stderr)
        return 2

    summary = aggregate(args.run_dir)
    out_path = args.run_dir / "summary.json"
    if not args.print_only:
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

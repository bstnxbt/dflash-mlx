# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

from dflash_mlx.bench_logger import log_post as _bench_log_post
from dflash_mlx.diagnostics import DiagnosticsConfig

def write_summary_line(
    *,
    summary_event: dict[str, Any],
    prompt_token_count: int,
) -> None:
    generation_tokens = int(summary_event.get("generation_tokens", 0) or 0)
    elapsed_us = float(summary_event.get("elapsed_us", 0.0) or 0.0)
    phase_timings_us = dict(summary_event.get("phase_timings_us") or {})
    prefill_us = float(phase_timings_us.get("prefill", 0.0) or 0.0)
    decode_s = max(0.0, (elapsed_us - prefill_us) / 1_000_000.0)
    tok_s = (generation_tokens / decode_s) if decode_s > 0.0 else 0.0
    acceptance_pct = float(summary_event.get("acceptance_ratio", 0.0) or 0.0) * 100.0
    total_s = elapsed_us / 1_000_000.0
    sys.stderr.write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] {tok_s:.1f} tok/s | "
        f"{acceptance_pct:.1f}% accepted | {generation_tokens} tokens | "
        f"{total_s:.1f}s | prompt: {prompt_token_count} tokens\n"
    )
    sys.stderr.flush()

def log_bench_post(
    *,
    request_id: int,
    summary_event: Optional[dict[str, Any]],
    request_start_ns: int,
    request_done_ns: int,
    first_token_ns: Optional[int],
    prefill_done_ns: Optional[int],
    prompt_token_count: int,
    live_token_count: int,
    cache_lookup_ms: float,
    cache_hit_tokens: int,
    cache_insert_ms: float,
    finish_reason: Optional[str],
    max_tokens: int,
    prompt_regime: Optional[dict[str, Any]] = None,
    memory_waterfall_peak: Optional[dict[str, Any]] = None,
    diagnostics: Optional[DiagnosticsConfig] = None,
) -> None:
    wall_ms = (request_done_ns - request_start_ns) / 1e6
    ttft_ms = (
        (first_token_ns - request_start_ns) / 1e6
        if first_token_ns is not None
        else None
    )
    prefill_ms = (
        (prefill_done_ns - request_start_ns) / 1e6
        if prefill_done_ns is not None
        else None
    )
    decode_ms = (
        (request_done_ns - prefill_done_ns) / 1e6
        if prefill_done_ns is not None
        else None
    )
    fallback_used = bool(summary_event.get("fallback_ar") if summary_event else False)
    generation_tokens = int(
        (summary_event or {}).get("generation_tokens", live_token_count) or 0
    )
    acceptance_ratio = float((summary_event or {}).get("acceptance_ratio", 0.0) or 0.0)
    cycles_completed = int((summary_event or {}).get("cycles_completed", 0) or 0)
    tokens_per_cycle = float((summary_event or {}).get("tokens_per_cycle", 0.0) or 0.0)
    _bench_log_post(
        diagnostics.trace if diagnostics is not None else None,
        request_id=request_id,
        mode_used="dflash_fallback" if fallback_used else "dflash",
        prompt_tokens=int(prompt_token_count),
        generated_tokens=generation_tokens,
        wall_ms=wall_ms,
        ttft_ms=ttft_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        cache_lookup_ms=cache_lookup_ms,
        cache_hit_tokens=cache_hit_tokens,
        cache_insert_ms=cache_insert_ms,
        acceptance_ratio=acceptance_ratio,
        tokens_per_cycle=tokens_per_cycle,
        cycles_completed=cycles_completed,
        finish_reason=finish_reason,
        max_tokens=int(max_tokens),
        prompt_regime=prompt_regime or {},
        memory_waterfall_peak=memory_waterfall_peak or {},
    )
    _append_diagnostics_summary(
        request_id=request_id,
        wall_ms=wall_ms,
        ttft_ms=ttft_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        prompt_token_count=prompt_token_count,
        generation_tokens=generation_tokens,
        acceptance_ratio=acceptance_ratio,
        tokens_per_cycle=tokens_per_cycle,
        cycles_completed=cycles_completed,
        cache_hit_tokens=cache_hit_tokens,
        diagnostics=diagnostics,
    )

def _append_diagnostics_summary(
    *,
    request_id: int,
    wall_ms: float,
    ttft_ms: Optional[float],
    prefill_ms: Optional[float],
    decode_ms: Optional[float],
    prompt_token_count: int,
    generation_tokens: int,
    acceptance_ratio: float,
    tokens_per_cycle: float,
    cycles_completed: int,
    cache_hit_tokens: int,
    diagnostics: Optional[DiagnosticsConfig],
) -> None:
    if diagnostics is None or diagnostics.run_dir is None:
        return
    path = Path(diagnostics.run_dir) / "summary.md"
    line = (
        f"| {request_id} | {prompt_token_count} | {generation_tokens} | "
        f"{_fmt_ms(wall_ms)} | {_fmt_ms(ttft_ms)} | {_fmt_ms(prefill_ms)} | "
        f"{_fmt_ms(decode_ms)} | {acceptance_ratio:.3f} | "
        f"{tokens_per_cycle:.2f} | {cycles_completed} | {cache_hit_tokens} |\n"
    )
    try:
        current = path.read_text(errors="ignore") if path.exists() else ""
        with path.open("a") as fp:
            if "| request |" not in current:
                fp.write(
                    "\n## Requests\n\n"
                    "| request | prompt_tokens | generated | wall_ms | ttft_ms | "
                    "prefill_ms | decode_ms | acceptance | tokens_cycle | cycles | cache_hit_tokens |\n"
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
                )
            fp.write(line)
    except OSError:
        pass

def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.1f}"

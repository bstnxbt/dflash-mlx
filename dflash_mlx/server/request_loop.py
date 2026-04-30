# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

from dflash_mlx.bench_logger import log_cycle as _bench_log_cycle
from dflash_mlx.engine.memory_waterfall import (
    collect_memory_waterfall,
    format_memory_waterfall_summary,
    merge_memory_waterfall_peak,
)
from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow
from dflash_mlx.server.protocol import make_response, match_stream_token

@dataclass
class RequestLoopResult:
    summary_event: Optional[dict[str, Any]]
    request_start_ns: int
    first_token_ns: Optional[int]
    prefill_done_ns: Optional[int]
    live_token_count: int
    finish_reason: Optional[str]
    cache_lookup_ms: float
    cache_hit_tokens: int
    cache_insert_ms: float
    memory_waterfall_peak: Optional[dict[str, Any]] = None

def consume_dflash_events(
    *,
    event_iter: Any,
    rqueue: Any,
    ctx: Any,
    tokenizer: Any,
    prompt: list[int],
    max_tokens: int,
    eos_token_ids: set[int],
    request_start_ns: int,
    prefix_flow: Optional[PrefixCacheFlow] = None,
    sm: Optional[Any] = None,
    sm_state: Optional[Any] = None,
    bench_active: bool = False,
    request_id: int = 0,
    runtime_context: Optional[Any] = None,
) -> RequestLoopResult:
    detokenizer = tokenizer.detokenizer
    if hasattr(detokenizer, "reset"):
        detokenizer.reset()

    pending_token: Optional[int] = None
    pending_text = ""
    pending_state: Optional[str] = "normal"
    pending_match: Optional[tuple[int, ...]] = None
    pending_finish_reason: Optional[str] = None
    first_token_flushed = False
    finish_reason: Optional[str] = None
    summary_event: Optional[dict[str, Any]] = None
    prefill_done_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    prefill_elapsed_s = 0.0
    live_token_count = 0
    live_prompt_len = len(prompt)
    printed_prefill_progress = False
    client_done = False
    memory_peak: Optional[dict[str, Any]] = None
    diagnostics = (
        runtime_context.diagnostics
        if runtime_context is not None
        else None
    )
    trace = diagnostics.trace if diagnostics is not None else None
    memory_enabled = bool(diagnostics is not None and diagnostics.memory_waterfall)

    try:
        for event in event_iter:
            event_name = event.get("event")
            if bench_active and event_name == "cycle_complete":
                cycle_event = {k: v for k, v in event.items() if k != "event"}
                _bench_log_cycle(trace, request_id=request_id, **cycle_event)
                continue
            if event_name == "memory_waterfall":
                memory_event = {k: v for k, v in event.items() if k != "event"}
                memory_peak = merge_memory_waterfall_peak(memory_peak, memory_event)
                if bench_active:
                    _bench_log_cycle(trace, request_id=request_id, **memory_event)
                continue
            if event_name in ("prefill", "prefill_progress"):
                processed = int(
                    event.get(
                        "tokens_processed",
                        event.get("prompt_token_count", len(prompt)),
                    )
                )
                total = int(
                    event.get(
                        "tokens_total",
                        event.get("prompt_token_count", len(prompt)),
                    )
                )
                elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
                if event_name == "prefill_progress":
                    sys.stderr.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] "
                        f"prefill: {processed}/{total} tokens | {elapsed_s:.1f}s\n"
                    )
                    sys.stderr.flush()
                    rqueue.put((processed, total))
                    printed_prefill_progress = True
                else:
                    prefill_elapsed_s = elapsed_s
                    prefill_done_ns = time.perf_counter_ns()
                    if not printed_prefill_progress:
                        sys.stderr.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] "
                            f"prefill: {processed}/{total} tokens | {elapsed_s:.1f}s\n"
                        )
                        sys.stderr.flush()
                continue
            if event_name == "prefill_snapshot_ready":
                if prefix_flow is not None:
                    prefix_flow.handle_prefill_snapshot(event)
                continue
            if event_name == "generation_snapshot_ready":
                if prefix_flow is not None:
                    prefix_flow.handle_generation_snapshot(event)
                continue
            if event_name != "token":
                if event_name == "summary":
                    summary_event = event
                    generated_token_ids = list(event.get("generated_token_ids", []) or [])
                    if generated_token_ids:
                        last_token = int(generated_token_ids[-1])
                        if last_token in eos_token_ids:
                            finish_reason = "stop"
                        elif int(event.get("generation_tokens", 0)) >= int(max_tokens):
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"
                    else:
                        finish_reason = "stop"
                continue

            if client_done:
                continue
            token = int(event["token_id"])
            if first_token_ns is None:
                first_token_ns = time.perf_counter_ns()
            live_token_count += 1
            live_acceptance_pct = float(event.get("acceptance_ratio", 0.0) or 0.0) * 100.0
            elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
            live_tok_s = live_token_count / max(0.001, elapsed_s - prefill_elapsed_s)
            if live_token_count % 2048 == 0:
                sys.stderr.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] "
                    f"{live_tok_s:.1f} tok/s | {live_acceptance_pct:.1f}% accepted | "
                    f"{live_token_count} tokens | {elapsed_s:.1f}s | "
                    f"prompt: {live_prompt_len} tokens\n"
                )
                sys.stderr.flush()

            token_finish_reason: Optional[str] = None
            sm_state, match_sequence, current_state, terminal_match = match_stream_token(
                sm,
                sm_state,
                token,
            )
            if terminal_match or token in eos_token_ids:
                token_finish_reason = "stop"
            elif live_token_count >= int(max_tokens):
                token_finish_reason = "length"

            text = ""
            if token not in eos_token_ids:
                detokenizer.add_token(token)
                text = detokenizer.last_segment

            if not first_token_flushed:
                rqueue.put(
                    make_response(
                        text=text,
                        token=token,
                        state=current_state or "normal",
                        match=match_sequence,
                        finish_reason=token_finish_reason,
                    )
                )
                first_token_flushed = True
                if _context_should_stop(ctx):
                    break
                if token_finish_reason is not None:
                    client_done = True
                continue

            if pending_token is not None:
                rqueue.put(
                    make_response(
                        text=pending_text,
                        token=pending_token,
                        state=pending_state,
                        match=pending_match,
                        finish_reason=pending_finish_reason,
                    )
                )

            pending_token = token
            pending_text = text
            pending_state = current_state or "normal"
            pending_match = match_sequence
            pending_finish_reason = token_finish_reason

            if _context_should_stop(ctx):
                break
            if token_finish_reason is not None:
                client_done = True
    finally:
        close = getattr(event_iter, "close", None)
        if close is not None:
            close()
        if memory_enabled:
            memory_event = collect_memory_waterfall(
                phase="after_cleanup",
                prefix_cache=prefix_flow.cache if prefix_flow is not None else None,
            )
            memory_peak = merge_memory_waterfall_peak(memory_peak, memory_event)
            if bench_active:
                _bench_log_cycle(trace, request_id=request_id, **memory_event)

    detokenizer.finalize()
    tail = detokenizer.last_segment
    if pending_token is not None:
        rqueue.put(
            make_response(
                text=pending_text + tail,
                token=pending_token,
                state=pending_state,
                match=pending_match,
                finish_reason=finish_reason or pending_finish_reason,
            )
        )

    if memory_peak:
        sys.stderr.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{format_memory_waterfall_summary(memory_peak)}\n"
        )
        sys.stderr.flush()

    return RequestLoopResult(
        summary_event=summary_event,
        request_start_ns=request_start_ns,
        first_token_ns=first_token_ns,
        prefill_done_ns=prefill_done_ns,
        live_token_count=live_token_count,
        finish_reason=finish_reason,
        cache_lookup_ms=prefix_flow.lookup_ms if prefix_flow is not None else 0.0,
        cache_hit_tokens=prefix_flow.hit_tokens if prefix_flow is not None else 0,
        cache_insert_ms=prefix_flow.insert_ms if prefix_flow is not None else 0.0,
        memory_waterfall_peak=memory_peak,
    )

def _context_should_stop(ctx: Any) -> bool:
    return bool(getattr(ctx, "_should_stop", False))

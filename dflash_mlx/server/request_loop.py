# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

from dflash_mlx.engine.memory_waterfall import (
    collect_memory_waterfall,
    format_memory_waterfall_summary,
    merge_memory_waterfall_peak,
    prefix_cache_memory_fields,
)
from dflash_mlx.engine.events import (
    CycleCompleteEvent,
    MemoryWaterfallEvent,
    PrefillCompleteEvent,
    PrefillProgressEvent,
    SnapshotPublishedEvent,
    SummaryEvent,
    TokenEvent,
)
from dflash_mlx.cache.manager import current_runtime_cache_manager
from dflash_mlx.observability.memory import process_memory_snapshot
from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow
from dflash_mlx.server.metrics import record_cycle_diagnostic, update_live_request
from dflash_mlx.server.protocol import make_response, match_stream_token

_GB = 1_000_000_000.0

@dataclass
class RequestLoopResult:
    summary_event: Optional[SummaryEvent]
    prefill_event: Optional[PrefillCompleteEvent]
    request_start_ns: int
    first_token_ns: Optional[int]
    prefill_done_ns: Optional[int]
    live_token_count: int
    finish_reason: Optional[str]
    cache_lookup_ms: float
    cache_hit_tokens: int
    cache_insert_ms: float
    memory_waterfall_peak: Optional[dict[str, Any]] = None
    memory_waterfall_start: Optional[dict[str, Any]] = None
    memory_waterfall_end: Optional[dict[str, Any]] = None

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
    summary_event: Optional[SummaryEvent] = None
    prefill_event: Optional[PrefillCompleteEvent] = None
    prefill_done_ns: Optional[int] = None
    first_token_ns: Optional[int] = None
    prefill_elapsed_s = 0.0
    live_token_count = 0
    live_prompt_len = len(prompt)
    printed_prefill_progress = False
    client_done = False
    memory_peak: Optional[dict[str, Any]] = None
    memory_start: Optional[dict[str, Any]] = None
    memory_end: Optional[dict[str, Any]] = None
    diagnostics = (
        runtime_context.diagnostics
        if runtime_context is not None
        else None
    )
    memory_enabled = bool(diagnostics is not None and diagnostics.memory_waterfall)
    memory_boundary_enabled = bool(
        diagnostics is not None and getattr(diagnostics, "mode", "off") != "off"
    )
    if memory_boundary_enabled:
        memory_start = (
            collect_memory_waterfall(
                phase="request_start",
                prefix_cache_memory=(
                    prefix_flow.prefix_cache_memory_bytes()
                    if prefix_flow is not None
                    else None
                ),
            )
            if memory_enabled
            else _boundary_memory_snapshot("request_start")
        )
        memory_peak = merge_memory_waterfall_peak(memory_peak, memory_start)
        if bench_active:
            record_cycle_diagnostic(
                diagnostics=diagnostics,
                request_id=request_id,
                fields=memory_start,
            )

    cache_manager = current_runtime_cache_manager()
    if cache_manager is not None:
        cache_manager.begin_request()
    try:
        for event in event_iter:
            if isinstance(event, CycleCompleteEvent):
                if bench_active:
                    record_cycle_diagnostic(
                        diagnostics=diagnostics,
                        request_id=request_id,
                        fields=event.to_payload(),
                    )
                continue
            if isinstance(event, MemoryWaterfallEvent):
                memory_event = dict(event.fields)
                memory_event = _with_prefix_cache_memory(memory_event, prefix_flow)
                memory_peak = merge_memory_waterfall_peak(memory_peak, memory_event)
                if bench_active:
                    record_cycle_diagnostic(
                        diagnostics=diagnostics,
                        request_id=request_id,
                        fields=memory_event,
                    )
                continue
            if isinstance(event, PrefillProgressEvent | PrefillCompleteEvent):
                if isinstance(event, PrefillCompleteEvent):
                    prefill_event = event
                    processed = int(event.prompt_token_count)
                    total = int(event.prompt_token_count)
                else:
                    processed = int(event.tokens_processed)
                    total = int(event.tokens_total)
                elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
                update_live_request(
                    request_id=request_id,
                    state=(
                        "prefill"
                        if isinstance(event, PrefillProgressEvent)
                        else "decode"
                    ),
                    prefill_tokens_processed=processed,
                    prefill_tokens_total=total,
                    prefill_s=(
                        elapsed_s if isinstance(event, PrefillCompleteEvent) else None
                    ),
                    prefill_done=isinstance(event, PrefillCompleteEvent),
                    prefill_phase_timings_us=(
                        event.phase_timings()
                        if isinstance(event, PrefillCompleteEvent)
                        else None
                    ),
                )
                if isinstance(event, PrefillProgressEvent):
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
            if isinstance(event, SnapshotPublishedEvent):
                continue
            if isinstance(event, SummaryEvent):
                summary_event = event
                update_live_request(
                    request_id=request_id,
                    state="finishing",
                    generated_tokens=int(event.generation_tokens),
                    acceptance_rate=float(event.acceptance_ratio),
                    tokens_per_cycle=float(event.tokens_per_cycle),
                    cycles=int(event.cycles_completed),
                    adaptive_block_reductions=int(event.adaptive_block_reductions),
                    adaptive_block_cycles=int(event.adaptive_block_cycles),
                    adaptive_block_min=event.adaptive_block_min,
                    copyspec_hits=int(event.copyspec_hits),
                    copyspec_tokens=int(event.copyspec_tokens),
                    phase_timings_us=dict(event.phase_timings_us),
                )
                generated_token_ids = list(event.generated_token_ids)
                if generated_token_ids:
                    last_token = int(generated_token_ids[-1])
                    if last_token in eos_token_ids:
                        finish_reason = "stop"
                    elif int(event.generation_tokens) >= int(max_tokens):
                        finish_reason = "length"
                    else:
                        finish_reason = "stop"
                else:
                    finish_reason = "stop"
                continue
            if not isinstance(event, TokenEvent):
                raise TypeError(f"Unsupported DFlash engine event: {type(event).__name__}")

            if client_done:
                continue
            token = int(event.token_id)
            if first_token_ns is None:
                first_token_ns = time.perf_counter_ns()
                ttft_s = (first_token_ns - request_start_ns) / 1e9
            else:
                ttft_s = None
            live_token_count += 1
            live_acceptance_pct = float(event.acceptance_ratio) * 100.0
            elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
            live_tok_s = live_token_count / max(0.001, elapsed_s - prefill_elapsed_s)
            cycles_completed = int(event.cycles_completed)
            tokens_per_cycle = (
                float(event.generated_tokens) / float(cycles_completed)
                if cycles_completed > 0
                else None
            )
            update_live_request(
                request_id=request_id,
                state="decode",
                generated_tokens=live_token_count,
                ttft_s=ttft_s,
                decode_tok_s=live_tok_s,
                acceptance_rate=live_acceptance_pct / 100.0,
                tokens_per_cycle=tokens_per_cycle,
                cycles=cycles_completed,
                adaptive_block_reductions=int(event.adaptive_block_reductions),
                adaptive_block_cycles=int(event.adaptive_block_cycles),
                adaptive_block_min=event.adaptive_block_min,
                copyspec_hits=int(event.copyspec_hits),
                copyspec_tokens=int(event.copyspec_tokens),
            )
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
        if cache_manager is not None:
            cache_manager.end_request()
        if memory_boundary_enabled:
            memory_event = (
                collect_memory_waterfall(
                    phase="after_cleanup",
                    prefix_cache_memory=(
                        prefix_flow.prefix_cache_memory_bytes()
                        if prefix_flow is not None
                        else None
                    ),
                )
                if memory_enabled
                else _boundary_memory_snapshot("request_end")
            )
            memory_end = memory_event
            memory_peak = merge_memory_waterfall_peak(memory_peak, memory_event)
            if bench_active:
                record_cycle_diagnostic(
                    diagnostics=diagnostics,
                    request_id=request_id,
                    fields=memory_event,
                )

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
        prefill_event=prefill_event,
        request_start_ns=request_start_ns,
        first_token_ns=first_token_ns,
        prefill_done_ns=prefill_done_ns,
        live_token_count=live_token_count,
        finish_reason=finish_reason,
        cache_lookup_ms=prefix_flow.lookup_ms if prefix_flow is not None else 0.0,
        cache_hit_tokens=prefix_flow.hit_tokens if prefix_flow is not None else 0,
        cache_insert_ms=prefix_flow.insert_ms if prefix_flow is not None else 0.0,
        memory_waterfall_peak=memory_peak,
        memory_waterfall_start=memory_start,
        memory_waterfall_end=memory_end,
    )

def _with_prefix_cache_memory(
    event: dict[str, Any],
    prefix_flow: Optional[PrefixCacheFlow],
) -> dict[str, Any]:
    if prefix_flow is None:
        return event
    prefix_memory = prefix_flow.prefix_cache_memory_bytes()
    if prefix_memory is None:
        return event
    return {**event, **prefix_cache_memory_fields(prefix_memory)}

def _boundary_memory_snapshot(phase: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "memory_phase": str(phase),
        **process_memory_snapshot(include_system_wired=False),
    }
    for key, value in list(payload.items()):
        if key.endswith("_bytes") and value is not None:
            payload[key[:-6] + "_gb"] = float(value) / _GB
    return payload

def _context_should_stop(ctx: Any) -> bool:
    return bool(getattr(ctx, "_should_stop", False))

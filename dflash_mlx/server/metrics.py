# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dflash_mlx.observability.writer import (
    log_cycle as _bench_log_cycle,
    log_post as _bench_log_post,
    report_observability_failure,
)
from dflash_mlx.observability.cache import (
    live_prefix_cache_payload,
    live_prefix_cache_totals,
)
from dflash_mlx.cache.manager import (
    RuntimeCacheManagerClosed,
    current_runtime_cache_manager,
    sync_runtime_cache_manager,
)
from dflash_mlx.diagnostics import DiagnosticsConfig
from dflash_mlx.engine.events import PrefillCompleteEvent, SummaryEvent
from dflash_mlx.observability.memory import live_memory_payload
from dflash_mlx.server.prefix_cache_manager import build_prefix_key

_LIVE_LOCK = threading.Lock()
_LIVE_STARTED_AT = time.time()
_LIVE_SERVER: dict[str, Any] = {
    "version": None,
    "model": None,
    "draft": None,
    "draft_quant": None,
    "draft_quant_source": None,
    "mode": "dflash",
}
_LIVE_RUNTIME: dict[str, Any] = {
    "prefill_step_size": None,
    "prefix_cache_enabled": None,
    "target_fa_window": None,
    "diagnostics": None,
}
_LIVE_PREFIX_CONFIG: dict[str, Any] = {
    "enabled": False,
    "max_entries": None,
    "max_bytes": None,
}
_LIVE_METAL_LIMITS: dict[str, Any] = {
    "wired_limit_bytes": None,
}
_LIVE_LAST_REQUEST: Optional[dict[str, Any]] = None
_LIVE_CURRENT_REQUEST: Optional[dict[str, Any]] = None
_LIVE_RECENT_REQUESTS = deque(maxlen=32)
_LIVE_REQUEST_FINISH_TIMES = deque(maxlen=512)
_LIVE_LAST_PREFIX: dict[str, Optional[int]] = {
    "restored": None,
    "computed": None,
}
_LIVE_TOTALS: dict[str, Any] = {
    "requests": 0,
    "generated_tokens": 0,
    "decode_s": 0.0,
    "prefill_tokens_physical": 0,
    "prefill_tokens_restored": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}

@dataclass(frozen=True)
class _RequestAccounting:
    request_id: int
    mode_used: str
    wall_ms: float
    max_tokens: int
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    ttft_ms: Optional[float] = None
    prefill_ms: Optional[float] = None
    prefill_tok_s: Optional[float] = None
    prefill_tok_s_physical: Optional[float] = None
    prefill_tok_s_apparent: Optional[float] = None
    prefill_tok_s_restored: Optional[float] = None
    decode_ms: Optional[float] = None
    decode_tok_s: Optional[float] = None
    acceptance_ratio: Optional[float] = None
    tokens_per_cycle: Optional[float] = None
    cycles_completed: Optional[int] = None
    adaptive_block_reductions: int = 0
    adaptive_block_cycles: int = 0
    adaptive_block_min: Optional[int] = None
    adaptive_metrics: Optional[dict[str, Any]] = None
    copyspec_hits: int = 0
    copyspec_tokens: int = 0
    finish_reason: Optional[str] = None
    cache_lookup_ms: Optional[float] = None
    cache_hit_tokens: int = 0
    hit_kind: str = "miss"
    cache_insert_ms: Optional[float] = None
    prompt_regime: Optional[dict[str, Any]] = None
    prefill_event_payload: Optional[dict[str, Any]] = None
    prefill_accounting: Optional[dict[str, int]] = None
    prefill_phase_timings_us: Optional[dict[str, float]] = None
    phase_timings_us: Optional[dict[str, float]] = None
    cycle_profile_totals_us: Optional[dict[str, float]] = None
    runtime_config: Optional[dict[str, Any]] = None
    memory_waterfall_peak: Optional[dict[str, Any]] = None
    memory_waterfall_start: Optional[dict[str, Any]] = None
    memory_waterfall_end: Optional[dict[str, Any]] = None

    @classmethod
    def target_only(
        cls,
        *,
        request_id: int,
        mode_used: str,
        wall_ms: float,
        max_tokens: int,
    ) -> "_RequestAccounting":
        return cls(
            request_id=int(request_id),
            mode_used=mode_used,
            wall_ms=float(wall_ms),
            max_tokens=int(max_tokens),
        )

    @classmethod
    def dflash(
        cls,
        *,
        request_id: int,
        summary_event: Optional[SummaryEvent],
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
        prompt_regime: Optional[dict[str, Any]],
        memory_waterfall_peak: Optional[dict[str, Any]],
        memory_waterfall_start: Optional[dict[str, Any]] = None,
        memory_waterfall_end: Optional[dict[str, Any]] = None,
        prefill_event: Optional[PrefillCompleteEvent] = None,
        runtime_config: Optional[Any] = None,
    ) -> "_RequestAccounting":
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
        prefill_tok_s = (
            prompt_token_count / (float(prefill_ms) / 1_000.0)
            if prefill_ms is not None and float(prefill_ms) > 0.0
            else None
        )
        decode_ms = (
            (request_done_ns - prefill_done_ns) / 1e6
            if prefill_done_ns is not None
            else None
        )
        generation_tokens = int(
            summary_event.generation_tokens
            if summary_event is not None
            else live_token_count
        )
        acceptance_ratio = float(summary_event.acceptance_ratio if summary_event else 0.0)
        cycles_completed = int(summary_event.cycles_completed if summary_event else 0)
        tokens_per_cycle = float(summary_event.tokens_per_cycle if summary_event else 0.0)
        adaptive_block_reductions = int(
            summary_event.adaptive_block_reductions if summary_event else 0
        )
        adaptive_block_cycles = int(
            summary_event.adaptive_block_cycles if summary_event else 0
        )
        adaptive_block_min = (
            summary_event.adaptive_block_min if summary_event is not None else None
        )
        adaptive_metrics = (
            dict(summary_event.adaptive_metrics)
            if summary_event is not None and summary_event.adaptive_metrics
            else {}
        )
        copyspec_hits = int(summary_event.copyspec_hits if summary_event else 0)
        copyspec_tokens = int(summary_event.copyspec_tokens if summary_event else 0)
        cycle_profile_totals_us = (
            dict(summary_event.cycle_profile_totals_us)
            if summary_event is not None
            and summary_event.cycle_profile_totals_us is not None
            else {}
        )
        prefill_phase_timings_us = _prefill_phase_timings(prefill_event)
        prefill_accounting = _prefill_accounting(prefill_event)
        runtime_config_payload = _runtime_config_payload(runtime_config)
        logical_prefill_tokens = int(
            prefill_accounting.get("logical_ctx_tokens", prompt_token_count)
        )
        physical_prefill_tokens = prefill_accounting.get("physical_prefill_tokens")
        restored_prefill_tokens = prefill_accounting.get("prefill_tokens_restored")
        prefill_tok_s_apparent = (
            logical_prefill_tokens / (float(prefill_ms) / 1_000.0)
            if prefill_ms is not None and float(prefill_ms) > 0.0
            else None
        )
        prefill_tok_s_physical = (
            int(physical_prefill_tokens) / (float(prefill_ms) / 1_000.0)
            if physical_prefill_tokens is not None
            and prefill_ms is not None
            and float(prefill_ms) > 0.0
            else None
        )
        prefill_tok_s_restored = (
            int(restored_prefill_tokens) / (float(prefill_ms) / 1_000.0)
            if restored_prefill_tokens is not None
            and prefill_ms is not None
            and float(prefill_ms) > 0.0
            else None
        )
        decode_tok_s = (
            generation_tokens / (float(decode_ms) / 1_000.0)
            if decode_ms is not None and float(decode_ms) > 0.0
            else None
        )
        fallback_used = bool(summary_event.fallback_ar if summary_event else False)
        hit_kind = str(summary_event.hit_kind if summary_event is not None else "miss")
        return cls(
            request_id=int(request_id),
            mode_used="dflash_fallback" if fallback_used else "dflash",
            prompt_tokens=int(prompt_token_count),
            generated_tokens=generation_tokens,
            wall_ms=wall_ms,
            ttft_ms=ttft_ms,
            prefill_ms=prefill_ms,
            prefill_tok_s=prefill_tok_s,
            prefill_tok_s_physical=prefill_tok_s_physical,
            prefill_tok_s_apparent=prefill_tok_s_apparent,
            prefill_tok_s_restored=prefill_tok_s_restored,
            decode_ms=decode_ms,
            decode_tok_s=decode_tok_s,
            acceptance_ratio=acceptance_ratio,
            tokens_per_cycle=tokens_per_cycle,
            cycles_completed=cycles_completed,
            adaptive_block_reductions=adaptive_block_reductions,
            adaptive_block_cycles=adaptive_block_cycles,
            adaptive_block_min=adaptive_block_min,
            adaptive_metrics=adaptive_metrics,
            copyspec_hits=copyspec_hits,
            copyspec_tokens=copyspec_tokens,
            finish_reason=finish_reason,
            max_tokens=int(max_tokens),
            cache_lookup_ms=float(cache_lookup_ms),
            cache_hit_tokens=int(cache_hit_tokens),
            hit_kind=hit_kind,
            cache_insert_ms=float(cache_insert_ms),
            prompt_regime=dict(prompt_regime or {}),
            prefill_event_payload=(
                prefill_event.to_payload() if prefill_event is not None else {}
            ),
            prefill_accounting=dict(prefill_accounting),
            prefill_phase_timings_us=dict(prefill_phase_timings_us),
            phase_timings_us=dict(summary_event.phase_timings_us if summary_event else {}),
            cycle_profile_totals_us=cycle_profile_totals_us,
            runtime_config=runtime_config_payload,
            memory_waterfall_peak=dict(memory_waterfall_peak or {}),
            memory_waterfall_start=dict(memory_waterfall_start or {}),
            memory_waterfall_end=dict(memory_waterfall_end or {}),
        )

    @property
    def physical_prefill_tokens(self) -> Optional[int]:
        return _int_or_none((self.prefill_accounting or {}).get("physical_prefill_tokens"))

    @property
    def restored_prefill_tokens(self) -> Optional[int]:
        return _int_or_none((self.prefill_accounting or {}).get("prefill_tokens_restored"))

    @property
    def computed_prefill_tokens(self) -> Optional[int]:
        return _int_or_none((self.prefill_accounting or {}).get("prefill_tokens_computed"))

    @property
    def prefix_cache_enabled(self) -> bool:
        runtime_config = self.runtime_config or {}
        return bool(
            runtime_config.get("prefix_cache", False)
            and int(runtime_config.get("target_fa_window", 0) or 0) <= 0
        )

    def last_request_payload(self) -> dict[str, Any]:
        payload = _last_request_payload(
            request_id=self.request_id,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=self.generated_tokens,
            wall_ms=self.wall_ms,
            ttft_ms=self.ttft_ms,
            prefill_ms=self.prefill_ms,
            decode_ms=self.decode_ms,
            prefill_tok_s_physical=self.prefill_tok_s_physical,
            prefill_tok_s_apparent=self.prefill_tok_s_apparent,
            prefill_tok_s_restored=self.prefill_tok_s_restored,
            prefill_tokens_physical=self.physical_prefill_tokens,
            prefill_tokens_restored=self.restored_prefill_tokens,
            prefill_tokens_computed=self.computed_prefill_tokens,
            decode_tok_s=self.decode_tok_s,
            acceptance_rate=self.acceptance_ratio,
            tokens_per_cycle=self.tokens_per_cycle,
            cycles=self.cycles_completed,
            adaptive_block_reductions=self.adaptive_block_reductions,
            adaptive_block_cycles=self.adaptive_block_cycles,
            adaptive_block_min=self.adaptive_block_min,
            copyspec_hits=self.copyspec_hits,
            copyspec_tokens=self.copyspec_tokens,
            finish_reason=self.finish_reason,
            cache_hit_tokens=self.cache_hit_tokens,
            hit_kind=self.hit_kind,
            prefill_phase_timings_us=self.prefill_phase_timings_us,
            phase_timings_us=self.phase_timings_us,
        )
        if self.cycle_profile_totals_us:
            payload["cycle_profile_totals_us"] = dict(self.cycle_profile_totals_us)
        if self.adaptive_metrics:
            payload["adaptive_metrics"] = dict(self.adaptive_metrics)
        payload["mode_used"] = self.mode_used
        payload["max_tokens"] = int(self.max_tokens)
        return payload

    def post_event_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": int(self.request_id),
            "mode_used": self.mode_used,
            "max_tokens": int(self.max_tokens),
            "wall_ms": float(self.wall_ms),
            "cache_hit_tokens": int(self.cache_hit_tokens),
            "cache_status": _cache_status(self.cache_hit_tokens),
            "hit_kind": self.hit_kind,
        }
        if self.prompt_tokens is None:
            return payload
        payload.update(
            {
                "prompt_tokens": int(self.prompt_tokens),
                "generated_tokens": int(self.generated_tokens or 0),
                "ttft_ms": self.ttft_ms,
                "prefill_ms": self.prefill_ms,
                "prefill_tok_s": self.prefill_tok_s,
                "prefill_tok_s_physical": self.prefill_tok_s_physical,
                "prefill_tok_s_apparent": self.prefill_tok_s_apparent,
                "prefill_tok_s_restored": self.prefill_tok_s_restored,
                "decode_tok_s": self.decode_tok_s,
                "decode_ms": self.decode_ms,
                "cache_lookup_ms": self.cache_lookup_ms,
                "cache_insert_ms": self.cache_insert_ms,
                "acceptance_ratio": self.acceptance_ratio,
                "tokens_per_cycle": self.tokens_per_cycle,
                "cycles_completed": self.cycles_completed,
                "adaptive_block_reductions": int(self.adaptive_block_reductions),
                "adaptive_block_cycles": int(self.adaptive_block_cycles),
                "adaptive_block_min": self.adaptive_block_min,
                "adaptive_metrics": dict(self.adaptive_metrics or {}),
                "copyspec_hits": int(self.copyspec_hits),
                "copyspec_tokens": int(self.copyspec_tokens),
                "finish_reason": self.finish_reason,
                "prompt_regime": dict(self.prompt_regime or {}),
                "prefill_event": dict(self.prefill_event_payload or {}),
                **dict(self.prefill_accounting or {}),
                "prefill_phase_timings_us": dict(self.prefill_phase_timings_us or {}),
                "phase_timings_us": dict(self.phase_timings_us or {}),
                "cycle_profile_totals_us": dict(self.cycle_profile_totals_us or {}),
                "runtime_config": dict(self.runtime_config or {}),
                "memory_waterfall_peak": dict(self.memory_waterfall_peak or {}),
                "memory_waterfall_start": dict(self.memory_waterfall_start or {}),
                "memory_waterfall_end": dict(self.memory_waterfall_end or {}),
                "memory_boundary_start": dict(self.memory_waterfall_start or {}),
                "memory_boundary_end": dict(self.memory_waterfall_end or {}),
            }
        )
        return payload

def configure_live_metrics(
    *,
    version: str,
    model_provider: Any,
) -> None:
    cli_args = model_provider.cli_args
    runtime_config = getattr(cli_args, "runtime_config", None)
    model_key = getattr(model_provider, "model_key", None) or ()
    target_ref = model_key[0] if len(model_key) > 0 else getattr(cli_args, "model", None)
    draft_ref = model_key[2] if len(model_key) > 2 else getattr(cli_args, "draft_model", None)
    draft_meta = getattr(model_provider, "draft_meta", {}) or {}
    target_meta = getattr(model_provider, "target_meta", {}) or {}
    target_fa_window = _int_or_none(getattr(runtime_config, "target_fa_window", None))
    prefix_cache_enabled = bool(
        runtime_config is not None
        and getattr(runtime_config, "prefix_cache", False)
        and int(target_fa_window or 0) <= 0
    )
    runtime_context = getattr(cli_args, "runtime_context", None)
    cache_identity: Any = tuple(model_key)
    draft_model = getattr(model_provider, "draft_model", None)
    if prefix_cache_enabled and runtime_context is not None and draft_model is not None:
        cache_identity = build_prefix_key(model_provider, draft_model, runtime_context)
    sync_runtime_cache_manager(runtime_context, cache_identity=cache_identity)
    metal_limits = getattr(cli_args, "metal_limits", None)
    with _LIVE_LOCK:
        global _LIVE_STARTED_AT
        _LIVE_STARTED_AT = time.time()
        _LIVE_SERVER.update(
            {
                "version": version,
                "model": str(target_ref) if target_ref is not None else None,
                "draft": str(draft_ref) if draft_ref is not None else None,
                "draft_quant": getattr(model_provider, "effective_draft_quant", None),
                "draft_quant_source": draft_meta.get("draft_quant_source"),
                "draft_load_dtype": draft_meta.get("draft_load_dtype"),
                "draft_load_dtype_source": draft_meta.get("draft_load_dtype_source"),
                "mode": "dflash",
            }
        )
        _LIVE_RUNTIME.update(
            {
                "prefill_step_size": _int_or_none(
                    getattr(runtime_config, "prefill_step_size", None)
                ),
                "prefix_cache_enabled": prefix_cache_enabled,
                "target_fa_window": (
                    None if int(target_fa_window or 0) <= 0 else int(target_fa_window)
                ),
                "diagnostics": getattr(cli_args, "diagnostics", None),
            }
        )
        _LIVE_PREFIX_CONFIG.update(
            {
                "enabled": prefix_cache_enabled,
                "max_entries": _int_or_none(
                    getattr(runtime_config, "prefix_cache_max_entries", None)
                ),
                "max_bytes": _int_or_none(
                    getattr(runtime_config, "prefix_cache_max_bytes", None)
                ),
            }
        )
        _LIVE_METAL_LIMITS.update(
            {
                "wired_limit_bytes": _int_or_none(
                    getattr(metal_limits, "wired_bytes", None)
                ),
            }
        )

def get_live_metrics_payload() -> dict[str, Any]:
    prefix_cache_manager = current_runtime_cache_manager()
    raw_prefix_stats = _prefix_cache_stats(prefix_cache_manager)
    now = time.time()
    with _LIVE_LOCK:
        server = dict(_LIVE_SERVER)
        server["uptime_s"] = max(0.0, now - _LIVE_STARTED_AT)
        started_at = _LIVE_STARTED_AT
        runtime = dict(_LIVE_RUNTIME)
        last_request = None if _LIVE_LAST_REQUEST is None else dict(_LIVE_LAST_REQUEST)
        current_request = _current_request_payload(_LIVE_CURRENT_REQUEST, now)
        recent_requests = [dict(request) for request in _LIVE_RECENT_REQUESTS]
        totals = dict(_LIVE_TOTALS)
        metal_limits = dict(_LIVE_METAL_LIMITS)
        finish_times = list(_LIVE_REQUEST_FINISH_TIMES)
        prefix_config = dict(_LIVE_PREFIX_CONFIG)
        last_prefix = dict(_LIVE_LAST_PREFIX)
    prefix_stats = live_prefix_cache_payload(
        stats=raw_prefix_stats,
        config=prefix_config,
        last_request=last_request,
        last_prefix=last_prefix,
    )
    payload = {
        "server": server,
        "runtime": runtime,
        "memory": _memory_payload(metal_limits),
        "current_request": current_request,
        "last_request": last_request,
        "recent_requests": recent_requests,
        "rates": _rates_payload(
            totals=totals,
            current_request=current_request,
            last_request=last_request,
            finish_times=finish_times,
            started_at=started_at,
            now=now,
        ),
        "prefix_cache": prefix_stats,
        "totals": totals,
    }
    payload["totals"].update(live_prefix_cache_totals(raw_prefix_stats))
    return payload

def start_live_request(
    *,
    request_id: int,
    mode_used: str,
    prompt_tokens: Optional[int],
    max_tokens: int,
    cache_hit_tokens: int = 0,
    cache_lookup_ms: float = 0.0,
    hit_kind: str = "miss",
) -> None:
    current = {
        "request_id": int(request_id),
        "mode_used": mode_used,
        "state": "queued",
        "prompt_tokens": _int_or_none(prompt_tokens),
        "max_tokens": int(max_tokens),
        "generated_tokens": 0,
        "cache_hit_tokens": int(cache_hit_tokens),
        "cache_status": _cache_status(cache_hit_tokens),
        "hit_kind": str(hit_kind),
        "cache_lookup_ms": float(cache_lookup_ms),
        "prefill_tokens_processed": None,
        "prefill_tokens_total": _int_or_none(prompt_tokens),
        "prefill_s": None,
        "ttft_s": None,
        "decode_s": None,
        "decode_tok_s": None,
        "acceptance_rate": None,
        "tokens_per_cycle": None,
        "cycles": None,
        "adaptive_block_reductions": 0,
        "adaptive_block_cycles": 0,
        "adaptive_block_min": None,
        "copyspec_hits": 0,
        "copyspec_tokens": 0,
        "finish_reason": None,
        "prefill_phase_timings_us": {},
        "phase_timings_us": {},
        "_started_at": time.time(),
        "_prefill_done_at": None,
    }
    with _LIVE_LOCK:
        global _LIVE_CURRENT_REQUEST
        _LIVE_CURRENT_REQUEST = current

def update_live_request(
    *,
    request_id: int,
    state: Optional[str] = None,
    generated_tokens: Optional[int] = None,
    prefill_tokens_processed: Optional[int] = None,
    prefill_tokens_total: Optional[int] = None,
    prefill_s: Optional[float] = None,
    ttft_s: Optional[float] = None,
    prefill_done: bool = False,
    decode_tok_s: Optional[float] = None,
    acceptance_rate: Optional[float] = None,
    tokens_per_cycle: Optional[float] = None,
    cycles: Optional[int] = None,
    adaptive_block_reductions: Optional[int] = None,
    adaptive_block_cycles: Optional[int] = None,
    adaptive_block_min: Optional[int] = None,
    copyspec_hits: Optional[int] = None,
    copyspec_tokens: Optional[int] = None,
    finish_reason: Optional[str] = None,
    prefill_phase_timings_us: Optional[dict[str, float]] = None,
    phase_timings_us: Optional[dict[str, float]] = None,
) -> None:
    with _LIVE_LOCK:
        current = _LIVE_CURRENT_REQUEST
        if current is None or int(current.get("request_id", -1)) != int(request_id):
            return
        if state is not None:
            current["state"] = state
        if generated_tokens is not None:
            current["generated_tokens"] = int(generated_tokens)
        if prefill_tokens_processed is not None:
            current["prefill_tokens_processed"] = int(prefill_tokens_processed)
        if prefill_tokens_total is not None:
            current["prefill_tokens_total"] = int(prefill_tokens_total)
        if prefill_s is not None:
            current["prefill_s"] = float(prefill_s)
        if ttft_s is not None:
            current["ttft_s"] = float(ttft_s)
        if prefill_done:
            current["_prefill_done_at"] = time.time()
        if decode_tok_s is not None:
            current["decode_tok_s"] = float(decode_tok_s)
        if acceptance_rate is not None:
            current["acceptance_rate"] = float(acceptance_rate)
        if tokens_per_cycle is not None:
            current["tokens_per_cycle"] = float(tokens_per_cycle)
        if cycles is not None:
            current["cycles"] = int(cycles)
        if adaptive_block_reductions is not None:
            current["adaptive_block_reductions"] = int(adaptive_block_reductions)
        if adaptive_block_cycles is not None:
            current["adaptive_block_cycles"] = int(adaptive_block_cycles)
        if adaptive_block_min is not None:
            current["adaptive_block_min"] = int(adaptive_block_min)
        if copyspec_hits is not None:
            current["copyspec_hits"] = int(copyspec_hits)
        if copyspec_tokens is not None:
            current["copyspec_tokens"] = int(copyspec_tokens)
        if finish_reason is not None:
            current["finish_reason"] = finish_reason
        if prefill_phase_timings_us is not None:
            current["prefill_phase_timings_us"] = dict(prefill_phase_timings_us)
        if phase_timings_us is not None:
            current["phase_timings_us"] = dict(phase_timings_us)

def clear_live_request(*, request_id: int) -> None:
    with _LIVE_LOCK:
        global _LIVE_CURRENT_REQUEST
        if (
            _LIVE_CURRENT_REQUEST is not None
            and int(_LIVE_CURRENT_REQUEST.get("request_id", -1)) == int(request_id)
        ):
            _LIVE_CURRENT_REQUEST = None

def record_target_only_request(
    *,
    request_id: int,
    mode_used: str,
    wall_ms: float,
    max_tokens: int,
    diagnostics: Optional[DiagnosticsConfig] = None,
) -> None:
    accounting = _RequestAccounting.target_only(
        request_id=request_id,
        mode_used=mode_used,
        wall_ms=wall_ms,
        max_tokens=max_tokens,
    )
    _record_live_accounting(accounting)
    _bench_log_post(
        diagnostics.trace if diagnostics is not None else None,
        **accounting.post_event_payload(),
    )

def record_cycle_diagnostic(
    *,
    diagnostics: Optional[DiagnosticsConfig],
    request_id: int,
    fields: dict[str, Any],
) -> None:
    _bench_log_cycle(
        diagnostics.trace if diagnostics is not None else None,
        request_id=request_id,
        **fields,
    )

def finalize_request_observability(
    *,
    request_id: int,
    summary_event: Optional[SummaryEvent],
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
    memory_waterfall_start: Optional[dict[str, Any]] = None,
    memory_waterfall_end: Optional[dict[str, Any]] = None,
    diagnostics: Optional[DiagnosticsConfig] = None,
    prefill_event: Optional[PrefillCompleteEvent] = None,
    runtime_config: Optional[Any] = None,
) -> None:
    accounting = _RequestAccounting.dflash(
        request_id=request_id,
        summary_event=summary_event,
        request_start_ns=request_start_ns,
        request_done_ns=request_done_ns,
        first_token_ns=first_token_ns,
        prefill_done_ns=prefill_done_ns,
        prompt_token_count=prompt_token_count,
        live_token_count=live_token_count,
        cache_lookup_ms=cache_lookup_ms,
        cache_hit_tokens=cache_hit_tokens,
        cache_insert_ms=cache_insert_ms,
        finish_reason=finish_reason,
        max_tokens=max_tokens,
        prompt_regime=prompt_regime,
        memory_waterfall_peak=memory_waterfall_peak,
        memory_waterfall_start=memory_waterfall_start,
        memory_waterfall_end=memory_waterfall_end,
        prefill_event=prefill_event,
        runtime_config=runtime_config,
    )
    if summary_event is not None:
        _write_summary_line_from_accounting(accounting)
    _record_request_accounting(accounting, diagnostics=diagnostics)
    write_post_request_memory_line(request_id=request_id)

def _reset_live_metrics_state() -> None:
    with _LIVE_LOCK:
        global _LIVE_STARTED_AT, _LIVE_LAST_REQUEST, _LIVE_CURRENT_REQUEST
        _LIVE_STARTED_AT = time.time()
        _LIVE_SERVER.update(
            {
                "version": None,
                "model": None,
                "draft": None,
                "mode": "dflash",
            }
        )
        _LIVE_RUNTIME.update(
            {
                "prefill_step_size": None,
                "prefix_cache_enabled": None,
                "target_fa_window": None,
                "diagnostics": None,
            }
        )
        _LIVE_PREFIX_CONFIG.update(
            {"enabled": False, "max_entries": None, "max_bytes": None}
        )
        _LIVE_METAL_LIMITS.update({"wired_limit_bytes": None})
        _LIVE_LAST_REQUEST = None
        _LIVE_CURRENT_REQUEST = None
        _LIVE_RECENT_REQUESTS.clear()
        _LIVE_REQUEST_FINISH_TIMES.clear()
        _LIVE_LAST_PREFIX.update({"restored": None, "computed": None})
        for key in _LIVE_TOTALS:
            _LIVE_TOTALS[key] = 0.0 if key == "decode_s" else 0

def _write_summary_line_from_accounting(accounting: _RequestAccounting) -> None:
    generation_tokens = int(accounting.generated_tokens or 0)
    prompt_tokens = int(accounting.prompt_tokens or 0)
    prefill_tok_s_apparent = _float_or_none(accounting.prefill_tok_s_apparent)
    if prefill_tok_s_apparent is None:
        prefill_tok_s_apparent = _float_or_none(accounting.prefill_tok_s) or 0.0
    prefill_tok_s_physical = _float_or_none(accounting.prefill_tok_s_physical) or 0.0
    prefill_tok_s_restored = _float_or_none(accounting.prefill_tok_s_restored) or 0.0
    tok_s = _float_or_none(accounting.decode_tok_s) or 0.0
    acceptance_pct = (_float_or_none(accounting.acceptance_ratio) or 0.0) * 100.0
    total_s = float(accounting.wall_ms) / 1_000.0
    _write_observability_line(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] decode {tok_s:.1f} tok/s | "
        f"prefill logical {prefill_tok_s_apparent:.1f} tok/s | "
        f"prefill real {prefill_tok_s_physical:.1f} tok/s | "
        f"prefill restored {prefill_tok_s_restored:.1f} tok/s | "
        f"{acceptance_pct:.1f}% accepted | {generation_tokens} tokens | "
        f"{total_s:.1f}s | prompt: {prompt_tokens} tokens\n"
    )


def write_post_request_memory_line(*, request_id: int) -> None:
    try:
        memory = get_memory_snapshot()
    except Exception as exc:
        _write_observability_line(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] req#{request_id} memory snapshot failed: {exc}\n"
        )
        return
    unavailable = [
        key
        for key in (
            "mlx_active_gb",
            "mlx_cache_gb",
            "mlx_peak_gb",
            "rss_gb",
            "rss_peak_gb",
        )
        if memory.get(key) is None
    ]
    if unavailable:
        _write_observability_line(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] req#{request_id} memory snapshot partial "
            f"unavailable={','.join(unavailable)}\n"
        )
    mlx_active_gb = _float_or_none(memory.get("mlx_active_gb"))
    mlx_cache_gb = _float_or_none(memory.get("mlx_cache_gb"))
    mlx_peak_gb = _float_or_none(memory.get("mlx_peak_gb"))
    rss_now_gb = _float_or_none(memory.get("rss_gb"))
    rss_peak_gb = _float_or_none(memory.get("rss_peak_gb"))
    untracked_gb = (
        max(0.0, rss_now_gb - mlx_active_gb - mlx_cache_gb)
        if (
            rss_now_gb is not None
            and mlx_active_gb is not None
            and mlx_cache_gb is not None
        )
        else None
    )
    _write_observability_line(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] req#{request_id} "
        f"mlx_active={_fmt_gb_value(mlx_active_gb)} "
        f"mlx_cache={_fmt_gb_value(mlx_cache_gb)} "
        f"mlx_peak={_fmt_gb_value(mlx_peak_gb)} "
        f"rss_now={_fmt_gb_value(rss_now_gb)} "
        f"rss_peak={_fmt_gb_value(rss_peak_gb)} "
        f"untracked={_fmt_gb_value(untracked_gb)} GB\n"
    )

def _write_observability_line(line: str) -> None:
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception as exc:
        fallback = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] observability stderr write failed: {exc}\n"
        )
        try:
            os.write(2, fallback.encode())
        except OSError:
            return


def _record_request_accounting(
    accounting: _RequestAccounting,
    *,
    diagnostics: Optional[DiagnosticsConfig],
) -> None:
    _record_live_accounting(accounting)
    _bench_log_post(
        diagnostics.trace if diagnostics is not None else None,
        **accounting.post_event_payload(),
    )
    _append_diagnostics_summary(accounting=accounting, diagnostics=diagnostics)

def _append_diagnostics_summary(
    *,
    accounting: _RequestAccounting,
    diagnostics: Optional[DiagnosticsConfig],
) -> None:
    if diagnostics is None or diagnostics.run_dir is None:
        return
    path = Path(diagnostics.run_dir) / "summary.md"
    runtime_config = accounting.runtime_config or {}
    prefill_step_size = runtime_config.get("prefill_step_size", "")
    line = (
        f"| {accounting.request_id} | {_fmt_int(accounting.prompt_tokens)} | "
        f"{_fmt_int(accounting.generated_tokens)} | {_fmt_ms(accounting.wall_ms)} | "
        f"{_fmt_ms(accounting.ttft_ms)} | {_fmt_ms(accounting.prefill_ms)} | "
        f"{_fmt_float(accounting.prefill_tok_s_apparent)} | "
        f"{_fmt_float(accounting.prefill_tok_s_physical)} | "
        f"{_fmt_float(accounting.prefill_tok_s_restored)} | {prefill_step_size} | "
        f"{_fmt_ms(accounting.decode_ms)} | "
        f"{_fmt_float3(accounting.acceptance_ratio)} | "
        f"{_fmt_float(accounting.tokens_per_cycle)} | "
        f"{_fmt_int(accounting.cycles_completed)} | {accounting.cache_hit_tokens} |\n"
    )
    try:
        current = path.read_text(errors="ignore") if path.exists() else ""
        with path.open("a") as fp:
            if "| request |" not in current:
                fp.write(
                    "\n## Requests\n\n"
                    "| request | prompt_tokens | generated | wall_ms | ttft_ms | "
                    "prefill_ms | prefill_logical_tok_s | prefill_real_tok_s | "
                    "prefill_restored_tok_s | prefill_step | decode_ms | "
                    "acceptance | tokens_cycle | cycles | cache_hit_tokens |\n"
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
                )
            fp.write(line)
    except OSError as exc:
        report_observability_failure("diagnostics summary append failed", exc)

def _prefill_phase_timings(prefill_event: Optional[PrefillCompleteEvent]) -> dict[str, float]:
    if not prefill_event:
        return {}
    return prefill_event.phase_timings()

def _prefill_accounting(prefill_event: Optional[PrefillCompleteEvent]) -> dict[str, int]:
    if not prefill_event:
        return {}
    return prefill_event.accounting()

def _runtime_config_payload(runtime_config: Optional[Any]) -> dict[str, Any]:
    if runtime_config is None:
        return {}
    keys = (
        "prefill_step_size",
        "draft_sink_size",
        "draft_window_size",
        "verify_len_cap",
        "prefix_cache",
        "prefix_cache_l2",
        "target_fa_window",
        "dflash_max_ctx",
        "max_snapshot_tokens",
        "clear_cache_boundaries",
        "verify_mode",
    )
    payload: dict[str, Any] = {}
    for key in keys:
        if hasattr(runtime_config, key):
            payload[key] = getattr(runtime_config, key)
    return payload

def _record_live_accounting(accounting: _RequestAccounting) -> None:
    last_request = accounting.last_request_payload()
    with _LIVE_LOCK:
        _LIVE_TOTALS["requests"] += 1
        generated_tokens = int(accounting.generated_tokens or 0)
        if accounting.generated_tokens is not None:
            _LIVE_TOTALS["generated_tokens"] += generated_tokens
        if (
            generated_tokens > 0
            and accounting.decode_ms is not None
            and float(accounting.decode_ms) > 0.0
        ):
            _LIVE_TOTALS["decode_s"] += float(accounting.decode_ms) / 1_000.0
        if accounting.physical_prefill_tokens is not None:
            _LIVE_TOTALS["prefill_tokens_physical"] += int(
                accounting.physical_prefill_tokens
            )
        if accounting.restored_prefill_tokens is not None:
            _LIVE_TOTALS["prefill_tokens_restored"] += int(
                accounting.restored_prefill_tokens
            )
        _LIVE_LAST_PREFIX.update(
            {
                "restored": _int_or_none(accounting.restored_prefill_tokens),
                "computed": _int_or_none(accounting.computed_prefill_tokens),
            }
        )
        if accounting.prefix_cache_enabled:
            if int(accounting.cache_hit_tokens) > 0:
                _LIVE_TOTALS["cache_hits"] += 1
            else:
                _LIVE_TOTALS["cache_misses"] += 1
        _finish_live_request_unlocked(last_request)

def _finish_live_request_unlocked(last_request: dict[str, Any]) -> None:
    global _LIVE_LAST_REQUEST, _LIVE_CURRENT_REQUEST
    _LIVE_LAST_REQUEST = dict(last_request)
    _LIVE_RECENT_REQUESTS.append(dict(last_request))
    _LIVE_REQUEST_FINISH_TIMES.append(time.time())
    if (
        _LIVE_CURRENT_REQUEST is not None
        and _LIVE_CURRENT_REQUEST.get("request_id") == last_request.get("request_id")
    ):
        _LIVE_CURRENT_REQUEST = None

def _rates_payload(
    *,
    totals: dict[str, Any],
    current_request: Optional[dict[str, Any]],
    last_request: Optional[dict[str, Any]],
    finish_times: list[float],
    started_at: float,
    now: float,
) -> dict[str, Optional[float]]:
    uptime_s = max(0.0, now - started_at)
    requests = int(totals.get("requests", 0) or 0)
    recent_window_s = min(60.0, uptime_s)
    recent_cutoff = now - 60.0
    recent_requests = sum(1 for ts in finish_times if ts >= recent_cutoff)
    source = current_request if current_request is not None else last_request
    return {
        "requests_per_s": _rate(requests, uptime_s),
        "requests_per_s_60s": _rate(recent_requests, recent_window_s),
        "generated_tokens_per_s": _rate(
            int(totals.get("generated_tokens", 0) or 0),
            uptime_s,
        ),
        "average_decode_tok_s": _rate(
            int(totals.get("generated_tokens", 0) or 0),
            float(totals.get("decode_s", 0.0) or 0.0),
        ),
        "prefill_tokens_physical_per_s": _rate(
            int(totals.get("prefill_tokens_physical", 0) or 0),
            uptime_s,
        ),
        "prefill_tokens_restored_per_s": _rate(
            int(totals.get("prefill_tokens_restored", 0) or 0),
            uptime_s,
        ),
        "active_decode_tok_s": _float_or_none(
            None if source is None else source.get("decode_tok_s")
        ),
    }

def _current_request_payload(
    current: Optional[dict[str, Any]],
    now: float,
) -> Optional[dict[str, Any]]:
    if current is None:
        return None
    started_at = float(current.get("_started_at", now) or now)
    prefill_done_at = current.get("_prefill_done_at")
    payload = {
        key: value
        for key, value in current.items()
        if not str(key).startswith("_")
    }
    payload["elapsed_s"] = max(0.0, now - started_at)
    if prefill_done_at is not None:
        payload["decode_s"] = max(0.0, now - float(prefill_done_at))
    return payload

def _last_request_payload(
    *,
    request_id: int,
    prompt_tokens: Optional[int],
    generated_tokens: Optional[int],
    wall_ms: Optional[float],
    ttft_ms: Optional[float],
    prefill_ms: Optional[float],
    decode_ms: Optional[float],
    prefill_tok_s_physical: Optional[float],
    prefill_tok_s_apparent: Optional[float],
    prefill_tok_s_restored: Optional[float],
    prefill_tokens_physical: Optional[int],
    prefill_tokens_restored: Optional[int],
    prefill_tokens_computed: Optional[int],
    decode_tok_s: Optional[float],
    acceptance_rate: Optional[float],
    tokens_per_cycle: Optional[float],
    cycles: Optional[int],
    adaptive_block_reductions: int,
    adaptive_block_cycles: int,
    adaptive_block_min: Optional[int],
    copyspec_hits: int,
    copyspec_tokens: int,
    finish_reason: Optional[str],
    cache_hit_tokens: int = 0,
    hit_kind: str = "miss",
    prefill_phase_timings_us: Optional[dict[str, float]] = None,
    phase_timings_us: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    return {
        "request_id": int(request_id),
        "prompt_tokens": _int_or_none(prompt_tokens),
        "generated_tokens": _int_or_none(generated_tokens),
        "wall_s": _seconds_or_none(wall_ms),
        "ttft_s": _seconds_or_none(ttft_ms),
        "prefill_s": _seconds_or_none(prefill_ms),
        "decode_s": _seconds_or_none(decode_ms),
        "prefill_tok_s_physical": _float_or_none(prefill_tok_s_physical),
        "prefill_tok_s_apparent": _float_or_none(prefill_tok_s_apparent),
        "prefill_tok_s_restored": _float_or_none(prefill_tok_s_restored),
        "prefill_tokens_physical": _int_or_none(prefill_tokens_physical),
        "prefill_tokens_restored": _int_or_none(prefill_tokens_restored),
        "prefill_tokens_computed": _int_or_none(prefill_tokens_computed),
        "decode_tok_s": _float_or_none(decode_tok_s),
        "acceptance_rate": _float_or_none(acceptance_rate),
        "tokens_per_cycle": _float_or_none(tokens_per_cycle),
        "cycles": _int_or_none(cycles),
        "adaptive_block_reductions": int(adaptive_block_reductions),
        "adaptive_block_cycles": int(adaptive_block_cycles),
        "adaptive_block_min": _int_or_none(adaptive_block_min),
        "copyspec_hits": int(copyspec_hits),
        "copyspec_tokens": int(copyspec_tokens),
        "finish_reason": finish_reason,
        "cache_hit_tokens": int(cache_hit_tokens),
        "cache_status": _cache_status(cache_hit_tokens),
        "hit_kind": str(hit_kind),
        "prefill_phase_timings_us": dict(prefill_phase_timings_us or {}),
        "phase_timings_us": dict(phase_timings_us or {}),
    }

def _prefix_cache_stats(prefix_cache_manager: Optional[Any]) -> Optional[dict[str, Any]]:
    if prefix_cache_manager is None:
        return None
    try:
        return prefix_cache_manager.stats()
    except RuntimeCacheManagerClosed:
        return None

def _memory_payload(metal_limits: dict[str, Any]) -> dict[str, Any]:
    return live_memory_payload(wired_limit_bytes=metal_limits.get("wired_limit_bytes"))

def get_memory_snapshot() -> dict[str, Any]:
    return _memory_payload({"wired_limit_bytes": None})

def _seconds_or_none(value_ms: Optional[float]) -> Optional[float]:
    if value_ms is None:
        return None
    return float(value_ms) / 1_000.0

def _rate(count: int, elapsed_s: float) -> Optional[float]:
    if elapsed_s <= 0.0:
        return None
    return float(count) / float(elapsed_s)

def _float_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _int_or_none(value: Optional[Any]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _cache_status(cache_hit_tokens: int) -> str:
    return "WARM" if int(cache_hit_tokens) > 0 else "COLD"


def _bool_or_none(value: Optional[Any]) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.1f}"

def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return ""
    return str(int(value))

def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"

def _fmt_gb_value(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"

def _fmt_float3(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"

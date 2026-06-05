# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True)
class PrefillProgressEvent:
    tokens_processed: int
    tokens_total: int


@dataclass(frozen=True)
class PrefillCompleteEvent:
    prefill_us: float
    prompt_token_count: int
    logical_ctx_tokens: int
    physical_prefill_tokens: int
    prefill_tokens_restored: int
    prefill_tokens_computed: int
    snap_prefix_len: int = 0
    snapshot_boundary: int = 0
    fallback_ar: bool = False
    fallback_reason: str | None = None
    phase_rebuild_us: float | None = None
    phase_cold_us: float | None = None
    phase_seam_us: float | None = None
    phase_tail_us: float | None = None

    def phase_timings(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in ("phase_rebuild_us", "phase_cold_us", "phase_seam_us", "phase_tail_us"):
            value = getattr(self, key)
            if value is not None:
                out[key] = float(value)
        return out

    def accounting(self) -> dict[str, int]:
        return {
            "logical_ctx_tokens": int(self.logical_ctx_tokens),
            "physical_prefill_tokens": int(self.physical_prefill_tokens),
            "prefill_tokens_restored": int(self.prefill_tokens_restored),
            "prefill_tokens_computed": int(self.prefill_tokens_computed),
        }

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prefill_us": float(self.prefill_us),
            "prompt_token_count": int(self.prompt_token_count),
            "snap_prefix_len": int(self.snap_prefix_len),
            "snapshot_boundary": int(self.snapshot_boundary),
            **self.accounting(),
        }
        if self.fallback_ar:
            payload["fallback_ar"] = True
            payload["fallback_reason"] = self.fallback_reason
        payload.update(self.phase_timings())
        return payload


@dataclass(frozen=True)
class TokenEvent:
    token_id: int
    generated_tokens: int
    acceptance_ratio: float
    cycles_completed: int
    fallback_ar: bool = False
    fallback_reason: str | None = None
    adaptive_block_reductions: int = 0
    adaptive_block_cycles: int = 0
    adaptive_block_min: int | None = None
    copyspec_hits: int = 0
    copyspec_tokens: int = 0


@dataclass(frozen=True)
class SnapshotPublishedEvent:
    kind: Literal["prefill", "generation"]
    snapshot_boundary: int
    prefix_len: int
    insert_ms: float
    admitted: bool
    from_snapshot: bool = False
    snap_prefix_len: int = 0


@dataclass(frozen=True)
class CycleCompleteEvent:
    cycle: int
    block_len: int
    commit_count: int
    acceptance_len: int
    draft_us: float
    verify_us: float
    acceptance_us: float
    hidden_extraction_us: float
    rollback_us: float
    other_us: float
    cycle_total_us: float
    verify_token_count: int | None = None
    commit_us: float = 0.0
    cycle_wall_us: float | None = None
    yield_pause_us: float = 0.0
    prefix_len: int | None = None
    draft_source: str | None = None
    candidate_count: int | None = None

    def to_payload(self) -> dict[str, Any]:
        cycle_wall_us = (
            float(self.cycle_wall_us)
            if self.cycle_wall_us is not None
            else float(self.cycle_total_us)
        )
        payload: dict[str, Any] = {
            "cycle": int(self.cycle),
            "block_len": int(self.block_len),
            "commit_count": int(self.commit_count),
            "acceptance_len": int(self.acceptance_len),
            "draft_us": float(self.draft_us),
            "verify_us": float(self.verify_us),
            "acceptance_us": float(self.acceptance_us),
            "hidden_extraction_us": float(self.hidden_extraction_us),
            "commit_us": float(self.commit_us),
            "rollback_us": float(self.rollback_us),
            "other_us": float(self.other_us),
            "cycle_total_us": float(self.cycle_total_us),
            "cycle_engine_us": float(self.cycle_total_us),
            "cycle_wall_us": cycle_wall_us,
            "yield_pause_us": float(self.yield_pause_us),
        }
        if self.prefix_len is not None:
            payload["prefix_len"] = int(self.prefix_len)
        if self.verify_token_count is not None:
            payload["verify_token_count"] = int(self.verify_token_count)
        if self.draft_source is not None:
            payload["draft_source"] = str(self.draft_source)
        if self.candidate_count is not None:
            payload["candidate_count"] = int(self.candidate_count)
        return payload


@dataclass(frozen=True)
class MemoryWaterfallEvent:
    fields: dict[str, Any]


@dataclass(frozen=True)
class SummaryEvent:
    elapsed_us: float
    prompt_token_count: int
    generated_token_ids: tuple[int, ...]
    generation_tokens: int
    accepted_from_draft: int
    acceptance_ratio: float
    cycles_completed: int
    phase_timings_us: dict[str, float]
    block_tokens: int | None = None
    verify_len_cap: int | None = None
    quantize_kv_cache: bool | None = None
    target_fa_window: int | None = None
    draft_sink_size: int | None = None
    draft_window_size: int | None = None
    clear_cache_boundaries: bool | None = None
    tokens_per_cycle: float = 0.0
    acceptance_history: tuple[int, ...] = ()
    acceptance_first_20_avg: float = 0.0
    acceptance_last_20_avg: float = 0.0
    adaptive_block_reductions: int = 0
    adaptive_block_cycles: int = 0
    adaptive_block_min: int | None = None
    peak_memory_gb: float | None = None
    cycle_profile_us: tuple[CycleCompleteEvent, ...] | None = None
    cycle_profile_totals_us: dict[str, float] | None = None
    fallback_ar: bool = False
    fallback_reason: str | None = None
    copyspec_hits: int = 0
    copyspec_tokens: int = 0
    adaptive_metrics: dict[str, Any] | None = None
    hit_kind: str = "miss"

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "elapsed_us": float(self.elapsed_us),
            "prompt_token_count": int(self.prompt_token_count),
            "generated_token_ids": list(self.generated_token_ids),
            "generation_tokens": int(self.generation_tokens),
            "accepted_from_draft": int(self.accepted_from_draft),
            "acceptance_ratio": float(self.acceptance_ratio),
            "cycles_completed": int(self.cycles_completed),
            "phase_timings_us": dict(self.phase_timings_us),
            "verify_len_cap": self.verify_len_cap,
            "tokens_per_cycle": float(self.tokens_per_cycle),
            "acceptance_history": list(self.acceptance_history),
        }
        optional = {
            "block_tokens": self.block_tokens,
            "quantize_kv_cache": self.quantize_kv_cache,
            "target_fa_window": self.target_fa_window,
            "draft_sink_size": self.draft_sink_size,
            "draft_window_size": self.draft_window_size,
            "clear_cache_boundaries": self.clear_cache_boundaries,
            "acceptance_first_20_avg": self.acceptance_first_20_avg,
            "acceptance_last_20_avg": self.acceptance_last_20_avg,
            "adaptive_block_reductions": self.adaptive_block_reductions,
            "adaptive_block_cycles": self.adaptive_block_cycles,
            "adaptive_block_min": self.adaptive_block_min,
            "peak_memory_gb": self.peak_memory_gb,
            "cycle_profile_us": (
                [entry.to_payload() for entry in self.cycle_profile_us]
                if self.cycle_profile_us is not None
                else None
            ),
            "cycle_profile_totals_us": (
                dict(self.cycle_profile_totals_us)
                if self.cycle_profile_totals_us is not None
                else None
            ),
            "copyspec_hits": self.copyspec_hits or None,
            "copyspec_tokens": self.copyspec_tokens or None,
            "adaptive_metrics": (
                dict(self.adaptive_metrics) if self.adaptive_metrics else None
            ),
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        payload["hit_kind"] = self.hit_kind
        if self.fallback_ar:
            payload["fallback_ar"] = True
            payload["fallback_reason"] = self.fallback_reason
        return payload


EngineEvent: TypeAlias = (
    PrefillProgressEvent
    | PrefillCompleteEvent
    | TokenEvent
    | SnapshotPublishedEvent
    | CycleCompleteEvent
    | MemoryWaterfallEvent
    | SummaryEvent
)

ENGINE_EVENT_TYPES: tuple[type[Any], ...] = (
    PrefillProgressEvent,
    PrefillCompleteEvent,
    TokenEvent,
    SnapshotPublishedEvent,
    CycleCompleteEvent,
    MemoryWaterfallEvent,
    SummaryEvent,
)

def is_engine_event(event: object) -> bool:
    return isinstance(event, ENGINE_EVENT_TYPES)

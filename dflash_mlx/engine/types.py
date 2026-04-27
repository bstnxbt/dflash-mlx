# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)
from __future__ import annotations

from typing import Any, Optional, TypedDict


class PhaseTimingsUs(TypedDict):
    prefill: float
    draft: float
    draft_prefill: float
    draft_incremental: float
    verify: float
    replay: float
    commit: float


class StreamSummary(TypedDict, total=False):
    event: str
    elapsed_us: float
    prompt_token_count: int
    generated_token_ids: list[int]
    generation_tokens: int
    accepted_from_draft: int
    acceptance_ratio: float
    block_tokens: int
    cycles_completed: int
    phase_timings_us: PhaseTimingsUs
    verify_len_cap: Optional[int]
    quantize_kv_cache: bool
    draft_sink_size: int
    draft_window_size: int
    tokens_per_cycle: float
    acceptance_history: list[int]
    acceptance_first_20_avg: float
    acceptance_last_20_avg: float
    peak_memory_gb: Optional[float]
    fallback_ar: bool
    fallback_reason: Optional[str]
    cycle_profile_us: list[dict[str, Any]]
    cycle_profile_totals_us: dict[str, float]


class TokenEvent(TypedDict, total=False):
    event: str
    token_id: int
    generated_tokens: int
    acceptance_ratio: float
    cycles_completed: int
    fallback_ar: bool
    fallback_reason: Optional[str]


class PrefillSnapshotReady(TypedDict, total=False):
    event: str
    token_ids: list[int]
    target_cache: Any
    target_hidden: Any
    last_logits: Any
    from_snapshot: bool
    snap_prefix_len: int
    snapshot_boundary: int

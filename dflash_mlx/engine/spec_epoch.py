# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from collections import deque
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import mlx.core as mx

from dflash_mlx.cache.codecs import hydrate_target_cache
from dflash_mlx.cache.snapshot_service import SnapshotPublication, SnapshotService
from dflash_mlx.cache.snapshot import (
    DFlashPrefixSnapshot,
    validate_prefix_snapshot as _validate_prefix_snapshot,
)
from dflash_mlx.draft_backend import DraftBackend
from dflash_mlx.engine.acceptance import match_acceptance_length as _match_acceptance_length
from dflash_mlx.engine.copyspec import CopySpecIndex
from dflash_mlx.engine.ddtree import (
    build_flat_ddtree,
    build_flat_tree_inputs,
    branch_positions as ddtree_branch_positions,
    candidate_token_ids as ddtree_candidate_token_ids,
    clone_cache_for_batch as ddtree_clone_cache_for_batch,
    copy_selected_cache as ddtree_copy_selected_cache,
    follow_verified_tree,
    restore_cache as ddtree_restore_cache,
    select_tree_slots as ddtree_select_tree_slots,
    snapshot_cache as ddtree_snapshot_cache,
    verify_candidates_batch as ddtree_verify_candidates_batch,
)
from dflash_mlx.engine.fallback import stream_baseline_generate
from dflash_mlx.engine.prefill import compute_snapshot_boundary
from dflash_mlx.engine.sampling import (
    build_suppress_token_mask,
    eval_logits_and_captured,
    greedy_tokens_with_mask,
    ns_to_us,
    prepare_prompt_tokens,
)
from dflash_mlx.engine.target_features import TargetFeatureStore
from dflash_mlx.engine.config import (
    _profile_dflash_cycles_enabled,
    resolve_draft_window,
    resolve_speculative_cycle_config,
    verify_token_count_for_block,
)
from dflash_mlx.engine.events import (
    CycleCompleteEvent,
    EngineEvent,
    MemoryWaterfallEvent,
    PrefillCompleteEvent,
    PrefillProgressEvent,
    SnapshotPublishedEvent,
    SummaryEvent,
    TokenEvent,
)
from dflash_mlx.model import DFlashDraftModel
from dflash_mlx.engine.memory_waterfall import (
    collect_memory_waterfall as _collect_memory_waterfall,
    memory_waterfall_enabled as _memory_waterfall_enabled,
    should_sample_cycle as _should_sample_memory_cycle,
)

_DECODE_CLEAR_CACHE_INTERVAL_TOKENS = 1024
_DDTREE_TOP_WIDTH = 2
_DDTREE_MAX_BRANCH_POSITIONS = 2
_ADAPTIVE_REDUCED_BURST_CYCLES = 64
_ADAPTIVE_LONG_CONTEXT_REDUCED_BURST_CYCLES = 64
_ADAPTIVE_PROBE_CYCLES = 8
_ADAPTIVE_EARLY_PROBE_CYCLES = 4
_ADAPTIVE_EARLY_PROBE_MIN_CYCLES = 16
_ADAPTIVE_EARLY_PROBE_WINDOW = 8
_ADAPTIVE_EARLY_PROBE_FULL_ACCEPTANCE = 0.75
_ADAPTIVE_DROP_ACCEPTANCE_THRESHOLD = 0.75
_ADAPTIVE_RESUME_ACCEPTANCE_THRESHOLD = 0.80
_ADAPTIVE_TPC_THRESHOLD = 3.5
_ADAPTIVE_RESUME_SPEED_RATIO = 1.0


@dataclass(frozen=True)
class _SessionRequest:
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    block_tokens: Optional[int] = None
    stop_token_ids: tuple[int, ...] = ()
    suppress_token_ids: Optional[list[int]] = None
    prefix_snapshot: Optional[DFlashPrefixSnapshot] = None
    snapshot_service: Optional[SnapshotService] = None
    stable_prefix_len: Optional[int] = None
    prefix_cache_active: bool = False
    publish_generation_snapshot: bool = True
    prompt_array: mx.array = field(init=False, repr=False)
    prompt_len: int = field(init=False)
    stop_token_array: Optional[mx.array] = field(init=False, repr=False)

    @classmethod
    def from_tokens(
        cls,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        block_tokens: Optional[int],
        stop_token_ids: Optional[list[int]],
        suppress_token_ids: Optional[list[int]],
        prefix_snapshot: Optional[DFlashPrefixSnapshot],
        snapshot_service: Optional[SnapshotService],
        stable_prefix_len: Optional[int],
        prefix_cache_active: bool,
        publish_generation_snapshot: bool = True,
    ) -> "_SessionRequest":
        return cls(
            prompt_tokens=tuple(int(token) for token in prompt_tokens),
            max_new_tokens=int(max_new_tokens),
            block_tokens=block_tokens,
            stop_token_ids=tuple(int(token) for token in (stop_token_ids or ())),
            suppress_token_ids=(
                [int(token) for token in suppress_token_ids]
                if suppress_token_ids is not None
                else None
            ),
            prefix_snapshot=prefix_snapshot,
            snapshot_service=snapshot_service,
            stable_prefix_len=stable_prefix_len,
            prefix_cache_active=bool(prefix_cache_active),
            publish_generation_snapshot=bool(publish_generation_snapshot),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_len", len(self.prompt_tokens))
        object.__setattr__(
            self,
            "prompt_array",
            mx.array(self.prompt_tokens, dtype=mx.uint32)[None],
        )
        stop_array = (
            mx.array(self.stop_token_ids, dtype=mx.uint32)
            if self.stop_token_ids
            else None
        )
        object.__setattr__(self, "stop_token_array", stop_array)

    def should_collect_generation_snapshot_hidden(
        self,
        supports_prefix_snapshot: bool,
    ) -> bool:
        if not supports_prefix_snapshot or not self.publish_generation_snapshot:
            return False
        if self.snapshot_service is None or not self.snapshot_service.active:
            return False
        return True


@dataclass(frozen=True)
class _PrefillResult:
    feature_store: TargetFeatureStore
    start_ns: int
    prefill_ns: int
    suppress_token_mask: mx.array | None
    supports_prefix_snapshot: bool


@dataclass(frozen=True)
class _DecodeResult:
    effective_block_tokens: int
    verify_len_cap: int
    adaptive_block_reductions: int
    adaptive_block_cycles: int
    adaptive_block_min: int | None
    draft_ns_total: int
    draft_prefill_ns: int
    draft_incremental_ns: int
    verify_ns_total: int
    replay_ns_total: int
    commit_ns_total: int
    cycle_profiles: tuple[CycleCompleteEvent, ...]
    profile_totals_ns: dict[str, int]
    copyspec_hits: int
    copyspec_tokens: int
    adaptive_totals: dict[str, Any] = field(default_factory=dict)


@dataclass
class _AdaptiveBlockPolicy:
    full_block_tokens: int
    min_block_tokens: int = 4
    window_size: int = 4
    reduced_burst_cycles: int = _ADAPTIVE_REDUCED_BURST_CYCLES
    long_context_prompt_tokens: int = 32768
    long_context_burst_cycles: int = _ADAPTIVE_LONG_CONTEXT_REDUCED_BURST_CYCLES
    probe_cycles: int = _ADAPTIVE_PROBE_CYCLES
    early_probe_cycles: int = _ADAPTIVE_EARLY_PROBE_CYCLES
    early_probe_min_cycles: int = _ADAPTIVE_EARLY_PROBE_MIN_CYCLES
    early_probe_window: int = _ADAPTIVE_EARLY_PROBE_WINDOW
    early_probe_full_acceptance: float = _ADAPTIVE_EARLY_PROBE_FULL_ACCEPTANCE
    drop_acceptance_threshold: float = _ADAPTIVE_DROP_ACCEPTANCE_THRESHOLD
    resume_acceptance_threshold: float = _ADAPTIVE_RESUME_ACCEPTANCE_THRESHOLD
    tpc_threshold: float = _ADAPTIVE_TPC_THRESHOLD
    resume_speed_ratio: float = _ADAPTIVE_RESUME_SPEED_RATIO
    mode: Literal["full", "reduced", "probe"] = "full"
    reduced_cycles_since_probe: int = 0
    probe_cycles_remaining: int = 0
    reductions: int = 0
    reduced_cycles: int = 0
    min_seen: int | None = None
    full_commits: deque[int] = field(default_factory=lambda: deque(maxlen=12))
    probe_commits: list[int] = field(default_factory=list)
    probe_cost_ns: list[int] = field(default_factory=list)
    probe_is_early: bool = False
    early_probe_lockout: bool = False
    reduced_commits: deque[int] = field(default_factory=lambda: deque(maxlen=32))
    reduced_cost_ns: deque[int] = field(default_factory=lambda: deque(maxlen=32))
    reduced_full_accepts: deque[int] = field(
        default_factory=lambda: deque(maxlen=max(1, _ADAPTIVE_EARLY_PROBE_WINDOW))
    )
    stats_mode_cycles: dict[str, int] = field(default_factory=dict)
    stats_mode_commit_tokens: dict[str, int] = field(default_factory=dict)
    stats_mode_cost_ns: dict[str, int] = field(default_factory=dict)
    stats_block_cycles: dict[int, int] = field(default_factory=dict)
    stats_block_accept_tokens: dict[int, int] = field(default_factory=dict)
    stats_block_commit_tokens: dict[int, int] = field(default_factory=dict)
    stats_block_cost_ns: dict[int, int] = field(default_factory=dict)
    stats_block_full_accepts: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_runtime(
        cls,
        *,
        runtime_config: Any,
        effective_block_tokens: int,
        verify_len_cap: int,
        prompt_len: int = 0,
    ) -> "_AdaptiveBlockPolicy | None":
        if str(getattr(runtime_config, "verify_mode", "dflash")) != "adaptive":
            return None
        full_block_tokens = int(effective_block_tokens)
        if full_block_tokens <= 4:
            return None
        if int(verify_len_cap) < full_block_tokens:
            return None
        policy = cls(full_block_tokens=full_block_tokens)
        if int(prompt_len) >= policy.long_context_prompt_tokens:
            policy.mode = "reduced"
            policy.reductions = 1
            policy.reduced_burst_cycles = policy.long_context_burst_cycles
        return policy

    def block_limit(self) -> int:
        if self.mode == "reduced":
            return int(min(self.full_block_tokens, self.min_block_tokens))
        return int(self.full_block_tokens)

    @staticmethod
    def _window_stats(commits: Sequence[int]) -> tuple[float, float]:
        cycles = len(commits)
        if cycles <= 0:
            return 0.0, 0.0
        total_commits = sum(int(x) for x in commits)
        if total_commits <= 0:
            return 0.0, 0.0
        accepted_draft = max(0, total_commits - cycles)
        acceptance = accepted_draft / total_commits
        tokens_per_cycle = total_commits / cycles
        return float(acceptance), float(tokens_per_cycle)

    @staticmethod
    def _draft_capacity(block_len: int) -> int:
        return max(0, int(block_len) - 1)

    def _enter_reduced(self, *, early_probe_lockout: bool = False) -> None:
        self.mode = "reduced"
        self.reduced_cycles_since_probe = 0
        self.reduced_burst_cycles = _ADAPTIVE_REDUCED_BURST_CYCLES
        self.probe_cycles_remaining = 0
        self.probe_is_early = False
        self.early_probe_lockout = bool(early_probe_lockout)
        self.probe_commits.clear()
        self.probe_cost_ns.clear()
        self.reduced_commits.clear()
        self.reduced_cost_ns.clear()
        self.reduced_full_accepts.clear()
        self.full_commits.clear()
        self.reductions += 1

    def _enter_probe(self, *, early: bool = False) -> None:
        self.mode = "probe"
        self.probe_is_early = bool(early)
        if not early:
            self.early_probe_lockout = False
        self.probe_cycles_remaining = max(
            1,
            int(self.early_probe_cycles if early else self.probe_cycles),
        )
        self.probe_commits.clear()
        self.probe_cost_ns.clear()
        self.reduced_burst_cycles = _ADAPTIVE_REDUCED_BURST_CYCLES
        self.reduced_full_accepts.clear()

    def _should_probe_reduced_early(self) -> bool:
        if self.early_probe_min_cycles <= 0:
            return False
        if self.early_probe_lockout:
            return False
        if self.reduced_cycles_since_probe < self.early_probe_min_cycles:
            return False
        window = max(1, int(self.early_probe_window))
        if len(self.reduced_full_accepts) < window:
            return False
        full_accept_rate = sum(self.reduced_full_accepts) / float(
            len(self.reduced_full_accepts)
        )
        return full_accept_rate >= self.early_probe_full_acceptance

    @staticmethod
    def _tokens_per_second(commits: Sequence[int], costs_ns: Sequence[int]) -> float:
        cost_total = sum(int(value) for value in costs_ns)
        if cost_total <= 0:
            return 0.0
        return sum(int(value) for value in commits) / (cost_total / 1_000_000_000.0)

    def _record_stats(
        self,
        *,
        mode: str,
        block_len: int,
        acceptance_len: int,
        commit_count: int,
        cycle_cost_ns: int | None,
    ) -> None:
        self.stats_mode_cycles[mode] = self.stats_mode_cycles.get(mode, 0) + 1
        self.stats_mode_commit_tokens[mode] = (
            self.stats_mode_commit_tokens.get(mode, 0) + int(commit_count)
        )
        self.stats_block_cycles[block_len] = self.stats_block_cycles.get(block_len, 0) + 1
        self.stats_block_accept_tokens[block_len] = (
            self.stats_block_accept_tokens.get(block_len, 0) + int(acceptance_len)
        )
        self.stats_block_commit_tokens[block_len] = (
            self.stats_block_commit_tokens.get(block_len, 0) + int(commit_count)
        )
        if cycle_cost_ns is not None and cycle_cost_ns > 0:
            self.stats_mode_cost_ns[mode] = (
                self.stats_mode_cost_ns.get(mode, 0) + int(cycle_cost_ns)
            )
            self.stats_block_cost_ns[block_len] = (
                self.stats_block_cost_ns.get(block_len, 0) + int(cycle_cost_ns)
            )
        draft_capacity = self._draft_capacity(block_len)
        if draft_capacity > 0 and acceptance_len >= draft_capacity:
            self.stats_block_full_accepts[block_len] = (
                self.stats_block_full_accepts.get(block_len, 0) + 1
            )

    def metrics(self) -> dict[str, Any]:
        mode_tpc = {
            mode: self.stats_mode_commit_tokens.get(mode, 0) / float(cycles)
            for mode, cycles in self.stats_mode_cycles.items()
            if cycles
        }
        mode_speed = {
            mode: (
                self.stats_mode_commit_tokens.get(mode, 0)
                / (self.stats_mode_cost_ns.get(mode, 0) / 1_000_000_000.0)
                if self.stats_mode_cost_ns.get(mode, 0) > 0
                else 0.0
            )
            for mode in self.stats_mode_cycles
        }
        block_tpc = {
            str(block): self.stats_block_commit_tokens.get(block, 0) / float(cycles)
            for block, cycles in self.stats_block_cycles.items()
            if cycles
        }
        block_speed = {
            str(block): (
                self.stats_block_commit_tokens.get(block, 0)
                / (self.stats_block_cost_ns.get(block, 0) / 1_000_000_000.0)
                if self.stats_block_cost_ns.get(block, 0) > 0
                else 0.0
            )
            for block in self.stats_block_cycles
        }
        block_accept = {
            str(block): (
                self.stats_block_accept_tokens.get(block, 0)
                / float(self._draft_capacity(int(block)) * cycles)
                if self._draft_capacity(int(block)) > 0 and cycles
                else 0.0
            )
            for block, cycles in self.stats_block_cycles.items()
        }
        block_full_accept = {
            str(block): self.stats_block_full_accepts.get(block, 0) / float(cycles)
            for block, cycles in self.stats_block_cycles.items()
            if cycles
        }
        return {
            "reduced_burst_cycles": int(self.reduced_burst_cycles),
            "probe_cycles": int(self.probe_cycles),
            "early_probe_cycles": int(self.early_probe_cycles),
            "early_probe_min_cycles": int(self.early_probe_min_cycles),
            "early_probe_window": int(self.early_probe_window),
            "early_probe_full_acceptance": float(self.early_probe_full_acceptance),
            "resume_speed_ratio": float(self.resume_speed_ratio),
            "cycles_by_mode": dict(self.stats_mode_cycles),
            "tokens_per_cycle_by_mode": mode_tpc,
            "tokens_per_second_by_mode": mode_speed,
            "cycles_by_block": {str(k): int(v) for k, v in self.stats_block_cycles.items()},
            "tokens_per_cycle_by_block": block_tpc,
            "tokens_per_second_by_block": block_speed,
            "acceptance_by_block": block_accept,
            "full_accept_rate_by_block": block_full_accept,
            "full_accepts_by_block": {
                str(k): int(v) for k, v in self.stats_block_full_accepts.items()
            },
        }

    def record(
        self,
        *,
        block_len: int,
        acceptance_len: int,
        cycle_cost_ns: int | None = None,
    ) -> None:
        block_len = int(block_len)
        acceptance_len = int(acceptance_len)
        commit_count = 1 + acceptance_len
        self._record_stats(
            mode=str(self.mode),
            block_len=block_len,
            acceptance_len=acceptance_len,
            commit_count=commit_count,
            cycle_cost_ns=cycle_cost_ns,
        )
        if self.mode == "reduced" and block_len < self.full_block_tokens:
            self.reduced_cycles += 1
            self.reduced_cycles_since_probe += 1
            self.reduced_commits.append(commit_count)
            if cycle_cost_ns is not None and cycle_cost_ns > 0:
                self.reduced_cost_ns.append(int(cycle_cost_ns))
            self.min_seen = (
                block_len
                if self.min_seen is None
                else min(int(self.min_seen), block_len)
            )
            draft_capacity = self._draft_capacity(block_len)
            self.reduced_full_accepts.append(
                1 if draft_capacity > 0 and acceptance_len >= draft_capacity else 0
            )
            if self._should_probe_reduced_early():
                self._enter_probe(early=True)
            elif self.reduced_cycles_since_probe >= self.reduced_burst_cycles:
                self._enter_probe()
            return

        if block_len < self.full_block_tokens:
            return

        self.full_commits.append(commit_count)
        while len(self.full_commits) > self.window_size:
            self.full_commits.popleft()
        if self.mode == "probe":
            self.probe_commits.append(commit_count)
            if cycle_cost_ns is not None and cycle_cost_ns > 0:
                self.probe_cost_ns.append(int(cycle_cost_ns))
            self.probe_cycles_remaining = max(0, self.probe_cycles_remaining - 1)
            if self.probe_cycles_remaining > 0:
                return
            probe_acceptance, probe_tpc = self._window_stats(self.probe_commits)
            reduced_speed = self._tokens_per_second(
                tuple(self.reduced_commits),
                tuple(self.reduced_cost_ns),
            )
            probe_speed = self._tokens_per_second(
                tuple(self.probe_commits),
                tuple(self.probe_cost_ns),
            )
            speed_ok = (
                self.resume_speed_ratio <= 0.0
                or reduced_speed <= 0.0
                or probe_speed >= reduced_speed * self.resume_speed_ratio
            )
            if (
                probe_acceptance >= self.resume_acceptance_threshold
                and probe_tpc >= self.tpc_threshold
                and speed_ok
            ):
                self.mode = "full"
                self.reduced_cycles_since_probe = 0
                self.reduced_burst_cycles = _ADAPTIVE_REDUCED_BURST_CYCLES
                self.probe_commits.clear()
                self.probe_cost_ns.clear()
                self.reduced_full_accepts.clear()
                self.full_commits.clear()
                self.probe_cycles_remaining = 0
            else:
                self._enter_reduced(early_probe_lockout=True)
            return

        if len(self.full_commits) < self.window_size:
            return

        rolling_acceptance, rolling_tpc = self._window_stats(self.full_commits)
        if (
            rolling_acceptance < self.drop_acceptance_threshold
            or rolling_tpc < self.tpc_threshold
        ):
            self._enter_reduced()


@dataclass
class SpeculativeSession:
    target_model: Any
    draft_model: DFlashDraftModel
    target_ops: Any
    target_cache: Any
    draft_cache: list[Any]
    draft_backend: DraftBackend
    runtime_config: Any
    quantize_kv_cache: bool
    snap_prefix_len: int
    supports_prefix_snapshot: bool
    allow_full_context_draft_layers: bool
    draft_sink_size: int
    draft_window_size: int
    target_layer_id_list: list[int]
    capture_layer_ids: set[int]
    profile_cycles: bool
    memory_waterfall: bool
    clear_cache_boundaries: bool
    target_fa_window: int
    copyspec_index: CopySpecIndex

    @classmethod
    def open(
        cls,
        *,
        target_model: Any,
        draft_model: DFlashDraftModel,
        draft_backend: DraftBackend,
        target_ops: Any,
        supports_prefix_snapshot: bool,
        allow_full_context_draft_layers: bool,
        prompt_tokens: Sequence[int],
        max_new_tokens: int,
        prefix_snapshot: Optional[DFlashPrefixSnapshot],
        quantize_kv_cache: bool,
        target_fa_window: int,
        runtime_context: Any,
    ) -> "SpeculativeSession":
        runtime_config = runtime_context.runtime
        draft_sink_size, draft_window_size = resolve_draft_window(
            runtime_config,
            draft_model,
            context_len=len(prompt_tokens) + max(0, int(max_new_tokens)),
            allow_full_attention_context=allow_full_context_draft_layers,
        )
        snap_prefix_len = _validate_prefix_snapshot(prefix_snapshot, prompt_tokens)
        if not supports_prefix_snapshot:
            snap_prefix_len = 0
        if snap_prefix_len > 0 and (quantize_kv_cache or target_fa_window > 0):
            snap_prefix_len = 0
        if snap_prefix_len > 0:
            template_cache = target_ops.make_cache(
                target_model,
                enable_speculative_linear_cache=True,
                quantize_kv_cache=quantize_kv_cache,
                target_fa_window=target_fa_window,
            )
            try:
                assert prefix_snapshot is not None
                target_cache = hydrate_target_cache(prefix_snapshot, template_cache)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"prefix snapshot hydrate failed for {snap_prefix_len} tokens"
                ) from exc
            finally:
                del template_cache
        else:
            target_cache = target_ops.make_cache(
                target_model,
                enable_speculative_linear_cache=True,
                quantize_kv_cache=quantize_kv_cache,
                target_fa_window=target_fa_window,
            )
        draft_cache = draft_backend.make_cache(
            draft_model=draft_model,
            sink_size=draft_sink_size,
            window_size=draft_window_size,
            allow_full_context_layers=allow_full_context_draft_layers,
        )
        diagnostics = runtime_context.diagnostics
        profile_cycles = _profile_dflash_cycles_enabled(diagnostics)
        memory_waterfall = _memory_waterfall_enabled(diagnostics)
        return cls(
            target_model=target_model,
            draft_model=draft_model,
            target_ops=target_ops,
            target_cache=target_cache,
            draft_cache=draft_cache,
            draft_backend=draft_backend,
            runtime_config=runtime_config,
            quantize_kv_cache=bool(quantize_kv_cache),
            snap_prefix_len=snap_prefix_len,
            supports_prefix_snapshot=supports_prefix_snapshot,
            allow_full_context_draft_layers=allow_full_context_draft_layers,
            draft_sink_size=draft_sink_size,
            draft_window_size=draft_window_size,
            target_layer_id_list=list(draft_model.target_layer_ids),
            capture_layer_ids={
                int(layer_id) + 1 for layer_id in draft_model.target_layer_ids
            },
            profile_cycles=profile_cycles,
            memory_waterfall=memory_waterfall,
            clear_cache_boundaries=bool(runtime_config.clear_cache_boundaries),
            target_fa_window=target_fa_window,
            copyspec_index=CopySpecIndex(prompt_tokens),
        )

    def clear_cache_boundary(self) -> None:
        if not self.clear_cache_boundaries:
            return
        synchronize = getattr(mx, "synchronize", None)
        if callable(synchronize):
            synchronize()
        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()

    def memory_waterfall_event(
        self,
        phase: str,
        *,
        target_hidden_value: Any = None,
        gen_hidden_chunks_value: Any = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[MemoryWaterfallEvent]:
        if not self.memory_waterfall:
            return None
        return MemoryWaterfallEvent(
            fields=_collect_memory_waterfall(
                phase=phase,
                target_cache=self.target_cache,
                draft_cache=self.draft_cache,
                target_hidden=target_hidden_value,
                gen_hidden_chunks=gen_hidden_chunks_value,
                extra=extra,
            ),
        )

    def _run_prefill_events(
        self,
        *,
        request: _SessionRequest,
        state: "_RequestState",
        yield_pause: "_YieldPauseTracker",
    ) -> Generator[EngineEvent, None, _PrefillResult]:
        target_cache = self.target_cache
        target_ops = self.target_ops
        target_model = self.target_model
        draft_model = self.draft_model
        runtime_config = self.runtime_config
        snap_prefix_len = self.snap_prefix_len
        supports_prefix_snapshot = self.supports_prefix_snapshot
        allow_full_context_draft_layers = self.allow_full_context_draft_layers
        target_layer_id_list = self.target_layer_id_list
        capture_layer_ids = self.capture_layer_ids
        profile_cycles = self.profile_cycles
        prompt_tokens = request.prompt_tokens
        prompt_array = request.prompt_array
        prompt_len = request.prompt_len
        suppress_token_ids = request.suppress_token_ids
        prefix_snapshot = request.prefix_snapshot
        snapshot_service = request.snapshot_service
        stable_prefix_len = request.stable_prefix_len
        prefix_cache_active = request.prefix_cache_active

        feature_store = TargetFeatureStore(
            prompt_len=prompt_len,
            project_context=draft_model.project_target_hidden,
        )
        if supports_prefix_snapshot and snapshot_service is None:
            if prefix_cache_active:
                raise ValueError("snapshot_service is required when prefix cache is active")
        publish_prefix_snapshots = bool(
            supports_prefix_snapshot and snapshot_service is not None
        )

        start_ns = time.perf_counter_ns()
        evt = self.memory_waterfall_event("after_target_cache_create")
        if evt is not None:
            _pre_yield = yield_pause.mark()
            yield evt
            yield_pause.done(_pre_yield)
        prefill_start_ns = time.perf_counter_ns()
        prefill_step_size = int(runtime_config.prefill_step_size)

        _phase_rebuild_ns = 0
        _phase_cold_ns = 0
        _phase_seam_ns = 0
        _phase_tail_ns = 0

        if snap_prefix_len > 0:
            assert prefix_snapshot is not None
            if profile_cycles:
                _t = time.perf_counter_ns()
            feature_store.hydrate_from_snapshot(
                prefix_snapshot,
                snap_prefix_len=snap_prefix_len,
            )
            if profile_cycles:
                _phase_rebuild_ns += time.perf_counter_ns() - _t
            evt = self.memory_waterfall_event(
                "after_prefix_hydrate",
                target_hidden_value=feature_store.current_hidden,
                extra={"snap_prefix_len": int(snap_prefix_len)},
            )
            if evt is not None:
                _pre_yield = yield_pause.mark()
                yield evt
                yield_pause.done(_pre_yield)

        snapshot_boundary = (
            compute_snapshot_boundary(prompt_len, stable_prefix_len)
            if supports_prefix_snapshot
            else 0
        )
        prefill_context_len = max(0, snapshot_boundary - 1)
        chunked_start = min(snap_prefix_len, prefill_context_len)
        for chunk_start in range(chunked_start, prefill_context_len, prefill_step_size):
            if profile_cycles:
                _t = time.perf_counter_ns()
            chunk_end = min(chunk_start + prefill_step_size, prefill_context_len)
            chunk_ids = prompt_array[:, chunk_start:chunk_end]
            state.prefill_logits, prefill_hidden_states = (
                target_ops.forward_with_hidden_capture(
                    target_model,
                    input_ids=chunk_ids,
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                    logits_last_only=True,
                )
            )
            eval_logits_and_captured(state.prefill_logits, prefill_hidden_states)
            feat = target_ops.extract_context_feature(
                prefill_hidden_states,
                target_layer_id_list,
            )
            feature_store.write_prompt_slice(
                start=chunk_start,
                end=chunk_end,
                features=feat,
            )
            del feat, prefill_hidden_states
            if (
                publish_prefix_snapshots
                and snapshot_service is not None
                and snapshot_service.should_publish_frontier(chunk_end)
                and chunk_end < snapshot_boundary
            ):
                _snapshot_build = yield_pause.mark()
                snapshot_event = _publish_snapshot_event(
                    token_ids=list(prompt_tokens[:chunk_end]),
                    target_cache=target_cache,
                    target_hidden=feature_store.prefix_view(chunk_end),
                    last_logits=(
                        state.prefill_logits[:, -1, :]
                        if state.prefill_logits is not None
                        else None
                    ),
                    snapshot_service=snapshot_service,
                    kind="prefill",
                    require_logits=True,
                    snapshot_boundary=chunk_end,
                    allow_full_context_draft_layers=allow_full_context_draft_layers,
                    from_snapshot=bool(snap_prefix_len > 0),
                    snap_prefix_len=min(int(snap_prefix_len), int(chunk_end)),
                    l2_only=True,
                )
                yield_pause.done(_snapshot_build)
                if snapshot_event is not None:
                    _pre_yield = yield_pause.mark()
                    yield snapshot_event
                    yield_pause.done(_pre_yield)
            if profile_cycles:
                _phase_cold_ns += time.perf_counter_ns() - _t
            self.clear_cache_boundary()
            _pre_yield = yield_pause.mark()
            yield PrefillProgressEvent(
                tokens_processed=int(chunk_end),
                tokens_total=int(prompt_len),
            )
            yield_pause.done(_pre_yield)
            evt = self.memory_waterfall_event(
                "after_prefill_chunk",
                target_hidden_value=feature_store.current_hidden,
                extra={
                    "chunk_start": int(chunk_start),
                    "chunk_end": int(chunk_end),
                },
            )
            if evt is not None:
                _pre_yield = yield_pause.mark()
                yield evt
                yield_pause.done(_pre_yield)

        if (
            snap_prefix_len > 0
            and snap_prefix_len == snapshot_boundary
            and prefix_snapshot is not None
            and prefix_snapshot.last_logits is not None
        ):
            if profile_cycles:
                _t = time.perf_counter_ns()
            last_logits_2d = prefix_snapshot.last_logits
            state.prefill_logits = mx.expand_dims(last_logits_2d, axis=1)
            mx.eval(state.prefill_logits)
            if profile_cycles:
                _phase_seam_ns += time.perf_counter_ns() - _t
        elif snapshot_boundary > 0 and snap_prefix_len < snapshot_boundary:
            if profile_cycles:
                _t = time.perf_counter_ns()
            final_prompt_start = snapshot_boundary - 1
            state.prefill_logits, prefill_hidden_states = (
                target_ops.forward_with_hidden_capture(
                    target_model,
                    input_ids=prompt_array[:, final_prompt_start:snapshot_boundary],
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                    logits_last_only=True,
                )
            )
            eval_logits_and_captured(state.prefill_logits, prefill_hidden_states)
            feat = target_ops.extract_context_feature(
                prefill_hidden_states,
                target_layer_id_list,
            )
            feature_store.write_prompt_slice(
                start=final_prompt_start,
                end=snapshot_boundary,
                features=feat,
            )
            del feat, prefill_hidden_states
            if profile_cycles:
                _phase_seam_ns += time.perf_counter_ns() - _t
        _pre_yield = yield_pause.mark()
        yield PrefillProgressEvent(
            tokens_processed=int(snapshot_boundary),
            tokens_total=int(prompt_len),
        )
        yield_pause.done(_pre_yield)
        self.clear_cache_boundary()

        exact_snapshot_restore = bool(
            snap_prefix_len > 0 and snap_prefix_len == snapshot_boundary
        )
        if (
            exact_snapshot_restore
            and snapshot_service is not None
            and prefix_snapshot is not None
            and prefix_snapshot.last_logits is None
        ):
            raise ValueError("prefill snapshot requires last_logits")
        if publish_prefix_snapshots and not exact_snapshot_restore:
            _snapshot_build = yield_pause.mark()
            snapshot_event = _publish_snapshot_event(
                token_ids=list(prompt_tokens[:snapshot_boundary]),
                target_cache=target_cache,
                target_hidden=feature_store.prefix_view(snapshot_boundary),
                last_logits=(
                    state.prefill_logits[:, -1, :]
                    if state.prefill_logits is not None
                    else None
                ),
                snapshot_service=snapshot_service,
                kind="prefill",
                require_logits=True,
                snapshot_boundary=snapshot_boundary,
                allow_full_context_draft_layers=allow_full_context_draft_layers,
                from_snapshot=bool(snap_prefix_len > 0),
                snap_prefix_len=snap_prefix_len,
            )
            yield_pause.done(_snapshot_build)
            if snapshot_event is not None:
                _pre_yield = yield_pause.mark()
                yield snapshot_event
                yield_pause.done(_pre_yield)
            evt = self.memory_waterfall_event(
                "after_prefill_snapshot_ready",
                target_hidden_value=feature_store.current_hidden,
                extra={"snapshot_boundary": int(snapshot_boundary)},
            )
            if evt is not None:
                _pre_yield = yield_pause.mark()
                yield evt
                yield_pause.done(_pre_yield)

        if snapshot_boundary < prompt_len:
            if profile_cycles:
                _t = time.perf_counter_ns()
            tail_logits, tail_hidden_states = target_ops.forward_with_hidden_capture(
                target_model,
                input_ids=prompt_array[:, snapshot_boundary:prompt_len],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
                logits_last_only=True,
            )
            eval_logits_and_captured(tail_logits, tail_hidden_states)
            tail_feat = target_ops.extract_context_feature(
                tail_hidden_states,
                target_layer_id_list,
            )
            feature_store.write_prompt_slice(
                start=snapshot_boundary,
                end=prompt_len,
                features=tail_feat,
            )
            state.prefill_logits = tail_logits
            del tail_feat, tail_hidden_states
            if profile_cycles:
                _phase_tail_ns += time.perf_counter_ns() - _t
            self.clear_cache_boundary()
            _pre_yield = yield_pause.mark()
            yield PrefillProgressEvent(
                tokens_processed=int(prompt_len),
                tokens_total=int(prompt_len),
            )
            yield_pause.done(_pre_yield)
        evt = self.memory_waterfall_event(
            "after_tail_prefill",
            target_hidden_value=feature_store.current_hidden,
            extra={"prompt_len": int(prompt_len)},
        )
        if evt is not None:
            _pre_yield = yield_pause.mark()
            yield evt
            yield_pause.done(_pre_yield)

        prefill_ns = time.perf_counter_ns() - prefill_start_ns

        collect_generation_snapshot_hidden = (
            request.should_collect_generation_snapshot_hidden(supports_prefix_snapshot)
        )
        feature_store.freeze_prefill_for_snapshot(
            enabled=collect_generation_snapshot_hidden
        )

        if state.prefill_logits is None:
            raise RuntimeError("prefill logits unavailable after prefix snapshot restore")
        suppress_token_mask = build_suppress_token_mask(
            int(state.prefill_logits.shape[-1]),
            suppress_token_ids,
        )
        state.staged_first = greedy_tokens_with_mask(
            state.prefill_logits[:, -1, :],
            suppress_token_mask,
        ).reshape(-1)
        prefill_tokens_restored = max(0, min(int(snap_prefix_len), int(prompt_len)))
        prefill_tokens_computed = max(0, int(prompt_len) - prefill_tokens_restored)

        prefill_event = PrefillCompleteEvent(
            prefill_us=prefill_ns / 1_000.0,
            prompt_token_count=int(prompt_len),
            snap_prefix_len=int(snap_prefix_len),
            snapshot_boundary=int(snapshot_boundary),
            logical_ctx_tokens=int(prompt_len),
            physical_prefill_tokens=int(prefill_tokens_computed),
            prefill_tokens_restored=int(prefill_tokens_restored),
            prefill_tokens_computed=int(prefill_tokens_computed),
            phase_rebuild_us=(
                _phase_rebuild_ns / 1_000.0 if profile_cycles else None
            ),
            phase_cold_us=_phase_cold_ns / 1_000.0 if profile_cycles else None,
            phase_seam_us=_phase_seam_ns / 1_000.0 if profile_cycles else None,
            phase_tail_us=_phase_tail_ns / 1_000.0 if profile_cycles else None,
        )
        _pre_yield = yield_pause.mark()
        yield prefill_event
        yield_pause.done(_pre_yield)

        return _PrefillResult(
            feature_store=feature_store,
            start_ns=start_ns,
            prefill_ns=prefill_ns,
            suppress_token_mask=suppress_token_mask,
            supports_prefix_snapshot=supports_prefix_snapshot,
        )

    def _run_generation_snapshot_events(
        self,
        *,
        request: _SessionRequest,
        state: "_RequestState",
        feature_store: TargetFeatureStore,
        supports_prefix_snapshot: bool,
        yield_pause: "_YieldPauseTracker",
    ) -> Iterator[EngineEvent]:
        if (
            not state.generated_token_ids
            or not request.should_collect_generation_snapshot_hidden(
                supports_prefix_snapshot
            )
        ):
            return

        end_target_hidden = feature_store.generation_snapshot_hidden()
        if end_target_hidden is None:
            return

        if state.last_cycle_logits is not None:
            mx.eval(state.last_cycle_logits)
        self.clear_cache_boundary()
        end_total_len = request.prompt_len + len(state.generated_token_ids)
        _snapshot_build = yield_pause.mark()
        snapshot_event = _publish_snapshot_event(
            token_ids=list(request.prompt_tokens) + list(state.generated_token_ids),
            target_cache=self.target_cache,
            target_hidden=end_target_hidden,
            last_logits=state.last_cycle_logits,
            snapshot_service=request.snapshot_service,
            kind="generation",
            require_logits=False,
            snapshot_boundary=end_total_len,
            allow_full_context_draft_layers=self.allow_full_context_draft_layers,
        )
        yield_pause.done(_snapshot_build)
        if snapshot_event is not None:
            _pre_yield = yield_pause.mark()
            yield snapshot_event
            yield_pause.done(_pre_yield)
        evt = self.memory_waterfall_event(
            "after_generation_snapshot_build",
            target_hidden_value=end_target_hidden,
            gen_hidden_chunks_value=feature_store.generation_chunks,
            extra={"snapshot_boundary": int(end_total_len)},
        )
        if evt is not None:
            _pre_yield = yield_pause.mark()
            yield evt
            yield_pause.done(_pre_yield)

    def _run_ddtree_decode_events(
        self,
        *,
        request: _SessionRequest,
        state: "_RequestState",
        prefill: _PrefillResult,
        yield_pause: "_YieldPauseTracker",
    ) -> Generator[EngineEvent, None, _DecodeResult]:
        target_cache = self.target_cache
        target_ops = self.target_ops
        draft_cache = self.draft_cache
        draft_backend = self.draft_backend
        target_layer_id_list = self.target_layer_id_list
        capture_layer_ids = self.capture_layer_ids
        profile_cycles = self.profile_cycles
        memory_waterfall = self.memory_waterfall
        target_model = self.target_model
        draft_model = self.draft_model
        runtime_config = self.runtime_config
        prompt_len = request.prompt_len
        max_new_tokens = request.max_new_tokens
        block_tokens = request.block_tokens
        stop_token_array = request.stop_token_array
        feature_store = prefill.feature_store
        suppress_token_mask = prefill.suppress_token_mask
        supports_prefix_snapshot = prefill.supports_prefix_snapshot
        collect_generation_snapshot_hidden = (
            request.should_collect_generation_snapshot_hidden(supports_prefix_snapshot)
        )
        supports_target_tree_verify = bool(
            target_ops.capabilities_for(target_model).supports_tree_verify
            and target_ops.supports_tree_cache(target_cache)
        )
        forbidden_copy_tokens = (
            set(int(token) for token in request.suppress_token_ids)
            if request.suppress_token_ids is not None
            else None
        )

        def _copy_draft_for_block(
            staged_first: mx.array,
            block_len: int,
            draft_context: mx.array,
        ) -> mx.array | None:
            if state.copyspec_disabled:
                return None
            candidate = self.copyspec_index.draft_after(
                int(staged_first.item()),
                max_tokens=max(0, int(block_len) - 1),
                forbidden_tokens=forbidden_copy_tokens,
            )
            if candidate is None:
                return None
            draft_backend.advance_context(
                draft_model=draft_model,
                draft_cache=draft_cache,
                draft_context=draft_context,
            )
            return mx.array(candidate, dtype=mx.uint32)

        def _waterfall_event(
            phase: str,
            *,
            target_hidden_value: Any = None,
            gen_hidden_chunks_value: Any = None,
            extra: Optional[dict[str, Any]] = None,
        ) -> Optional[MemoryWaterfallEvent]:
            return self.memory_waterfall_event(
                phase,
                target_hidden_value=target_hidden_value,
                gen_hidden_chunks_value=gen_hidden_chunks_value,
                extra=extra,
            )

        first_token_yielded = False
        if max_new_tokens > 0:
            first_token_yielded = True
            assert state.staged_first is not None
            _pre_yield = yield_pause.mark()
            yield TokenEvent(
                token_id=int(state.staged_first.item()),
                generated_tokens=1,
                acceptance_ratio=0.0,
                cycles_completed=0,
            )
            yield_pause.done(_pre_yield)

        cycle_config = resolve_speculative_cycle_config(
            runtime_config,
            draft_model,
            block_tokens,
        )
        effective_block_tokens = cycle_config.effective_block_tokens
        verify_len_cap = cycle_config.verify_len_cap
        state.start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        copyspec_hits_total = 0
        copyspec_tokens_total = 0
        seen_draft_cycle = False
        cycle_profiles: list[CycleCompleteEvent] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "commit": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
            "cycle_wall": 0,
            "yield_pause": 0,
        }
        decode_clear_interval = (
            _DECODE_CLEAR_CACHE_INTERVAL_TOKENS if self.clear_cache_boundaries else 0
        )
        next_decode_clear_at = decode_clear_interval

        while len(state.generated_token_ids) < max_new_tokens:
            cycle_start_ns = time.perf_counter_ns() if profile_cycles else 0
            cycle_pause_start_ns = yield_pause.pause_ns if profile_cycles else 0
            cycle_prefix_len = int(state.start)
            if (
                decode_clear_interval > 0
                and len(state.generated_token_ids) >= next_decode_clear_at
            ):
                self.clear_cache_boundary()
                while next_decode_clear_at <= len(state.generated_token_ids):
                    next_decode_clear_at += decode_clear_interval

            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            commit_cycle_ns = 0
            remaining = max_new_tokens - len(state.generated_token_ids)
            block_len = max(1, min(effective_block_tokens, remaining))
            verify_token_count = verify_token_count_for_block(block_len, verify_len_cap)
            assert state.staged_first is not None
            current_staged_first = state.staged_first.astype(mx.uint32)
            copyspec_tokens_cycle = 0

            if block_len <= 1:
                candidate_ids_list = [current_staged_first[:1]]
                candidate_sources = ["greedy"]
            else:
                draft_context = feature_store.require_current_hidden()
                draft_start_ns = time.perf_counter_ns()
                copied = _copy_draft_for_block(
                    current_staged_first,
                    block_len,
                    draft_context,
                )
                if copied is not None:
                    copied_ids = ddtree_candidate_token_ids(
                        prefix_tokens=current_staged_first,
                        suffix_tokens=copied,
                        block_len=block_len,
                    )
                    candidate_ids_list = [copied_ids]
                    candidate_sources = ["copyspec"]
                    copyspec_tokens_cycle = int(copied.shape[0])
                    draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    draft_ns_total += draft_cycle_ns
                    if not seen_draft_cycle:
                        draft_prefill_ns += draft_cycle_ns
                        seen_draft_cycle = True
                    else:
                        draft_incremental_ns += draft_cycle_ns
                else:
                    copyspec_tokens_cycle = 0
                    draft_prefix_snapshot = ddtree_snapshot_cache(draft_cache)
                    topk_block_len = (
                        verify_token_count
                        if supports_target_tree_verify
                        else block_len
                    )
                    drafted_all, top_ids, top_values = draft_backend.draft_with_topk(
                        target_model=target_model,
                        target_ops=target_ops,
                        draft_model=draft_model,
                        draft_cache=draft_cache,
                        prefix_tokens=current_staged_first,
                        draft_context=draft_context,
                        block_len=topk_block_len,
                        suppress_token_mask=suppress_token_mask,
                        top_width=_DDTREE_TOP_WIDTH,
                    )
                    draft_after_greedy_snapshot = ddtree_snapshot_cache(draft_cache)
                    if supports_target_tree_verify:
                        ddtree_restore_cache(draft_after_greedy_snapshot)
                        draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                        draft_ns_total += draft_cycle_ns
                        if not seen_draft_cycle:
                            draft_prefill_ns += draft_cycle_ns
                            seen_draft_cycle = True
                        else:
                            draft_incremental_ns += draft_cycle_ns

                        tree_budget = max(0, int(verify_token_count) - 1)
                        tree = build_flat_ddtree(
                            top_token_ids_desc=top_ids[:tree_budget],
                            top_scores_desc=top_values[:tree_budget],
                            budget=tree_budget,
                        )
                        tree_inputs = build_flat_tree_inputs(
                            tree,
                            root_token_id=int(current_staged_first.item()),
                            prefix_len=state.start,
                        )
                        sample_memory_cycle = memory_waterfall and _should_sample_memory_cycle(
                            state.cycles_completed + 1
                        )
                        if sample_memory_cycle:
                            evt = _waterfall_event(
                                "before_verify_cycle",
                                target_hidden_value=feature_store.current_hidden,
                                gen_hidden_chunks_value=feature_store.generation_chunks,
                                extra={
                                    "cycle": int(state.cycles_completed + 1),
                                    "start": int(state.start),
                                    "ddtree_tree_size": int(tree_inputs.size),
                                },
                            )
                            if evt is not None:
                                _pre_yield = yield_pause.mark()
                                yield evt
                                yield_pause.done(_pre_yield)

                        verify_start_ns = time.perf_counter_ns()
                        logits, hidden_states = target_ops.verify_tree_block(
                            target_model=target_model,
                            tree_inputs=tree_inputs,
                            target_cache=target_cache,
                            capture_layer_ids=capture_layer_ids,
                        )
                        eval_logits_and_captured(logits, hidden_states)
                        posterior = greedy_tokens_with_mask(logits, suppress_token_mask).squeeze(0)
                        mx.eval(posterior)
                        verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
                        verify_ns_total += verify_cycle_ns
                        if sample_memory_cycle:
                            evt = _waterfall_event(
                                "after_verify_cycle",
                                target_hidden_value=feature_store.current_hidden,
                                gen_hidden_chunks_value=feature_store.generation_chunks,
                                extra={
                                    "cycle": int(state.cycles_completed + 1),
                                    "start": int(state.start),
                                    "ddtree_tree_size": int(tree_inputs.size),
                                },
                            )
                            if evt is not None:
                                _pre_yield = yield_pause.mark()
                                yield evt
                                yield_pause.done(_pre_yield)

                        acceptance_start_ns = time.perf_counter_ns() if profile_cycles else 0
                        accepted_slots, next_token_id = follow_verified_tree(
                            tree,
                            [int(token_id) for token_id in posterior.tolist()],
                        )
                        acceptance_len = max(0, len(accepted_slots) - 1)
                        state.acceptance_history.append(acceptance_len)
                        if profile_cycles:
                            acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns

                        hidden_extract_start_ns = time.perf_counter_ns() if profile_cycles else 0
                        accepted_hidden_states = ddtree_select_tree_slots(
                            hidden_states,
                            accepted_slots,
                        )
                        committed_hidden = target_ops.extract_context_feature(
                            accepted_hidden_states,
                            target_layer_id_list,
                        )
                        if profile_cycles:
                            mx.eval(committed_hidden, posterior)
                        else:
                            mx.async_eval(committed_hidden)
                        if profile_cycles:
                            hidden_extract_cycle_ns = (
                                time.perf_counter_ns() - hidden_extract_start_ns
                            )

                        commit_count = len(accepted_slots)
                        committed_segment = mx.take(
                            tree_inputs.token_ids,
                            mx.array(accepted_slots, dtype=mx.int32),
                            axis=0,
                        )
                        commit_start_ns = time.perf_counter_ns()
                        state.start += commit_count
                        feature_store.commit_generation(
                            committed_hidden,
                            collect_snapshot=collect_generation_snapshot_hidden,
                        )
                        state.last_cycle_logits = logits[:, int(accepted_slots[-1]), :]
                        commit_cycle_ns = time.perf_counter_ns() - commit_start_ns
                        replay_cycle_ns = target_ops.restore_after_tree_acceptance(
                            target_cache,
                            accepted_tree_indices=accepted_slots,
                        )
                        if sample_memory_cycle:
                            evt = _waterfall_event(
                                "after_rollback",
                                target_hidden_value=feature_store.current_hidden,
                                gen_hidden_chunks_value=feature_store.generation_chunks,
                                extra={
                                    "cycle": int(state.cycles_completed + 1),
                                    "start": int(state.start),
                                    "commit_count": int(commit_count),
                                    "ddtree_tree_size": int(tree_inputs.size),
                                },
                            )
                            if evt is not None:
                                _pre_yield = yield_pause.mark()
                                yield evt
                                yield_pause.done(_pre_yield)
                        replay_ns_total += replay_cycle_ns
                        state.cycles_completed += 1
                        commit_ns_total += commit_cycle_ns

                        state.accepted_from_draft += acceptance_len
                        staged_first_next = mx.array([int(next_token_id)], dtype=mx.uint32)
                        committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
                        self.copyspec_index.append_committed(committed_ids)
                        for token_id in committed_ids:
                            if len(state.generated_token_ids) >= max_new_tokens:
                                break
                            state.generated_token_ids.append(token_id)
                            if first_token_yielded:
                                first_token_yielded = False
                                continue
                            _pre_yield = yield_pause.mark()
                            yield TokenEvent(
                                token_id=int(token_id),
                                generated_tokens=len(state.generated_token_ids),
                                acceptance_ratio=(
                                    state.accepted_from_draft / len(state.generated_token_ids)
                                    if state.generated_token_ids
                                    else 0.0
                                ),
                                cycles_completed=int(state.cycles_completed),
                                copyspec_hits=int(copyspec_hits_total),
                                copyspec_tokens=int(copyspec_tokens_total),
                            )
                            yield_pause.done(_pre_yield)

                        stop_hit = False
                        if stop_token_array is not None:
                            stop_hit = bool(
                                mx.any(
                                    mx.equal(
                                        committed_segment[:, None],
                                        stop_token_array[None, :],
                                    )
                                ).item()
                            )
                        if stop_hit:
                            break

                        state.staged_first = staged_first_next

                        if profile_cycles:
                            cycle_wall_ns = time.perf_counter_ns() - cycle_start_ns
                            cycle_pause_ns = max(
                                0,
                                yield_pause.pause_ns - cycle_pause_start_ns,
                            )
                            cycle_total_ns = max(0, cycle_wall_ns - cycle_pause_ns)
                            named_ns = (
                                draft_cycle_ns
                                + verify_cycle_ns
                                + acceptance_cycle_ns
                                + hidden_extract_cycle_ns
                                + commit_cycle_ns
                                + replay_cycle_ns
                            )
                            other_cycle_ns = max(0, cycle_total_ns - named_ns)
                            cycle_profile_entry = CycleCompleteEvent(
                                cycle=int(state.cycles_completed),
                                block_len=int(block_len),
                                commit_count=int(commit_count),
                                acceptance_len=int(acceptance_len),
                                prefix_len=int(cycle_prefix_len),
                                draft_us=ns_to_us(draft_cycle_ns),
                                verify_us=ns_to_us(verify_cycle_ns),
                                acceptance_us=ns_to_us(acceptance_cycle_ns),
                                hidden_extraction_us=ns_to_us(hidden_extract_cycle_ns),
                                commit_us=ns_to_us(commit_cycle_ns),
                                rollback_us=ns_to_us(replay_cycle_ns),
                                other_us=ns_to_us(other_cycle_ns),
                                cycle_total_us=ns_to_us(cycle_total_ns),
                                cycle_wall_us=ns_to_us(cycle_wall_ns),
                                yield_pause_us=ns_to_us(cycle_pause_ns),
                                verify_token_count=int(tree_inputs.size),
                                draft_source="ddtree_tree",
                                candidate_count=int(tree_inputs.size),
                            )
                            cycle_profiles.append(cycle_profile_entry)
                            _pre_yield = yield_pause.mark()
                            yield cycle_profile_entry
                            yield_pause.done(_pre_yield)
                            profile_totals_ns["draft"] += draft_cycle_ns
                            profile_totals_ns["verify"] += verify_cycle_ns
                            profile_totals_ns["acceptance"] += acceptance_cycle_ns
                            profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                            profile_totals_ns["commit"] += commit_cycle_ns
                            profile_totals_ns["rollback"] += replay_cycle_ns
                            profile_totals_ns["other"] += other_cycle_ns
                            profile_totals_ns["cycle_total"] += cycle_total_ns
                            profile_totals_ns["cycle_wall"] += cycle_wall_ns
                            profile_totals_ns["yield_pause"] += cycle_pause_ns
                        continue

                    greedy_ids = ddtree_candidate_token_ids(
                        prefix_tokens=current_staged_first,
                        suffix_tokens=drafted_all,
                        block_len=block_len,
                    )
                    branch_prefixes: list[mx.array] = []
                    branch_sources: list[str] = []
                    for branch_pos in ddtree_branch_positions(
                        top_values_desc=top_values,
                        block_len=block_len,
                        max_branch_positions=_DDTREE_MAX_BRANCH_POSITIONS,
                        strategy="first",
                    ):
                        greedy_token = int(greedy_ids[1 + branch_pos].item())
                        seen_branch_tokens = {greedy_token}
                        for branch_token in top_ids[branch_pos][:_DDTREE_TOP_WIDTH]:
                            branch_token = int(branch_token)
                            if branch_token in seen_branch_tokens:
                                continue
                            seen_branch_tokens.add(branch_token)
                            branch_prefixes.append(
                                mx.concatenate(
                                    [
                                        greedy_ids[: 1 + branch_pos],
                                        mx.array([branch_token], dtype=mx.uint32),
                                    ],
                                    axis=0,
                                )
                            )
                            branch_sources.append(f"ddtree:p{branch_pos}:t{branch_token}")

                    candidate_ids_list = [greedy_ids]
                    candidate_sources = ["greedy"]
                    if branch_prefixes:
                        ddtree_restore_cache(draft_prefix_snapshot)
                        branch_cache = ddtree_clone_cache_for_batch(
                            draft_cache,
                            len(branch_prefixes),
                        )
                        branch_ids_list = draft_backend.draft_branch_blocks_batch(
                            target_model=target_model,
                            target_ops=target_ops,
                            draft_model=draft_model,
                            draft_cache=branch_cache,
                            branch_prefixes=branch_prefixes,
                            draft_context=draft_context,
                            block_len=block_len,
                            suppress_token_mask=suppress_token_mask,
                        )
                        candidate_ids_list.extend(branch_ids_list)
                        candidate_sources.extend(branch_sources)
                    ddtree_restore_cache(draft_after_greedy_snapshot)
                    draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    draft_ns_total += draft_cycle_ns
                    if not seen_draft_cycle:
                        draft_prefill_ns += draft_cycle_ns
                        seen_draft_cycle = True
                    else:
                        draft_incremental_ns += draft_cycle_ns


            if verify_token_count < block_len:
                candidate_ids_list = [
                    candidate_ids[:verify_token_count]
                    for candidate_ids in candidate_ids_list
                ]

            sample_memory_cycle = memory_waterfall and _should_sample_memory_cycle(
                state.cycles_completed + 1
            )
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "before_verify_cycle",
                    target_hidden_value=feature_store.current_hidden,
                    gen_hidden_chunks_value=feature_store.generation_chunks,
                    extra={
                        "cycle": int(state.cycles_completed + 1),
                        "start": int(state.start),
                    },
                )
                if evt is not None:
                    _pre_yield = yield_pause.mark()
                    yield evt
                    yield_pause.done(_pre_yield)

            verify_start_ns = time.perf_counter_ns()
            batched_cache = ddtree_clone_cache_for_batch(
                target_cache,
                len(candidate_ids_list),
            )
            candidate_results, _verify_us = ddtree_verify_candidates_batch(
                target_model=target_model,
                target_ops=target_ops,
                target_cache=batched_cache,
                capture_layer_ids=capture_layer_ids,
                candidate_ids=candidate_ids_list,
                candidate_sources=candidate_sources,
                suppress_token_mask=suppress_token_mask,
                prefix_len=state.start,
            )
            verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
            verify_ns_total += verify_cycle_ns
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "after_verify_cycle",
                    target_hidden_value=feature_store.current_hidden,
                    gen_hidden_chunks_value=feature_store.generation_chunks,
                    extra={
                        "cycle": int(state.cycles_completed + 1),
                        "start": int(state.start),
                    },
                )
                if evt is not None:
                    _pre_yield = yield_pause.mark()
                    yield evt
                    yield_pause.done(_pre_yield)

            acceptance_start_ns = time.perf_counter_ns() if profile_cycles else 0
            best_index, best = max(
                enumerate(candidate_results),
                key=lambda pair: (
                    pair[1].commit_count,
                    1 if pair[1].source == "greedy" else 0,
                ),
            )
            acceptance_len = int(best.acceptance_len)
            state.acceptance_history.append(acceptance_len)
            if best.source == "copyspec":
                copyspec_hits_total += 1
                copyspec_tokens_total += copyspec_tokens_cycle or max(0, int(block_len) - 1)
                if acceptance_len == 0:
                    state.copyspec_disabled = True
            if profile_cycles:
                acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns

            ddtree_copy_selected_cache(
                dst_entries=target_cache,
                src_entries=batched_cache,
                batch_index=best_index,
            )

            hidden_extract_start_ns = time.perf_counter_ns() if profile_cycles else 0
            committed_hidden = target_ops.extract_context_feature(
                best.hidden_states,
                target_layer_id_list,
            )[:, : best.commit_count, :]
            if profile_cycles:
                mx.eval(committed_hidden, best.posterior)
            else:
                mx.async_eval(committed_hidden)
            if profile_cycles:
                hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = int(best.commit_count)
            committed_segment = best.ids[:commit_count]
            commit_start_ns = time.perf_counter_ns()
            state.start += commit_count
            feature_store.commit_generation(
                committed_hidden,
                collect_snapshot=collect_generation_snapshot_hidden,
            )
            state.last_cycle_logits = best.logits[:, acceptance_len, :]
            commit_cycle_ns = time.perf_counter_ns() - commit_start_ns
            replay_cycle_ns = target_ops.restore_after_acceptance(
                target_cache,
                target_len=state.start,
                acceptance_length=acceptance_len,
                drafted_tokens=max(0, int(best.ids.shape[0]) - 1),
            )
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "after_rollback",
                    target_hidden_value=feature_store.current_hidden,
                    gen_hidden_chunks_value=feature_store.generation_chunks,
                    extra={
                        "cycle": int(state.cycles_completed + 1),
                        "start": int(state.start),
                        "commit_count": int(commit_count),
                    },
                )
                if evt is not None:
                    _pre_yield = yield_pause.mark()
                    yield evt
                    yield_pause.done(_pre_yield)
            replay_ns_total += replay_cycle_ns
            state.cycles_completed += 1
            commit_ns_total += commit_cycle_ns

            state.accepted_from_draft += acceptance_len
            staged_first_next = best.posterior[acceptance_len : acceptance_len + 1]
            committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
            self.copyspec_index.append_committed(committed_ids)
            for token_id in committed_ids:
                if len(state.generated_token_ids) >= max_new_tokens:
                    break
                state.generated_token_ids.append(token_id)
                if first_token_yielded:
                    first_token_yielded = False
                    continue
                _pre_yield = yield_pause.mark()
                yield TokenEvent(
                    token_id=int(token_id),
                    generated_tokens=len(state.generated_token_ids),
                    acceptance_ratio=(
                        state.accepted_from_draft / len(state.generated_token_ids)
                        if state.generated_token_ids
                        else 0.0
                    ),
                    cycles_completed=int(state.cycles_completed),
                    copyspec_hits=int(copyspec_hits_total),
                    copyspec_tokens=int(copyspec_tokens_total),
                )
                yield_pause.done(_pre_yield)

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            if stop_hit:
                break

            state.staged_first = staged_first_next

            if profile_cycles:
                cycle_wall_ns = time.perf_counter_ns() - cycle_start_ns
                cycle_pause_ns = max(0, yield_pause.pause_ns - cycle_pause_start_ns)
                cycle_total_ns = max(0, cycle_wall_ns - cycle_pause_ns)
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + commit_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profile_entry = CycleCompleteEvent(
                    cycle=int(state.cycles_completed),
                    block_len=int(block_len),
                    commit_count=int(commit_count),
                    acceptance_len=int(acceptance_len),
                    prefix_len=int(cycle_prefix_len),
                    draft_us=ns_to_us(draft_cycle_ns),
                    verify_us=ns_to_us(verify_cycle_ns),
                    acceptance_us=ns_to_us(acceptance_cycle_ns),
                    hidden_extraction_us=ns_to_us(hidden_extract_cycle_ns),
                    commit_us=ns_to_us(commit_cycle_ns),
                    rollback_us=ns_to_us(replay_cycle_ns),
                    other_us=ns_to_us(other_cycle_ns),
                    cycle_total_us=ns_to_us(cycle_total_ns),
                    cycle_wall_us=ns_to_us(cycle_wall_ns),
                    yield_pause_us=ns_to_us(cycle_pause_ns),
                    verify_token_count=int(verify_token_count),
                    draft_source=str(best.source),
                    candidate_count=len(candidate_ids_list),
                )
                cycle_profiles.append(cycle_profile_entry)
                _pre_yield = yield_pause.mark()
                yield cycle_profile_entry
                yield_pause.done(_pre_yield)
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["commit"] += commit_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns
                profile_totals_ns["cycle_wall"] += cycle_wall_ns
                profile_totals_ns["yield_pause"] += cycle_pause_ns

        return _DecodeResult(
            effective_block_tokens=effective_block_tokens,
            verify_len_cap=verify_len_cap,
            adaptive_block_reductions=0,
            adaptive_block_cycles=0,
            adaptive_block_min=None,
            draft_ns_total=draft_ns_total,
            draft_prefill_ns=draft_prefill_ns,
            draft_incremental_ns=draft_incremental_ns,
            verify_ns_total=verify_ns_total,
            replay_ns_total=replay_ns_total,
            commit_ns_total=commit_ns_total,
            cycle_profiles=tuple(cycle_profiles),
            profile_totals_ns=profile_totals_ns,
            copyspec_hits=int(copyspec_hits_total),
            copyspec_tokens=int(copyspec_tokens_total),
        )

    def _run_decode_events(
        self,
        *,
        request: _SessionRequest,
        state: "_RequestState",
        prefill: _PrefillResult,
        yield_pause: "_YieldPauseTracker",
    ) -> Generator[EngineEvent, None, _DecodeResult]:
        if str(getattr(self.runtime_config, "verify_mode", "dflash")) == "ddtree":
            return (
                yield from self._run_ddtree_decode_events(
                    request=request,
                    state=state,
                    prefill=prefill,
                    yield_pause=yield_pause,
                )
            )

        target_cache = self.target_cache
        target_ops = self.target_ops
        draft_cache = self.draft_cache
        draft_backend = self.draft_backend
        target_layer_id_list = self.target_layer_id_list
        capture_layer_ids = self.capture_layer_ids
        profile_cycles = self.profile_cycles
        memory_waterfall = self.memory_waterfall
        target_model = self.target_model
        draft_model = self.draft_model
        runtime_config = self.runtime_config
        prompt_len = request.prompt_len
        max_new_tokens = request.max_new_tokens
        block_tokens = request.block_tokens
        stop_token_array = request.stop_token_array
        feature_store = prefill.feature_store
        suppress_token_mask = prefill.suppress_token_mask
        supports_prefix_snapshot = prefill.supports_prefix_snapshot
        collect_generation_snapshot_hidden = (
            request.should_collect_generation_snapshot_hidden(supports_prefix_snapshot)
        )

        def _waterfall_event(
            phase: str,
            *,
            target_hidden_value: Any = None,
            gen_hidden_chunks_value: Any = None,
            extra: Optional[dict[str, Any]] = None,
        ) -> Optional[MemoryWaterfallEvent]:
            return self.memory_waterfall_event(
                phase,
                target_hidden_value=target_hidden_value,
                gen_hidden_chunks_value=gen_hidden_chunks_value,
                extra=extra,
            )

        first_token_yielded = False
        if max_new_tokens > 0:
            first_token_yielded = True
            assert state.staged_first is not None
            _pre_yield = yield_pause.mark()
            yield TokenEvent(
                token_id=int(state.staged_first.item()),
                generated_tokens=1,
                acceptance_ratio=0.0,
                cycles_completed=0,
            )
            yield_pause.done(_pre_yield)

        cycle_config = resolve_speculative_cycle_config(
            runtime_config,
            draft_model,
            block_tokens,
        )
        effective_block_tokens = cycle_config.effective_block_tokens
        adaptive_block_policy = _AdaptiveBlockPolicy.from_runtime(
            runtime_config=runtime_config,
            effective_block_tokens=effective_block_tokens,
            verify_len_cap=cycle_config.verify_len_cap,
            prompt_len=request.prompt_len,
        )
        block_token_buffer = mx.full(
            (effective_block_tokens,),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        mask_token_tail = mx.full(
            (max(0, effective_block_tokens - 1),),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        verify_len_cap = cycle_config.verify_len_cap
        state.start = prompt_len
        forbidden_copy_tokens = (
            set(int(token) for token in request.suppress_token_ids)
            if request.suppress_token_ids is not None
            else None
        )

        def _copy_draft_for_block(
            staged_first: mx.array,
            block_len: int,
            draft_context: mx.array,
        ) -> mx.array | None:
            if state.copyspec_disabled:
                return None
            candidate = self.copyspec_index.draft_after(
                int(staged_first.item()),
                max_tokens=max(0, int(block_len) - 1),
                forbidden_tokens=forbidden_copy_tokens,
            )
            if candidate is None:
                return None
            draft_backend.advance_context(
                draft_model=draft_model,
                draft_cache=draft_cache,
                draft_context=draft_context,
            )
            return mx.array(candidate, dtype=mx.uint32)

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        seen_draft_cycle = False
        cycle_profiles: list[CycleCompleteEvent] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "commit": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
            "cycle_wall": 0,
            "yield_pause": 0,
        }
        decode_clear_interval = (
            _DECODE_CLEAR_CACHE_INTERVAL_TOKENS if self.clear_cache_boundaries else 0
        )
        next_decode_clear_at = decode_clear_interval

        while len(state.generated_token_ids) < max_new_tokens:
            track_cycle_wall = profile_cycles or adaptive_block_policy is not None
            cycle_start_ns = time.perf_counter_ns() if track_cycle_wall else 0
            cycle_pause_start_ns = yield_pause.pause_ns if track_cycle_wall else 0
            cycle_prefix_len = int(state.start)
            if (
                decode_clear_interval > 0
                and len(state.generated_token_ids) >= next_decode_clear_at
            ):
                self.clear_cache_boundary()
                while next_decode_clear_at <= len(state.generated_token_ids):
                    next_decode_clear_at += decode_clear_interval
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            commit_cycle_ns = 0
            remaining = max_new_tokens - len(state.generated_token_ids)
            block_limit = (
                adaptive_block_policy.block_limit()
                if adaptive_block_policy is not None
                else effective_block_tokens
            )
            block_len = max(1, min(effective_block_tokens, block_limit, remaining))
            block_token_buffer[:block_len] = int(draft_model.mask_token_id)
            assert state.staged_first is not None
            block_token_buffer[:1] = state.staged_first
            block_token_ids = block_token_buffer[:block_len]
            current_staged_first = state.staged_first
            drafted = None
            draft_source = "none"
            copyspec_tokens = 0

            if block_len > 1:
                if profile_cycles:
                    draft_start_ns = time.perf_counter_ns()
                    drafted = _copy_draft_for_block(
                        current_staged_first,
                        block_len,
                        feature_store.require_current_hidden(),
                    )
                    if drafted is not None:
                        draft_source = "copyspec"
                        copyspec_tokens = int(drafted.shape[0])
                    else:
                        drafted = draft_backend.draft_greedy(
                            target_model=target_model,
                            target_ops=target_ops,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=current_staged_first,
                            draft_context=feature_store.require_current_hidden(),
                            block_len=block_len,
                            mask_token_tail=mask_token_tail,
                            suppress_token_mask=suppress_token_mask,
                            async_launch=False,
                        )
                        draft_source = "dflash"
                    mx.eval(drafted)
                    draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    block_token_ids[1:block_len] = drafted
                else:
                    if (
                        state.prefetched_draft is not None
                        and int(state.prefetched_draft["block_len"]) == block_len
                    ):
                        drafted = state.prefetched_draft["drafted"]
                        current_staged_first = state.prefetched_draft["staged_first"]
                        draft_source = str(state.prefetched_draft["source"])
                    else:
                        draft_start_ns = time.perf_counter_ns()
                        drafted = _copy_draft_for_block(
                            current_staged_first,
                            block_len,
                            feature_store.require_current_hidden(),
                        )
                        if drafted is not None:
                            draft_source = "copyspec"
                            copyspec_tokens = int(drafted.shape[0])
                        else:
                            drafted = draft_backend.draft_greedy(
                                target_model=target_model,
                                target_ops=target_ops,
                                draft_model=draft_model,
                                draft_cache=draft_cache,
                                staged_first=current_staged_first,
                                draft_context=feature_store.require_current_hidden(),
                                block_len=block_len,
                                mask_token_tail=mask_token_tail,
                                suppress_token_mask=suppress_token_mask,
                                async_launch=True,
                            )
                            draft_source = "dflash"
                        draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    state.prefetched_draft = None
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            verify_token_count = verify_token_count_for_block(block_len, verify_len_cap)
            if profile_cycles or block_len <= 1:
                verify_token_ids = block_token_ids[:verify_token_count]
            elif verify_token_count <= 1:
                verify_token_ids = current_staged_first[:1]
            else:
                verify_token_ids = mx.concatenate(
                    [current_staged_first[:1], drafted[: verify_token_count - 1]],
                    axis=0,
                )
            verify_ids = verify_token_ids[None]
            target_ops.arm_rollback(target_cache, prefix_len=state.start)
            sample_memory_cycle = memory_waterfall and _should_sample_memory_cycle(
                state.cycles_completed + 1
            )
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "before_verify_cycle",
                    target_hidden_value=feature_store.current_hidden,
                    gen_hidden_chunks_value=feature_store.generation_chunks,
                    extra={
                        "cycle": int(state.cycles_completed + 1),
                        "start": int(state.start),
                    },
                )
                if evt is not None:
                    _pre_yield = yield_pause.mark()
                    yield evt
                    yield_pause.done(_pre_yield)
            verify_start_ns = time.perf_counter_ns()
            verify_logits, verify_hidden_states = target_ops.verify_block(
                target_model=target_model,
                verify_ids=verify_ids,
                target_cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            if profile_cycles:
                eval_logits_and_captured(verify_logits, verify_hidden_states)
            verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
            verify_ns_total += verify_cycle_ns
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "after_verify_cycle",
                    target_hidden_value=feature_store.current_hidden,
                    gen_hidden_chunks_value=feature_store.generation_chunks,
                    extra={
                        "cycle": int(state.cycles_completed + 1),
                        "start": int(state.start),
                    },
                )
                if evt is not None:
                    _pre_yield = yield_pause.mark()
                    yield evt
                    yield_pause.done(_pre_yield)

            acceptance_start_ns = time.perf_counter_ns() if profile_cycles else 0
            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
            if not profile_cycles:
                mx.async_eval(posterior)
            acceptance_len = int(
                _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
            )
            state.acceptance_history.append(acceptance_len)
            if profile_cycles:
                acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            hidden_extract_start_ns = time.perf_counter_ns() if profile_cycles else 0
            committed_hidden = target_ops.extract_context_feature(
                verify_hidden_states,
                target_layer_id_list,
            )[:, : (1 + acceptance_len), :]
            if profile_cycles:
                mx.eval(committed_hidden, posterior)
            else:
                mx.async_eval(committed_hidden)
            if profile_cycles:
                hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            commit_start_ns = time.perf_counter_ns()
            state.start += commit_count
            feature_store.commit_generation(
                committed_hidden,
                collect_snapshot=collect_generation_snapshot_hidden,
            )
            state.last_cycle_logits = verify_logits[:, acceptance_len, :]
            commit_cycle_ns = time.perf_counter_ns() - commit_start_ns
            replay_cycle_ns = target_ops.restore_after_acceptance(
                target_cache,
                target_len=state.start,
                acceptance_length=acceptance_len,
                drafted_tokens=max(0, verify_token_count - 1),
            )
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "after_rollback",
                    target_hidden_value=feature_store.current_hidden,
                    gen_hidden_chunks_value=feature_store.generation_chunks,
                    extra={
                        "cycle": int(state.cycles_completed + 1),
                        "start": int(state.start),
                        "commit_count": int(commit_count),
                    },
                )
                if evt is not None:
                    _pre_yield = yield_pause.mark()
                    yield evt
                    yield_pause.done(_pre_yield)
            replay_ns_total += replay_cycle_ns
            state.cycles_completed += 1
            commit_ns_total += commit_cycle_ns

            state.accepted_from_draft += acceptance_len
            if draft_source == "copyspec":
                state.copyspec_hits += 1
                state.copyspec_tokens += copyspec_tokens or max(0, int(block_len) - 1)
                if acceptance_len == 0:
                    state.copyspec_disabled = True
            staged_first_next = posterior[acceptance_len : acceptance_len + 1]
            if adaptive_block_policy is not None:
                cycle_wall_ns = time.perf_counter_ns() - cycle_start_ns
                cycle_pause_ns = max(0, yield_pause.pause_ns - cycle_pause_start_ns)
                adaptive_cycle_cost_ns = max(0, cycle_wall_ns - cycle_pause_ns)
                if adaptive_cycle_cost_ns <= 0:
                    adaptive_cycle_cost_ns = (
                        draft_cycle_ns
                        + verify_cycle_ns
                        + acceptance_cycle_ns
                        + hidden_extract_cycle_ns
                        + commit_cycle_ns
                        + replay_cycle_ns
                    )
                adaptive_block_policy.record(
                    block_len=block_len,
                    acceptance_len=acceptance_len,
                    cycle_cost_ns=adaptive_cycle_cost_ns,
                )
            committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
            self.copyspec_index.append_committed(committed_ids)
            if not profile_cycles:
                next_remaining = max_new_tokens - len(state.generated_token_ids) - commit_count
                next_block_limit = (
                    adaptive_block_policy.block_limit()
                    if adaptive_block_policy is not None
                    else effective_block_tokens
                )
                next_block_len = max(
                    1,
                    min(effective_block_tokens, next_block_limit, next_remaining),
                )
                if next_remaining > 0 and next_block_len > 1:
                    draft_start_ns = time.perf_counter_ns()
                    next_source = "copyspec"
                    next_drafted = _copy_draft_for_block(
                        staged_first_next,
                        next_block_len,
                        feature_store.require_current_hidden(),
                    )
                    if next_drafted is None:
                        next_source = "dflash"
                        next_drafted = draft_backend.draft_greedy(
                            target_model=target_model,
                            target_ops=target_ops,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=staged_first_next,
                            draft_context=feature_store.require_current_hidden(),
                            block_len=next_block_len,
                            mask_token_tail=mask_token_tail,
                            suppress_token_mask=suppress_token_mask,
                            async_launch=True,
                        )
                    launch_ns = time.perf_counter_ns() - draft_start_ns
                    draft_ns_total += launch_ns
                    draft_incremental_ns += launch_ns
                    state.prefetched_draft = {
                        "block_len": next_block_len,
                        "staged_first": staged_first_next,
                        "drafted": next_drafted,
                        "source": next_source,
                    }
                else:
                    state.prefetched_draft = None
            for token_id in committed_ids:
                if len(state.generated_token_ids) >= max_new_tokens:
                    break
                state.generated_token_ids.append(token_id)
                if first_token_yielded:
                    first_token_yielded = False
                    continue
                _pre_yield = yield_pause.mark()
                yield TokenEvent(
                    token_id=int(token_id),
                    generated_tokens=len(state.generated_token_ids),
                    acceptance_ratio=(
                        state.accepted_from_draft / len(state.generated_token_ids)
                        if state.generated_token_ids
                        else 0.0
                    ),
                    cycles_completed=int(state.cycles_completed),
                    adaptive_block_reductions=(
                        int(adaptive_block_policy.reductions)
                        if adaptive_block_policy is not None
                        else 0
                    ),
                    adaptive_block_cycles=(
                        int(adaptive_block_policy.reduced_cycles)
                        if adaptive_block_policy is not None
                        else 0
                    ),
                    adaptive_block_min=(
                        adaptive_block_policy.min_seen
                        if adaptive_block_policy is not None
                        else None
                    ),
                    copyspec_hits=int(state.copyspec_hits),
                    copyspec_tokens=int(state.copyspec_tokens),
                )
                yield_pause.done(_pre_yield)

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            if stop_hit:
                break

            state.staged_first = staged_first_next

            if profile_cycles:
                cycle_wall_ns = time.perf_counter_ns() - cycle_start_ns
                cycle_pause_ns = max(0, yield_pause.pause_ns - cycle_pause_start_ns)
                cycle_total_ns = max(0, cycle_wall_ns - cycle_pause_ns)
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + commit_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profile_entry = CycleCompleteEvent(
                    cycle=int(state.cycles_completed),
                    block_len=int(block_len),
                    commit_count=int(commit_count),
                    acceptance_len=int(acceptance_len),
                    prefix_len=int(cycle_prefix_len),
                    draft_us=ns_to_us(draft_cycle_ns),
                    verify_us=ns_to_us(verify_cycle_ns),
                    acceptance_us=ns_to_us(acceptance_cycle_ns),
                    hidden_extraction_us=ns_to_us(hidden_extract_cycle_ns),
                    commit_us=ns_to_us(commit_cycle_ns),
                    rollback_us=ns_to_us(replay_cycle_ns),
                    other_us=ns_to_us(other_cycle_ns),
                    cycle_total_us=ns_to_us(cycle_total_ns),
                    cycle_wall_us=ns_to_us(cycle_wall_ns),
                    yield_pause_us=ns_to_us(cycle_pause_ns),
                    verify_token_count=int(verify_token_count),
                    draft_source=str(draft_source),
                    candidate_count=1,
                )
                cycle_profiles.append(cycle_profile_entry)
                _pre_yield = yield_pause.mark()
                yield cycle_profile_entry
                yield_pause.done(_pre_yield)
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["commit"] += commit_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns
                profile_totals_ns["cycle_wall"] += cycle_wall_ns
                profile_totals_ns["yield_pause"] += cycle_pause_ns

        return _DecodeResult(
            effective_block_tokens=effective_block_tokens,
            verify_len_cap=verify_len_cap,
            adaptive_block_reductions=(
                int(adaptive_block_policy.reductions)
                if adaptive_block_policy is not None
                else 0
            ),
            adaptive_block_cycles=(
                int(adaptive_block_policy.reduced_cycles)
                if adaptive_block_policy is not None
                else 0
            ),
            adaptive_block_min=(
                adaptive_block_policy.min_seen
                if adaptive_block_policy is not None
                else None
            ),
            draft_ns_total=draft_ns_total,
            draft_prefill_ns=draft_prefill_ns,
            draft_incremental_ns=draft_incremental_ns,
            verify_ns_total=verify_ns_total,
            replay_ns_total=replay_ns_total,
            commit_ns_total=commit_ns_total,
            cycle_profiles=tuple(cycle_profiles),
            profile_totals_ns=profile_totals_ns,
            copyspec_hits=int(state.copyspec_hits),
            copyspec_tokens=int(state.copyspec_tokens),
            adaptive_totals=(
                adaptive_block_policy.metrics()
                if adaptive_block_policy is not None
                else {}
            ),
        )

    def run_events(self, request: _SessionRequest) -> Iterator[EngineEvent]:
        draft_sink_size = self.draft_sink_size
        draft_window_size = self.draft_window_size
        target_fa_window = self.target_fa_window
        profile_cycles = self.profile_cycles
        memory_waterfall = self.memory_waterfall
        clear_cache_boundaries = self.clear_cache_boundaries
        quantize_kv_cache = self.quantize_kv_cache
        prompt_len = request.prompt_len

        yield_pause = _YieldPauseTracker(enabled=bool(profile_cycles or memory_waterfall))
        state = _RequestState()

        try:
            prefill = yield from self._run_prefill_events(
                request=request,
                state=state,
                yield_pause=yield_pause,
            )
            feature_store = prefill.feature_store
            start_ns = prefill.start_ns
            prefill_ns = prefill.prefill_ns
            supports_prefix_snapshot = prefill.supports_prefix_snapshot

            decode = yield from self._run_decode_events(
                request=request,
                state=state,
                prefill=prefill,
                yield_pause=yield_pause,
            )

            yield from self._run_generation_snapshot_events(
                request=request,
                state=state,
                feature_store=feature_store,
                supports_prefix_snapshot=supports_prefix_snapshot,
                yield_pause=yield_pause,
            )

            elapsed_us = (time.perf_counter_ns() - start_ns - yield_pause.pause_ns) / 1_000.0
            first_20 = state.acceptance_history[:20]
            last_20 = state.acceptance_history[-20:]
            summary = SummaryEvent(
                elapsed_us=elapsed_us,
                prompt_token_count=int(prompt_len),
                generated_token_ids=tuple(int(x) for x in state.generated_token_ids),
                generation_tokens=len(state.generated_token_ids),
                accepted_from_draft=int(state.accepted_from_draft),
                acceptance_ratio=(
                    state.accepted_from_draft / len(state.generated_token_ids)
                    if state.generated_token_ids
                    else 0.0
                ),
                block_tokens=int(decode.effective_block_tokens),
                cycles_completed=int(state.cycles_completed),
                phase_timings_us={
                    "prefill": prefill_ns / 1_000.0,
                    "draft": decode.draft_ns_total / 1_000.0,
                    "draft_prefill": decode.draft_prefill_ns / 1_000.0,
                    "draft_incremental": decode.draft_incremental_ns / 1_000.0,
                    "verify": decode.verify_ns_total / 1_000.0,
                    "replay": decode.replay_ns_total / 1_000.0,
                    "commit": decode.commit_ns_total / 1_000.0,
                },
                verify_len_cap=int(decode.verify_len_cap),
                quantize_kv_cache=bool(quantize_kv_cache),
                target_fa_window=int(target_fa_window),
                draft_sink_size=int(draft_sink_size),
                draft_window_size=int(draft_window_size),
                clear_cache_boundaries=bool(clear_cache_boundaries),
                tokens_per_cycle=(
                    len(state.generated_token_ids) / state.cycles_completed
                    if state.cycles_completed > 0
                    else 0.0
                ),
                acceptance_history=tuple(int(x) for x in state.acceptance_history),
                acceptance_first_20_avg=(sum(first_20) / len(first_20)) if first_20 else 0.0,
                acceptance_last_20_avg=(sum(last_20) / len(last_20)) if last_20 else 0.0,
                adaptive_block_reductions=int(decode.adaptive_block_reductions),
                adaptive_block_cycles=int(decode.adaptive_block_cycles),
                adaptive_block_min=decode.adaptive_block_min,
                peak_memory_gb=(
                    float(mx.get_peak_memory()) / 1e9
                    if hasattr(mx, "get_peak_memory")
                    else None
                ),
                cycle_profile_us=(
                    decode.cycle_profiles
                    if profile_cycles
                    else None
                ),
                cycle_profile_totals_us=(
                    {
                        key: ns_to_us(value) for key, value in decode.profile_totals_ns.items()
                    }
                    if profile_cycles
                    else None
                ),
                copyspec_hits=int(decode.copyspec_hits),
                copyspec_tokens=int(decode.copyspec_tokens),
                adaptive_metrics=decode.adaptive_totals or None,
            )
            yield summary
        finally:
            self.close()

    def close(self) -> None:
        self.target_ops.cleanup_generation_caches(self.target_cache, self.draft_cache)
        self.clear_cache_boundary()


@dataclass
class _RequestState:
    prefill_logits: mx.array | None = None
    last_cycle_logits: mx.array | None = None
    generated_token_ids: list[int] = field(default_factory=list)
    accepted_from_draft: int = 0
    cycles_completed: int = 0
    acceptance_history: list[int] = field(default_factory=list)
    start: int = 0
    staged_first: mx.array | None = None
    prefetched_draft: dict[str, Any] | None = None
    copyspec_hits: int = 0
    copyspec_tokens: int = 0
    copyspec_disabled: bool = False


@dataclass
class _YieldPauseTracker:
    enabled: bool
    pause_ns: int = 0

    def mark(self) -> int:
        return time.perf_counter_ns() if self.enabled else 0

    def done(self, mark: int) -> None:
        if self.enabled:
            self.pause_ns += time.perf_counter_ns() - mark


def _publish_snapshot_event(
    *,
    token_ids: list[int],
    target_cache: Any,
    target_hidden: Optional[mx.array],
    last_logits: Optional[mx.array],
    snapshot_service: Optional[SnapshotService],
    kind: Literal["prefill", "generation"],
    require_logits: bool,
    snapshot_boundary: int,
    allow_full_context_draft_layers: bool,
    from_snapshot: bool = False,
    snap_prefix_len: int = 0,
    l2_only: bool = False,
) -> Optional[SnapshotPublishedEvent]:
    if snapshot_service is None or target_hidden is None:
        return None
    publication = snapshot_service.publish(
        token_ids=token_ids,
        target_cache=target_cache,
        target_hidden=target_hidden,
        last_logits=last_logits,
        kind=kind,
        require_logits=require_logits,
        snapshot_boundary=snapshot_boundary,
        allow_full_attention_context=allow_full_context_draft_layers,
        from_snapshot=from_snapshot,
        snap_prefix_len=snap_prefix_len,
        l2_only=l2_only,
    )
    if publication is None:
        return None
    return _snapshot_published_event(publication)


def _snapshot_published_event(publication: SnapshotPublication) -> SnapshotPublishedEvent:
    return SnapshotPublishedEvent(
        kind=publication.kind,
        snapshot_boundary=int(publication.snapshot_boundary),
        prefix_len=int(publication.prefix_len),
        insert_ms=float(publication.insert_ms),
        admitted=bool(publication.admitted),
        from_snapshot=bool(publication.from_snapshot),
        snap_prefix_len=int(publication.snap_prefix_len),
    )


def stream_dflash_generate_impl(
    *,
    target_model: Any,
    target_ops: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    draft_backend: DraftBackend,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    prefix_snapshot: Optional[DFlashPrefixSnapshot] = None,
    snapshot_service: Optional[SnapshotService] = None,
    stable_prefix_len: Optional[int] = None,
    prefix_cache_active: bool = False,
    publish_generation_snapshot: bool = True,
    runtime_context: Any,
) -> Iterator[EngineEvent]:
    target_capabilities = target_ops.capabilities_for(target_model)
    supports_prefix_snapshot = bool(
        getattr(target_capabilities, "supports_prefix_snapshot", True)
    )
    allow_full_context_draft_layers = bool(
        getattr(target_capabilities, "supports_full_context_draft_layers", False)
    )
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    fallback_reason: Optional[str] = None

    if runtime_context is None:
        raise ValueError("runtime_context is required")
    runtime_config = runtime_context.runtime
    prompt_len = len(prompt_tokens)
    configured_max_ctx = int(runtime_config.dflash_max_ctx)
    dflash_max_ctx = configured_max_ctx if configured_max_ctx > 0 else sys.maxsize
    target_fa_window = int(runtime_config.target_fa_window)
    projected_ctx = prompt_len + max(0, int(max_new_tokens))
    if projected_ctx >= dflash_max_ctx:
        fallback_reason = (
            f"projected_ctx={projected_ctx} "
            f"(prompt_len={prompt_len}, max_new_tokens={int(max_new_tokens)}) "
            f">= DFLASH_MAX_CTX={dflash_max_ctx}"
        )
        yield from stream_baseline_generate(
            target_model=target_model,
            target_ops=target_ops,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            fallback_reason=fallback_reason,
        )
        return
    request = _SessionRequest.from_tokens(
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        block_tokens=block_tokens,
        stop_token_ids=stop_token_ids,
        suppress_token_ids=suppress_token_ids,
        prefix_snapshot=prefix_snapshot,
        snapshot_service=snapshot_service,
        stable_prefix_len=stable_prefix_len,
        prefix_cache_active=prefix_cache_active,
        publish_generation_snapshot=publish_generation_snapshot,
    )

    session = SpeculativeSession.open(
        target_model=target_model,
        draft_model=draft_model,
        draft_backend=draft_backend,
        target_ops=target_ops,
        supports_prefix_snapshot=supports_prefix_snapshot,
        allow_full_context_draft_layers=allow_full_context_draft_layers,
        prompt_tokens=request.prompt_tokens,
        max_new_tokens=max_new_tokens,
        prefix_snapshot=prefix_snapshot,
        quantize_kv_cache=quantize_kv_cache,
        target_fa_window=target_fa_window,
        runtime_context=runtime_context,
    )
    yield from session.run_events(request)

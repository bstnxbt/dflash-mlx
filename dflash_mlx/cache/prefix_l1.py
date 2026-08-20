# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from dflash_mlx.observability.cache import record_cache_event
from dflash_mlx.diagnostics import TraceConfig
from dflash_mlx.cache.codecs import (
    slice_snapshot_at_sidecar_boundary,
    snapshot_covers_sidecar_context,
)
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.engine.prefill import snapshot_covers_prefix


@dataclass(frozen=True)
class PrefixCacheL1InsertResult:
    admitted: bool
    removed_snapshots: tuple[DFlashPrefixSnapshot, ...] = ()
    inserted_evicted_snapshot: DFlashPrefixSnapshot | None = None
    skipped_too_long: bool = False


class DFlashPrefixCache:
    def __init__(
        self,
        *,
        max_entries: int = 4,
        max_bytes: int = 8 * 1024 * 1024 * 1024,
        cross_kind_prune: bool = True,
        trace_config: Optional[TraceConfig] = None,
        max_snapshot_tokens: int = 24000,
        frontier_stride: int = 0,
    ):
        self._entries: dict[int, DFlashPrefixSnapshot] = {}
        self._lru_order: list[int] = []
        self._next_id = 0
        self._max_entries = int(max_entries)
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._cross_kind_prune = bool(cross_kind_prune)
        self._trace_config = trace_config
        self._max_snapshot_tokens = max(0, int(max_snapshot_tokens))
        self._frontier_stride = max(0, int(frontier_stride))
        self._stats: dict[str, int] = {
            "exact_hits": 0,
            "prefix_hits": 0,
            "misses": 0,
            "insertions": 0,
            "evictions": 0,
            "byte_budget_evictions": 0,
            "skipped_too_long": 0,
            "prefix_prunes": 0,
            "cross_kind_prunes": 0,
            "prefill_tokens_saved": 0,
            "fingerprint_rejects": 0,
            "sidecar_hits": 0,
            "coverage_rejects": 0,
            "generation_consumed": 0,
        }
        # Updated on every record=True lookup; "miss" / "l1_exact" / "l1_prefix".
        # Guarded by _lock; readable via last_hit_kind property.
        self._last_hit_kind: str = "miss"

    @property
    def frontier_stride(self) -> int:
        return int(self._frontier_stride)

    @property
    def last_hit_kind(self) -> str:
        with self._lock:
            return self._last_hit_kind

    @staticmethod
    def _usable_sidecar_boundary(
        snap: DFlashPrefixSnapshot,
        req_tuple: tuple[int, ...],
        require_full_coverage: bool = False,
    ) -> int:
        boundary = int(snap.sidecar_boundary)
        if boundary <= 0 or boundary > len(req_tuple):
            return 0
        if snap.sidecar_gdn_states is None or snap.sidecar_last_logits is None:
            return 0
        if req_tuple[:boundary] != snap.token_ids[:boundary]:
            return 0
        if not snapshot_covers_sidecar_context(
            snap,
            boundary,
            require_full_coverage=require_full_coverage,
        ):
            return 0
        return boundary

    def lookup(
        self,
        req_tokens: list[int] | tuple[int, ...],
        key: DFlashPrefixKey,
        *,
        record: bool = True,
        request_id: int | None = None,
        require_full_coverage: bool = False,
    ) -> tuple[int, Optional[DFlashPrefixSnapshot]]:
        req_tuple = tuple(int(t) for t in req_tokens)
        t_start = time.perf_counter_ns()

        with self._lock:
            best_len = 0
            best_id = -1
            best_snapshot: Optional[DFlashPrefixSnapshot] = None
            best_sidecar_len = 0
            best_sidecar_id = -1
            best_sidecar_carrier: Optional[DFlashPrefixSnapshot] = None
            saw_fingerprint_reject = 0
            coverage_rejects = 0
            longest_fingerprint_match_len = 0
            longest_fingerprint_first_divergence = -1
            for eid, snap in self._entries.items():
                if snap.key != key:
                    saw_fingerprint_reject += 1
                    continue
                snap_len = len(snap.token_ids)
                if snap_len == 0 or snap_len > len(req_tuple):
                    boundary = self._usable_sidecar_boundary(
                        snap, req_tuple, require_full_coverage
                    )
                    if boundary > best_sidecar_len:
                        best_sidecar_len = boundary
                        best_sidecar_id = eid
                        best_sidecar_carrier = snap
                    continue
                if req_tuple[:snap_len] != snap.token_ids:
                    common = 0
                    upper = min(snap_len, len(req_tuple))
                    for i in range(upper):
                        if req_tuple[i] != snap.token_ids[i]:
                            break
                        common += 1
                    if common > longest_fingerprint_match_len:
                        longest_fingerprint_match_len = common
                        longest_fingerprint_first_divergence = common
                    boundary = self._usable_sidecar_boundary(
                        snap, req_tuple, require_full_coverage
                    )
                    if boundary > best_sidecar_len:
                        best_sidecar_len = boundary
                        best_sidecar_id = eid
                        best_sidecar_carrier = snap
                    continue
                if require_full_coverage and not snapshot_covers_prefix(snap, snap_len):
                    coverage_rejects += 1
                    continue
                if snap_len > best_len:
                    best_len = snap_len
                    best_id = eid
                    best_snapshot = snap

            sidecar_used = False
            if best_sidecar_carrier is not None and best_sidecar_len > best_len:
                best_len = best_sidecar_len
                best_id = best_sidecar_id
                best_snapshot = slice_snapshot_at_sidecar_boundary(
                    best_sidecar_carrier,
                    require_full_coverage=require_full_coverage,
                )
                sidecar_used = True

            if record and coverage_rejects:
                self._stats["coverage_rejects"] += coverage_rejects

            if best_snapshot is not None and best_len > 0:
                exact = best_len == len(req_tuple)
                if exact and (
                    best_snapshot.kind != "prefill"
                    or best_snapshot.last_logits is None
                ):
                    best_snapshot = None
                else:
                    # Served generation snapshots are dominated by the successor
                    # published at request end; consume them on first serve.
                    if record:
                        consumed = sidecar_used or best_snapshot.kind == "generation"
                        if consumed:
                            self._entries.pop(best_id, None)
                            if best_id in self._lru_order:
                                self._lru_order.remove(best_id)
                            self._stats["generation_consumed"] += 1
                        else:
                            if best_id in self._lru_order:
                                self._lru_order.remove(best_id)
                            self._lru_order.append(best_id)
                        if exact:
                            self._stats["exact_hits"] += 1
                            self._last_hit_kind = "l1_exact"
                        else:
                            self._stats["prefix_hits"] += 1
                            self._last_hit_kind = "l1_prefix"
                        if sidecar_used:
                            self._stats["sidecar_hits"] += 1
                        self._stats["prefill_tokens_saved"] += best_len
                        entries_count_log = len(self._entries)
                        self._log_cache(
                            op="lookup",
                            result="exact_hit" if exact else "prefix_hit",
                            request_id=request_id,
                            req_tokens=len(req_tuple),
                            matched_len=int(best_len),
                            entries=entries_count_log,
                            consumed=int(consumed),
                            elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
                        )
                    return (best_len, best_snapshot)

            if not record:
                return (0, None)

            self._stats["misses"] += 1
            self._last_hit_kind = "miss"
            miss_reason = "empty_cache"
            if self._entries:
                if saw_fingerprint_reject == len(self._entries):
                    miss_reason = "fingerprint_reject_all"
                    self._stats["fingerprint_rejects"] += 1
                elif coverage_rejects > 0:
                    miss_reason = "coverage_reject"
                elif saw_fingerprint_reject > 0 and longest_fingerprint_match_len == 0:
                    miss_reason = "fingerprint_reject_partial_no_token_match"
                    self._stats["fingerprint_rejects"] += 1
                elif longest_fingerprint_match_len == 0:
                    miss_reason = "no_token_match"
                else:
                    miss_reason = "token_divergence"
            entries_count = len(self._entries)
            sfr = int(saw_fingerprint_reject)
            cov_rejects = int(coverage_rejects)
            longest_match = int(longest_fingerprint_match_len)
            first_div = int(longest_fingerprint_first_divergence)

        self._log_cache(
            op="lookup",
            result="miss",
            request_id=request_id,
            req_tokens=len(req_tuple),
            matched_len=0,
            entries=entries_count,
            fingerprint_reject=sfr,
            coverage_rejects=cov_rejects,
            miss_reason=miss_reason,
            longest_token_match_len=longest_match,
            first_divergence_pos=first_div,
            elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
        )

        return (0, None)

    def insert(self, snapshot: DFlashPrefixSnapshot) -> bool:
        return self.insert_with_evictions(snapshot).admitted

    def insert_with_evictions(
        self,
        snapshot: DFlashPrefixSnapshot,
        *,
        skip_too_long: bool = True,
    ) -> PrefixCacheL1InsertResult:
        t_start = time.perf_counter_ns()
        with self._lock:
            pre_entries = len(self._entries)
            pre_evictions = self._stats["evictions"]
            pre_byte_evictions = self._stats["byte_budget_evictions"]
            pre_prunes = self._stats["prefix_prunes"]
            pre_cross_prunes = self._stats["cross_kind_prunes"]

            max_tokens = self._max_snapshot_tokens
            token_cap_applies = snapshot.kind != "generation"
            if (
                skip_too_long
                and token_cap_applies
                and max_tokens > 0
                and len(snapshot.token_ids) > max_tokens
            ):
                self._stats["skipped_too_long"] += 1
                self._log_cache(
                    op="insert_skipped",
                    reason="too_long",
                    prefix_len=int(len(snapshot.token_ids)),
                    max_tokens=int(max_tokens),
                    entries_before=pre_entries,
                    entries_after=pre_entries,
                    elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
                )
                return PrefixCacheL1InsertResult(
                    admitted=False,
                    skipped_too_long=True,
                )

            if self._incoming_generation_should_yield_to_prefill(snapshot):
                self._stats["insertions"] += 1
                self._stats["evictions"] += 1
                self._stats["byte_budget_evictions"] += 1
                breakdown = snapshot.nbytes_breakdown()
                draft_context_bytes = int(
                    breakdown.get("draft_context", breakdown.get("target_hidden", 0))
                )
                self._log_cache(
                    op="insert",
                    kind=snapshot.kind,
                    prefix_len=int(snapshot.prefix_len),
                    nbytes=int(snapshot.nbytes),
                    bytes_fa_kv=int(breakdown.get("fa_kv", 0)),
                    bytes_gdn_state=int(breakdown.get("gdn_state", 0)),
                    bytes_draft_context=draft_context_bytes,
                    bytes_target_hidden=draft_context_bytes,
                    bytes_last_logits=int(breakdown.get("last_logits", 0)),
                    entries_before=pre_entries,
                    entries_after=pre_entries,
                    pruned=0,
                    cross_kind_pruned=0,
                    evicted=1,
                    byte_budget_evicted=1,
                    current_bytes=int(self._current_bytes()),
                    elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
                )
                return PrefixCacheL1InsertResult(
                    admitted=False,
                    removed_snapshots=(snapshot,),
                    inserted_evicted_snapshot=snapshot,
                )

            removed_ids = self._prune_dominated_prefixes(snapshot)
            eid = self._next_id
            self._next_id += 1
            self._entries[eid] = snapshot
            self._lru_order.append(eid)
            self._stats["insertions"] += 1
            removed_ids.extend(self._evict_until_under_budget())
            removed_snapshots = tuple(snapshot for _eid, snapshot in removed_ids)
            inserted_evicted_snapshot = next(
                (
                    removed_snapshot
                    for removed_eid, removed_snapshot in removed_ids
                    if removed_eid == eid
                ),
                None,
            )
            breakdown = snapshot.nbytes_breakdown()
            draft_context_bytes = int(
                breakdown.get("draft_context", breakdown.get("target_hidden", 0))
            )
            self._log_cache(
                op="insert",
                kind=snapshot.kind,
                prefix_len=int(snapshot.prefix_len),
                nbytes=int(snapshot.nbytes),
                bytes_fa_kv=int(breakdown.get("fa_kv", 0)),
                bytes_gdn_state=int(breakdown.get("gdn_state", 0)),
                bytes_draft_context=draft_context_bytes,
                bytes_target_hidden=draft_context_bytes,
                bytes_last_logits=int(breakdown.get("last_logits", 0)),
                entries_before=pre_entries,
                entries_after=len(self._entries),
                pruned=int(self._stats["prefix_prunes"] - pre_prunes),
                cross_kind_pruned=int(self._stats["cross_kind_prunes"] - pre_cross_prunes),
                evicted=int(self._stats["evictions"] - pre_evictions),
                byte_budget_evicted=int(self._stats["byte_budget_evictions"] - pre_byte_evictions),
                current_bytes=int(self._current_bytes()),
                elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
            )
            return PrefixCacheL1InsertResult(
                admitted=eid in self._entries,
                removed_snapshots=removed_snapshots,
                inserted_evicted_snapshot=inserted_evicted_snapshot,
            )

    def set_trace_config(self, trace_config: Optional[TraceConfig]) -> None:
        self._trace_config = trace_config

    def _log_cache(self, **fields: Any) -> None:
        record_cache_event(self._trace_config, **fields)

    def _prune_dominated_prefixes(
        self,
        snapshot: DFlashPrefixSnapshot,
    ) -> list[tuple[int, DFlashPrefixSnapshot]]:
        incoming = snapshot.token_ids
        doomed_same: list[int] = []
        doomed_cross: list[int] = []
        removed: list[tuple[int, DFlashPrefixSnapshot]] = []
        for eid, existing in self._entries.items():
            if existing.key != snapshot.key:
                continue
            same_kind = existing.kind == snapshot.kind
            if not same_kind and not self._cross_kind_prune:
                continue
            if existing.kind == "prefill" and snapshot.kind == "generation":
                # Generation snapshots are prefix-only; keep prefill for exact prompt reuse.
                continue
            n = len(existing.token_ids)
            if n <= len(incoming) and incoming[:n] == existing.token_ids:
                if same_kind:
                    if n == len(incoming) or not self._preserve_aligned_frontier(
                        existing_len=n,
                        incoming_len=len(incoming),
                        kind=existing.kind,
                    ):
                        doomed_same.append(eid)
                else:
                    doomed_cross.append(eid)
        for eid in doomed_same:
            if eid in self._entries:
                removed.append((eid, self._entries.pop(eid)))
                self._stats["prefix_prunes"] += 1
            if eid in self._lru_order:
                self._lru_order.remove(eid)
        for eid in doomed_cross:
            if eid in self._entries:
                removed.append((eid, self._entries.pop(eid)))
                self._stats["cross_kind_prunes"] += 1
            if eid in self._lru_order:
                self._lru_order.remove(eid)
        return removed

    def _preserve_aligned_frontier(
        self,
        *,
        existing_len: int,
        incoming_len: int,
        kind: str,
    ) -> bool:
        stride = self._frontier_stride
        return (
            kind == "prefill"
            and stride > 0
            and existing_len > 0
            and existing_len < incoming_len
            and existing_len % stride == 0
        )

    def _evict_until_under_budget(self) -> list[tuple[int, DFlashPrefixSnapshot]]:
        evicted: list[tuple[int, DFlashPrefixSnapshot]] = []
        while self._lru_order and (
            len(self._entries) > self._max_entries
            or self._current_bytes() > self._max_bytes
        ):
            byte_pressure = self._current_bytes() > self._max_bytes
            eid = self._lru_order.pop(0)
            if eid in self._entries:
                evicted_snapshot = self._entries.pop(eid)
                self._stats["evictions"] += 1
                if byte_pressure:
                    self._stats["byte_budget_evictions"] += 1
                evicted.append((eid, evicted_snapshot))
        return evicted

    def _incoming_generation_should_yield_to_prefill(
        self,
        snapshot: DFlashPrefixSnapshot,
    ) -> bool:
        if snapshot.kind != "generation":
            return False
        incoming = snapshot.token_ids
        for existing in self._entries.values():
            if existing is snapshot:
                continue
            if existing.key != snapshot.key or existing.kind != "prefill":
                continue
            n = len(existing.token_ids)
            if (
                n > 0
                and n <= len(incoming)
                and incoming[:n] == existing.token_ids
                and existing.nbytes + snapshot.nbytes > self._max_bytes
            ):
                return True
        return False

    def _current_bytes(self) -> int:
        return sum(s.nbytes for s in self._entries.values())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = dict(self._stats)
            out["current_entries"] = len(self._entries)
            out["current_bytes"] = self._current_bytes()
            out["max_entries"] = self._max_entries
            out["max_bytes"] = self._max_bytes
        return out

    def memory_waterfall_bytes(self) -> dict[str, int]:
        out = {
            "l1_snapshot_bytes": 0,
            "l1_snapshot_fa_kv_bytes": 0,
            "l1_snapshot_gdn_state_bytes": 0,
            "l1_snapshot_draft_context_bytes": 0,
            "l1_snapshot_target_hidden_bytes": 0,
            "l1_snapshot_last_logits_bytes": 0,
            "prefix_prunes": 0,
            "cross_kind_prunes": 0,
            "byte_budget_evictions": 0,
        }
        with self._lock:
            stats = dict(self._stats)
            entries = list(self._entries.values())
            current_bytes = self._current_bytes()
        out["l1_snapshot_bytes"] = int(current_bytes)
        out["prefix_prunes"] = int(stats.get("prefix_prunes", 0))
        out["cross_kind_prunes"] = int(stats.get("cross_kind_prunes", 0))
        out["byte_budget_evictions"] = int(stats.get("byte_budget_evictions", 0))
        if entries:
            fa = gdn = draft_context = logits = total = 0
            for snapshot in entries:
                breakdown = snapshot.nbytes_breakdown()
                fa += int(breakdown.get("fa_kv", 0) or 0)
                gdn += int(breakdown.get("gdn_state", 0) or 0)
                draft_context += int(
                    breakdown.get("draft_context", breakdown.get("target_hidden", 0)) or 0
                )
                logits += int(breakdown.get("last_logits", 0) or 0)
                total += int(snapshot.nbytes)
            out["l1_snapshot_bytes"] = total
            out["l1_snapshot_fa_kv_bytes"] = fa
            out["l1_snapshot_gdn_state_bytes"] = gdn
            out["l1_snapshot_draft_context_bytes"] = draft_context
            out["l1_snapshot_target_hidden_bytes"] = draft_context
            out["l1_snapshot_last_logits_bytes"] = logits
        return out

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._lru_order.clear()

    def resident_snapshots(self) -> tuple[DFlashPrefixSnapshot, ...]:
        with self._lock:
            return tuple(
                self._entries[eid] for eid in self._lru_order if eid in self._entries
            )

    def shutdown(self) -> None:
        return None

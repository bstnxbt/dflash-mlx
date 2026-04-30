# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from dflash_mlx.bench_logger import log_cache as _bench_log_cache
from dflash_mlx.diagnostics import TraceConfig
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot

class DFlashPrefixCache:
    def __init__(
        self,
        *,
        max_entries: int = 4,
        max_bytes: int = 8 * 1024 * 1024 * 1024,
        l2: Optional[Any] = None,
        cross_kind_prune: bool = True,
        trace_config: Optional[TraceConfig] = None,
        max_snapshot_tokens: int = 24000,
    ):
        self._entries: dict[int, DFlashPrefixSnapshot] = {}
        self._lru_order: list[int] = []
        self._next_id = 0
        self._max_entries = int(max_entries)
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._l2 = l2
        self._cross_kind_prune = bool(cross_kind_prune)
        self._trace_config = trace_config
        self._max_snapshot_tokens = max(0, int(max_snapshot_tokens))
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
            "l2_hits": 0,
            "l2_misses": 0,
        }

    def lookup(
        self,
        req_tokens: list[int] | tuple[int, ...],
        key: DFlashPrefixKey,
    ) -> tuple[int, Optional[DFlashPrefixSnapshot]]:
        req_tuple = tuple(int(t) for t in req_tokens)
        t_start = time.perf_counter_ns()

        with self._lock:
            best_len = 0
            best_id = -1
            best_snapshot: Optional[DFlashPrefixSnapshot] = None
            saw_fingerprint_reject = 0
            longest_fingerprint_match_len = 0
            longest_fingerprint_first_divergence = -1
            for eid, snap in self._entries.items():
                if snap.key != key:
                    saw_fingerprint_reject += 1
                    continue
                snap_len = len(snap.token_ids)
                if snap_len == 0 or snap_len > len(req_tuple):
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
                    continue
                if snap_len > best_len:
                    best_len = snap_len
                    best_id = eid
                    best_snapshot = snap

            if best_snapshot is not None and best_len > 0:
                exact = best_len == len(req_tuple)
                if exact and (
                    best_snapshot.kind != "prefill"
                    or best_snapshot.last_logits is None
                ):
                    best_snapshot = None
                else:

                    if best_id in self._lru_order:
                        self._lru_order.remove(best_id)
                    self._lru_order.append(best_id)
                    if exact:
                        self._stats["exact_hits"] += 1
                    else:
                        self._stats["prefix_hits"] += 1
                    self._stats["prefill_tokens_saved"] += best_len
                    entries_count_log = len(self._entries)
                    self._log_cache(
                        op="lookup",
                        result="exact_hit" if exact else "prefix_hit",
                        req_tokens=len(req_tuple),
                        matched_len=int(best_len),
                        entries=entries_count_log,
                        elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
                    )
                    return (best_len, best_snapshot)

            self._stats["misses"] += 1
            miss_reason = "empty_cache"
            if self._entries:
                if saw_fingerprint_reject == len(self._entries):
                    miss_reason = "fingerprint_reject_all"
                    self._stats["fingerprint_rejects"] += 1
                elif saw_fingerprint_reject > 0 and longest_fingerprint_match_len == 0:
                    miss_reason = "fingerprint_reject_partial_no_token_match"
                    self._stats["fingerprint_rejects"] += 1
                elif longest_fingerprint_match_len == 0:
                    miss_reason = "no_token_match"
                else:
                    miss_reason = "token_divergence"
            entries_count = len(self._entries)
            sfr = int(saw_fingerprint_reject)
            longest_match = int(longest_fingerprint_match_len)
            first_div = int(longest_fingerprint_first_divergence)

        self._log_cache(
            op="lookup",
            result="miss",
            req_tokens=len(req_tuple),
            matched_len=0,
            entries=entries_count,
            fingerprint_reject=sfr,
            miss_reason=miss_reason,
            longest_token_match_len=longest_match,
            first_divergence_pos=first_div,
            elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
        )

        if self._l2 is None:
            return (0, None)

        l2_snap = self._l2.lookup(req_tuple, key)

        if l2_snap is None:
            with self._lock:
                self._stats["l2_misses"] += 1
            return (0, None)

        l2_len = len(l2_snap.token_ids)
        with self._lock:

            for existing_eid, existing_snap in self._entries.items():
                if (
                    existing_snap.key == l2_snap.key
                    and existing_snap.kind == l2_snap.kind
                    and existing_snap.token_ids == l2_snap.token_ids
                ):
                    if existing_eid in self._lru_order:
                        self._lru_order.remove(existing_eid)
                    self._lru_order.append(existing_eid)
                    self._stats["l2_hits"] += 1
                    if l2_len == len(req_tuple):
                        self._stats["exact_hits"] += 1
                    else:
                        self._stats["prefix_hits"] += 1
                    self._stats["prefill_tokens_saved"] += l2_len
                    self._log_cache(
                        op="lookup",
                        result="l2_hit",
                        req_tokens=len(req_tuple),
                        matched_len=int(l2_len),
                        entries=len(self._entries),
                        elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
                    )
                    return (l2_len, existing_snap)

            eid = self._next_id
            self._next_id += 1
            self._entries[eid] = l2_snap
            self._lru_order.append(eid)
            self._evict_until_under_budget()
            self._stats["l2_hits"] += 1
            if l2_len == len(req_tuple):
                self._stats["exact_hits"] += 1
            else:
                self._stats["prefix_hits"] += 1
            self._stats["prefill_tokens_saved"] += l2_len
            self._log_cache(
                op="lookup",
                result="l2_hit",
                req_tokens=len(req_tuple),
                matched_len=int(l2_len),
                entries=len(self._entries),
                elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
            )
            return (l2_len, l2_snap)

    def insert(self, snapshot: DFlashPrefixSnapshot) -> None:
        t_start = time.perf_counter_ns()
        with self._lock:
            pre_entries = len(self._entries)
            pre_evictions = self._stats["evictions"]
            pre_byte_evictions = self._stats["byte_budget_evictions"]
            pre_prunes = self._stats["prefix_prunes"]
            pre_cross_prunes = self._stats["cross_kind_prunes"]

            max_tokens = self._max_snapshot_tokens
            if self._l2 is None and max_tokens > 0 and len(snapshot.token_ids) > max_tokens:
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
                return

            self._prune_dominated_prefixes(snapshot)
            eid = self._next_id
            self._next_id += 1
            self._entries[eid] = snapshot
            self._lru_order.append(eid)
            self._stats["insertions"] += 1
            self._evict_until_under_budget()
            breakdown = snapshot.nbytes_breakdown()
            self._log_cache(
                op="insert",
                kind=snapshot.kind,
                prefix_len=int(snapshot.prefix_len),
                nbytes=int(snapshot.nbytes),
                bytes_fa_kv=int(breakdown.get("fa_kv", 0)),
                bytes_gdn_state=int(breakdown.get("gdn_state", 0)),
                bytes_target_hidden=int(breakdown.get("target_hidden", 0)),
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

    def set_trace_config(self, trace_config: Optional[TraceConfig]) -> None:
        self._trace_config = trace_config

    def _log_cache(self, **fields: Any) -> None:
        _bench_log_cache(self._trace_config, **fields)

    def _prune_dominated_prefixes(self, snapshot: DFlashPrefixSnapshot) -> None:
        incoming = snapshot.token_ids
        doomed_same: list[int] = []
        doomed_cross: list[int] = []
        for eid, existing in self._entries.items():
            if existing.key != snapshot.key:
                continue
            same_kind = existing.kind == snapshot.kind
            if not same_kind and not self._cross_kind_prune:
                continue
            n = len(existing.token_ids)
            if n <= len(incoming) and incoming[:n] == existing.token_ids:
                if same_kind:
                    doomed_same.append(eid)
                else:
                    doomed_cross.append(eid)
        for eid in doomed_same:
            if eid in self._entries:
                del self._entries[eid]
                self._stats["prefix_prunes"] += 1
            if eid in self._lru_order:
                self._lru_order.remove(eid)
        for eid in doomed_cross:
            if eid in self._entries:
                del self._entries[eid]
                self._stats["cross_kind_prunes"] += 1
            if eid in self._lru_order:
                self._lru_order.remove(eid)

    def _evict_until_under_budget(self) -> None:
        while self._lru_order and (
            len(self._entries) > self._max_entries
            or self._current_bytes() > self._max_bytes
        ):
            byte_pressure = self._current_bytes() > self._max_bytes
            eid = self._lru_order.pop(0)
            if eid in self._entries:
                evicted = self._entries.pop(eid)
                self._stats["evictions"] += 1
                if byte_pressure:
                    self._stats["byte_budget_evictions"] += 1
                if self._l2 is not None:
                    self._l2.insert_async(evicted)

    def _current_bytes(self) -> int:
        return sum(s.nbytes for s in self._entries.values())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = dict(self._stats)
            out["current_entries"] = len(self._entries)
            out["current_bytes"] = self._current_bytes()
            out["max_entries"] = self._max_entries
            out["max_bytes"] = self._max_bytes
        if self._l2 is not None:
            out["l2"] = self._l2.stats()
        return out

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._lru_order.clear()
        if self._l2 is not None:
            self._l2.clear()

    def shutdown(self) -> None:
        if self._l2 is not None:
            self._l2.shutdown(wait=True)

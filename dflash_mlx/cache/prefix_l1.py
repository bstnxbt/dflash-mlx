from __future__ import annotations

import threading
import time
from typing import Any, Optional

from dflash_mlx.bench_logger import log_cache as _bench_log_cache
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot


class DFlashPrefixCache:
    def __init__(
        self,
        *,
        max_entries: int = 4,
        max_bytes: int = 8 * 1024 * 1024 * 1024,
    ):
        self._entries: dict[int, DFlashPrefixSnapshot] = {}
        self._lru_order: list[int] = []
        self._next_id = 0
        self._max_entries = int(max_entries)
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._stats: dict[str, int] = {
            "exact_hits": 0,
            "prefix_hits": 0,
            "misses": 0,
            "insertions": 0,
            "evictions": 0,
            "prefix_prunes": 0,
            "prefill_tokens_saved": 0,
            "fingerprint_rejects": 0,
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
            saw_fingerprint_reject = False
            for eid, snap in self._entries.items():
                if snap.key != key:
                    saw_fingerprint_reject = True
                    continue
                snap_len = len(snap.token_ids)

                if snap_len == 0 or snap_len > len(req_tuple):
                    continue
                if req_tuple[:snap_len] != snap.token_ids:
                    continue
                if snap_len > best_len:
                    best_len = snap_len
                    best_id = eid
                    best_snapshot = snap

            if best_snapshot is None or best_len == 0:
                self._stats["misses"] += 1
                if saw_fingerprint_reject and self._entries:
                    self._stats["fingerprint_rejects"] += 1
                _bench_log_cache(
                    op="lookup",
                    result="miss",
                    req_tokens=len(req_tuple),
                    matched_len=0,
                    entries=len(self._entries),
                    fingerprint_reject=bool(saw_fingerprint_reject),
                    elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
                )
                return (0, None)

            if best_id in self._lru_order:
                self._lru_order.remove(best_id)
            self._lru_order.append(best_id)

            exact = best_len == len(req_tuple)
            if exact:
                self._stats["exact_hits"] += 1
            else:
                self._stats["prefix_hits"] += 1
            self._stats["prefill_tokens_saved"] += best_len
            _bench_log_cache(
                op="lookup",
                result="exact_hit" if exact else "prefix_hit",
                req_tokens=len(req_tuple),
                matched_len=int(best_len),
                entries=len(self._entries),
                elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
            )
            return (best_len, best_snapshot)

    def insert(self, snapshot: DFlashPrefixSnapshot) -> None:
        t_start = time.perf_counter_ns()
        with self._lock:
            pre_entries = len(self._entries)
            pre_evictions = self._stats["evictions"]
            pre_prunes = self._stats["prefix_prunes"]
            self._prune_dominated_prefixes(snapshot)
            eid = self._next_id
            self._next_id += 1
            self._entries[eid] = snapshot
            self._lru_order.append(eid)
            self._stats["insertions"] += 1
            self._evict_until_under_budget()
            _bench_log_cache(
                op="insert",
                kind=snapshot.kind,
                prefix_len=int(snapshot.prefix_len),
                nbytes=int(snapshot.nbytes),
                entries_before=pre_entries,
                entries_after=len(self._entries),
                pruned=int(self._stats["prefix_prunes"] - pre_prunes),
                evicted=int(self._stats["evictions"] - pre_evictions),
                elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
            )

    def _prune_dominated_prefixes(self, snapshot: DFlashPrefixSnapshot) -> None:
        incoming = snapshot.token_ids
        doomed: list[int] = []
        for eid, existing in self._entries.items():
            if existing.key != snapshot.key:
                continue
            if existing.kind != snapshot.kind:
                continue
            n = len(existing.token_ids)
            if n <= len(incoming) and incoming[:n] == existing.token_ids:
                doomed.append(eid)
        for eid in doomed:
            if eid in self._entries:
                del self._entries[eid]
                self._stats["prefix_prunes"] += 1
            if eid in self._lru_order:
                self._lru_order.remove(eid)

    def _evict_until_under_budget(self) -> None:
        while self._lru_order and (
            len(self._entries) > self._max_entries
            or self._current_bytes() > self._max_bytes
        ):
            eid = self._lru_order.pop(0)
            if eid in self._entries:
                del self._entries[eid]
                self._stats["evictions"] += 1

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

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._lru_order.clear()

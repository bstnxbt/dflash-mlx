# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import threading
import time
from typing import Any

from dflash_mlx.diagnostics import TraceConfig
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.prefix_l2 import DFlashPrefixL2Cache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.observability.cache import record_cache_event


class PrefixSnapshotStore:
    def __init__(
        self,
        *,
        l1: DFlashPrefixCache,
        l2: DFlashPrefixL2Cache | None = None,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._lock = threading.Lock()
        self._stats: dict[str, int] = {
            "l2_hits": 0,
            "l2_misses": 0,
            "l2_exact_hits": 0,
            "l2_prefix_hits": 0,
            "l2_prefill_tokens_saved": 0,
        }
        self._trace_config: TraceConfig | None = None
        # Updated on every lookup to reflect the source of the last hit.
        # L1 path: propagated from self._l1.last_hit_kind.
        # L2 path: set directly in _record_l2_hit().
        # Readable by RuntimeCacheManager while holding _state_lock.
        self._last_hit_kind: str = "miss"

    def set_trace_config(self, trace_config: TraceConfig | None) -> None:
        self._trace_config = trace_config
        self._l1.set_trace_config(trace_config)

    def frontier_stride(self) -> int:
        if self._l2 is None:
            return 0
        return int(self._l1.frontier_stride)

    def lookup(
        self,
        req_tokens: list[int] | tuple[int, ...],
        key: DFlashPrefixKey,
        *,
        request_id: int | None = None,
    ) -> tuple[int, DFlashPrefixSnapshot | None]:
        req_tuple = tuple(int(t) for t in req_tokens)
        t_start = time.perf_counter_ns()
        if self._l2 is None:
            return self._l1_lookup(req_tuple, key, request_id=request_id)

        matched_len, snapshot = self._l1_lookup(
            req_tuple,
            key,
            record=False,
            request_id=request_id,
        )
        if snapshot is not None or matched_len > 0:
            if matched_len == len(req_tuple):
                return self._l1_lookup(req_tuple, key, request_id=request_id)
            l2_snapshot = self._l2.lookup(
                req_tuple,
                key,
                min_token_len=matched_len,
            )
            if l2_snapshot is not None:
                return self._record_l2_hit(
                    l2_snapshot,
                    req_tuple=req_tuple,
                    t_start=t_start,
                    request_id=request_id,
                )
            else:
                with self._lock:
                    self._stats["l2_misses"] += 1
            return self._l1_lookup(req_tuple, key, request_id=request_id)

        l2_snapshot = self._l2.lookup(req_tuple, key)
        if l2_snapshot is None:
            with self._lock:
                self._stats["l2_misses"] += 1
            return self._l1_lookup(req_tuple, key, request_id=request_id)

        return self._record_l2_hit(
            l2_snapshot,
            req_tuple=req_tuple,
            t_start=t_start,
            request_id=request_id,
        )

    def _record_l2_hit(
        self,
        l2_snapshot: DFlashPrefixSnapshot,
        *,
        req_tuple: tuple[int, ...],
        t_start: int,
        request_id: int | None = None,
    ) -> tuple[int, DFlashPrefixSnapshot]:
        l2_len = len(l2_snapshot.token_ids)
        exact = l2_len == len(req_tuple)
        promote = self._l1.insert_with_evictions(l2_snapshot, skip_too_long=True)
        self._write_snapshots_to_l2(promote.removed_snapshots)
        with self._lock:
            self._stats["l2_hits"] += 1
            if exact:
                self._stats["l2_exact_hits"] += 1
            else:
                self._stats["l2_prefix_hits"] += 1
            self._stats["l2_prefill_tokens_saved"] += int(l2_len)
        self._last_hit_kind = "l2_exact" if exact else "l2_prefix"
        record_cache_event(
            self._trace_config,
            op="lookup",
            result="l2_hit",
            hit_kind=self._last_hit_kind,
            request_id=request_id,
            req_tokens=len(req_tuple),
            matched_len=int(l2_len),
            entries=int(self._l1.stats().get("current_entries", 0)),
            elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
        )
        return l2_len, l2_snapshot

    def _l1_lookup(
        self,
        req_tuple: tuple[int, ...],
        key: DFlashPrefixKey,
        *,
        record: bool = True,
        request_id: int | None = None,
    ) -> tuple[int, DFlashPrefixSnapshot | None]:
        if request_id is None and record:
            result = self._l1.lookup(req_tuple, key)
        elif request_id is None:
            result = self._l1.lookup(req_tuple, key, record=record)
        else:
            result = self._l1.lookup(req_tuple, key, record=record, request_id=request_id)
        if record:
            self._last_hit_kind = self._l1.last_hit_kind
        return result

    def insert(self, snapshot: DFlashPrefixSnapshot) -> bool:
        inserted_l2_admitted = False
        result = self._l1.insert_with_evictions(
            snapshot,
            skip_too_long=True,
        )
        if self._l2 is not None:
            if result.admitted and snapshot.kind == "prefill":
                self._write_snapshots_to_l2((snapshot,))
            if result.skipped_too_long and snapshot.kind == "prefill":
                inserted_l2_admitted = self._write_snapshots_to_l2(
                    (snapshot,),
                    inserted_snapshot=snapshot,
                )
            inserted_l2_admitted = self._write_snapshots_to_l2(
                result.removed_snapshots,
                inserted_snapshot=result.inserted_evicted_snapshot,
            ) or inserted_l2_admitted
        return bool(result.admitted or inserted_l2_admitted)

    def insert_l2_only(self, snapshot: DFlashPrefixSnapshot) -> bool:
        if self._l2 is None:
            return False
        return self._write_snapshots_to_l2(
            (snapshot,),
            inserted_snapshot=snapshot,
        )

    def stats(self) -> dict[str, Any]:
        out = self._l1.stats()
        with self._lock:
            stats = dict(self._stats)
        out["l2_hits"] = int(stats["l2_hits"])
        out["l2_misses"] = int(stats["l2_misses"])
        out["exact_hits"] = int(out.get("exact_hits", 0)) + int(stats["l2_exact_hits"])
        out["prefix_hits"] = int(out.get("prefix_hits", 0)) + int(stats["l2_prefix_hits"])
        out["prefill_tokens_saved"] = int(out.get("prefill_tokens_saved", 0)) + int(
            stats["l2_prefill_tokens_saved"]
        )
        if self._l2 is not None:
            out["l2"] = self._l2.stats()
        return out

    def memory_waterfall_bytes(self) -> dict[str, int]:
        out = self._l1.memory_waterfall_bytes()
        out.setdefault("l2_disk_bytes", 0)
        out.setdefault("l2_hits", 0)
        out.setdefault("l2_writes", 0)
        out.setdefault("l2_misses", 0)
        with self._lock:
            out["l2_hits"] = int(self._stats["l2_hits"])
            out["l2_misses"] = int(self._stats["l2_misses"])
        if self._l2 is not None:
            l2 = self._l2.stats()
            out["l2_disk_bytes"] = int(l2.get("current_bytes", 0) or 0)
            out["l2_writes"] = int(l2.get("writes", 0) or 0)
        return out

    def clear(self) -> None:
        self._l1.clear()
        if self._l2 is not None:
            self._l2.clear()

    def begin_request(self) -> None:
        if self._l2 is not None:
            self._l2.begin_request()

    def end_request(self) -> None:
        if self._l2 is not None:
            self._l2.end_request()

    def shutdown(self) -> None:
        self._l1.shutdown()
        if self._l2 is not None:
            self._l2.shutdown(wait=True)

    def _write_snapshots_to_l2(
        self,
        snapshots: tuple[DFlashPrefixSnapshot, ...],
        *,
        inserted_snapshot: DFlashPrefixSnapshot | None = None,
    ) -> bool:
        if self._l2 is None:
            return False
        inserted_admitted = False
        for snapshot in snapshots:
            admitted = self._l2.insert_async(snapshot)
            if snapshot is inserted_snapshot:
                inserted_admitted = bool(admitted)
        return inserted_admitted

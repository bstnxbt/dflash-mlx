# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

HitKind = Literal["miss", "l1_exact", "l1_prefix", "l2_exact", "l2_prefix"]

from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.prefix_l2 import DFlashPrefixL2Cache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.cache.store import PrefixSnapshotStore

_DFLASH_RUNTIME_CACHE_MANAGER: Optional["RuntimeCacheManager"] = None
_DFLASH_RUNTIME_CACHE_CONFIG_KEY: Optional[tuple[Any, ...]] = None
_DFLASH_RUNTIME_CACHE_LOCK = threading.Lock()
_MIN_L2_FRONTIER_STRIDE = 8192


@dataclass(frozen=True)
class PrefixCacheLookupResult:
    matched_tokens: int
    snapshot: Optional[DFlashPrefixSnapshot]
    elapsed_ms: float
    hit_kind: HitKind = "miss"


@dataclass(frozen=True)
class PrefixCacheInsertResult:
    admitted: bool
    elapsed_ms: float


class RuntimeCacheManagerClosed(RuntimeError):
    pass


class RuntimeCacheManager:
    def __init__(self, store: PrefixSnapshotStore) -> None:
        self._store = store
        self._state_lock = threading.RLock()
        self._retired = False
        self._shutdown_complete = False

    def set_trace_config(self, trace_config: Any) -> None:
        with self._state_lock:
            self._ensure_open_locked()
            self._store.set_trace_config(trace_config)

    def shutdown(self) -> None:
        with self._state_lock:
            if self._shutdown_complete:
                return
            self._retired = True
            self._store.shutdown()
            self._shutdown_complete = True

    @property
    def active(self) -> bool:
        with self._state_lock:
            return not self._retired

    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            self._ensure_open_locked()
            return self._store.stats()

    def memory_waterfall_bytes(self) -> dict[str, int]:
        with self._state_lock:
            self._ensure_open_locked()
            return self._store.memory_waterfall_bytes()

    def frontier_stride(self) -> int:
        with self._state_lock:
            self._ensure_open_locked()
            return self._store.frontier_stride()

    def lookup(
        self,
        tokens: list[int] | tuple[int, ...],
        key: DFlashPrefixKey,
        *,
        request_id: int | None = None,
    ) -> PrefixCacheLookupResult:
        lookup_t0 = time.perf_counter_ns()
        with self._state_lock:
            self._ensure_open_locked()
            matched_len, snapshot = self._store.lookup(
                tokens,
                key,
                request_id=request_id,
            )
            hit_kind: HitKind = self._store._last_hit_kind  # type: ignore[assignment]
        return PrefixCacheLookupResult(
            matched_tokens=int(matched_len),
            snapshot=snapshot,
            elapsed_ms=(time.perf_counter_ns() - lookup_t0) / 1e6,
            hit_kind=hit_kind,
        )

    def maybe_insert_snapshot(
        self,
        snapshot: Any,
        *,
        key: DFlashPrefixKey,
        kind: str,
        require_logits: bool,
    ) -> PrefixCacheInsertResult:
        snapshot = self._validate_snapshot_contract(
            snapshot,
            key=key,
            kind=kind,
            require_logits=require_logits,
        )
        insert_t0 = time.perf_counter_ns()
        try:
            with self._state_lock:
                self._ensure_open_locked()
                admitted = self._store.insert(snapshot)
        except RuntimeCacheManagerClosed:
            raise
        except Exception as exc:
            _log_insert_failure(kind, exc)
            raise
        elapsed_ms = (time.perf_counter_ns() - insert_t0) / 1e6
        if admitted and kind == "generation":
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"[dflash] end-of-request snapshot saved "
                f"({snapshot.prefix_len} tokens)\n"
            )
            sys.stderr.flush()
        return PrefixCacheInsertResult(
            admitted=bool(admitted),
            elapsed_ms=float(elapsed_ms),
        )

    def maybe_insert_snapshot_l2_only(
        self,
        snapshot: Any,
        *,
        key: DFlashPrefixKey,
        kind: str,
        require_logits: bool,
    ) -> PrefixCacheInsertResult:
        snapshot = self._validate_snapshot_contract(
            snapshot,
            key=key,
            kind=kind,
            require_logits=require_logits,
        )
        insert_t0 = time.perf_counter_ns()
        try:
            with self._state_lock:
                self._ensure_open_locked()
                admitted = self._store.insert_l2_only(snapshot)
        except RuntimeCacheManagerClosed:
            raise
        except Exception as exc:
            _log_insert_failure(kind, exc)
            raise
        return PrefixCacheInsertResult(
            admitted=bool(admitted),
            elapsed_ms=(time.perf_counter_ns() - insert_t0) / 1e6,
        )

    def _validate_snapshot_contract(
        self,
        snapshot: Any,
        *,
        key: DFlashPrefixKey,
        kind: str,
        require_logits: bool,
    ) -> DFlashPrefixSnapshot:
        if not isinstance(snapshot, DFlashPrefixSnapshot):
            raise TypeError(f"expected DFlashPrefixSnapshot, got {type(snapshot).__name__}")
        if snapshot.key != key:
            raise ValueError("prefix snapshot key does not match request key")
        if snapshot.kind != kind:
            raise ValueError(f"expected {kind!r} prefix snapshot, got {snapshot.kind!r}")
        if require_logits and snapshot.last_logits is None:
            raise ValueError("prefix snapshot requires last_logits")
        return snapshot

    def log_stats(self, label: str = "") -> None:
        with self._state_lock:
            self._ensure_open_locked()
            _format_stats_line(self._store, label)

    def _ensure_open_locked(self) -> None:
        if self._retired:
            raise RuntimeCacheManagerClosed("runtime cache manager is shut down")

    def _is_retired(self) -> bool:
        with self._state_lock:
            return self._retired


def get_runtime_cache_manager(
    runtime_context: Optional[Any] = None,
    *,
    cache_identity: Any = None,
) -> Optional[RuntimeCacheManager]:
    global _DFLASH_RUNTIME_CACHE_CONFIG_KEY, _DFLASH_RUNTIME_CACHE_MANAGER
    if runtime_context is None:
        return None
    if _runtime_cache_disabled(runtime_context):
        _clear_runtime_cache_manager()
        return None
    config_key = _prefix_cache_config_key(runtime_context, cache_identity=cache_identity)
    trace_config = runtime_context.diagnostics.trace
    with _DFLASH_RUNTIME_CACHE_LOCK:
        manager = _DFLASH_RUNTIME_CACHE_MANAGER
        if (
            manager is not None
            and _DFLASH_RUNTIME_CACHE_CONFIG_KEY == config_key
            and not manager._is_retired()
        ):
            manager.set_trace_config(trace_config)
            return manager
        if manager is not None:
            _shutdown_manager(manager)
            _DFLASH_RUNTIME_CACHE_MANAGER = None
            _DFLASH_RUNTIME_CACHE_CONFIG_KEY = None
        manager = RuntimeCacheManager(_make_prefix_store(runtime_context))
        _DFLASH_RUNTIME_CACHE_MANAGER = manager
        _DFLASH_RUNTIME_CACHE_CONFIG_KEY = config_key
        return manager


def sync_runtime_cache_manager(
    runtime_context: Optional[Any] = None,
    *,
    cache_identity: Any = None,
) -> Optional[RuntimeCacheManager]:
    global _DFLASH_RUNTIME_CACHE_CONFIG_KEY, _DFLASH_RUNTIME_CACHE_MANAGER
    if runtime_context is None:
        return None
    if _runtime_cache_disabled(runtime_context):
        _clear_runtime_cache_manager()
        return None
    config_key = _prefix_cache_config_key(runtime_context, cache_identity=cache_identity)
    with _DFLASH_RUNTIME_CACHE_LOCK:
        manager = _DFLASH_RUNTIME_CACHE_MANAGER
        if manager is None:
            return None
        if _DFLASH_RUNTIME_CACHE_CONFIG_KEY != config_key or manager._is_retired():
            _shutdown_manager(manager)
            _DFLASH_RUNTIME_CACHE_MANAGER = None
            _DFLASH_RUNTIME_CACHE_CONFIG_KEY = None
            return None
        manager.set_trace_config(runtime_context.diagnostics.trace)
        return manager


def current_runtime_cache_manager() -> Optional[RuntimeCacheManager]:
    with _DFLASH_RUNTIME_CACHE_LOCK:
        manager = _DFLASH_RUNTIME_CACHE_MANAGER
        if manager is None or manager._is_retired():
            return None
        return manager


def shutdown_runtime_cache_manager() -> None:
    _clear_runtime_cache_manager(raise_on_error=False)


def _clear_runtime_cache_manager(*, raise_on_error: bool = True) -> None:
    global _DFLASH_RUNTIME_CACHE_CONFIG_KEY, _DFLASH_RUNTIME_CACHE_MANAGER
    with _DFLASH_RUNTIME_CACHE_LOCK:
        manager = _DFLASH_RUNTIME_CACHE_MANAGER
        if _shutdown_manager(manager, raise_on_error=raise_on_error):
            _DFLASH_RUNTIME_CACHE_MANAGER = None
            _DFLASH_RUNTIME_CACHE_CONFIG_KEY = None


def _prefix_cache_config_key(
    runtime_context: Any,
    *,
    cache_identity: Any = None,
) -> tuple[Any, ...]:
    runtime_config = runtime_context.runtime
    return (
        cache_identity,
        int(runtime_config.prefix_cache_max_entries),
        int(runtime_config.prefix_cache_max_bytes),
        int(runtime_config.max_snapshot_tokens),
        int(_prefix_cache_frontier_stride(runtime_config)),
        bool(runtime_config.prefix_cache_l2),
        str(runtime_config.prefix_cache_l2_dir),
        int(runtime_config.prefix_cache_l2_max_bytes),
    )


def _runtime_cache_disabled(runtime_context: Any) -> bool:
    runtime_config = runtime_context.runtime
    return bool(runtime_config.target_fa_window > 0 or not runtime_config.prefix_cache)


def _prefix_cache_frontier_stride(runtime_config: Any) -> int:
    if not getattr(runtime_config, "prefix_cache_l2", False):
        return 0
    prefill_step_size = max(0, int(getattr(runtime_config, "prefill_step_size", 0) or 0))
    if prefill_step_size <= 0:
        return 0
    steps = max(1, (_MIN_L2_FRONTIER_STRIDE + prefill_step_size - 1) // prefill_step_size)
    return int(steps * prefill_step_size)


def _shutdown_manager(
    manager: Optional[RuntimeCacheManager],
    *,
    raise_on_error: bool = True,
) -> bool:
    if manager is None:
        return True
    try:
        manager.shutdown()
        return True
    except Exception as exc:
        sys.stderr.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] runtime cache manager shutdown failed: {exc}\n"
        )
        sys.stderr.flush()
        if raise_on_error:
            raise
        return False


def _make_prefix_store(runtime_context: Any) -> PrefixSnapshotStore:
    runtime_config = runtime_context.runtime
    max_entries = int(runtime_config.prefix_cache_max_entries)
    max_bytes = int(runtime_config.prefix_cache_max_bytes)
    l2: Optional[DFlashPrefixL2Cache] = None
    if runtime_config.prefix_cache_l2:
        try:
            l2 = DFlashPrefixL2Cache(
                cache_dir=runtime_config.prefix_cache_l2_dir,
                max_bytes=runtime_config.prefix_cache_l2_max_bytes,
            )
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix L2 cache enabled "
                f"(dir={l2.cache_dir}, max_bytes={runtime_config.prefix_cache_l2_max_bytes}, "
                f"writable={l2.writable})\n"
            )
        except OSError as exc:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix L2 cache disabled: {exc}\n"
            )
            l2 = None
    trace_config = runtime_context.diagnostics.trace if runtime_context is not None else None
    l1 = DFlashPrefixCache(
        max_entries=max_entries,
        max_bytes=max_bytes,
        trace_config=trace_config,
        max_snapshot_tokens=runtime_config.max_snapshot_tokens,
        frontier_stride=(
            _prefix_cache_frontier_stride(runtime_config)
            if l2 is not None
            else 0
        ),
    )
    sys.stderr.write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix cache enabled "
        f"(max_entries={max_entries}, max_bytes={max_bytes})\n"
    )
    sys.stderr.flush()
    store = PrefixSnapshotStore(l1=l1, l2=l2)
    store.set_trace_config(trace_config)
    return store


def _format_stats_line(store: PrefixSnapshotStore, label: str = "") -> None:
    stats = store.stats()
    prefix = f" [{label}]" if label else ""
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix-cache-stats{prefix} "
        f"entries={stats['current_entries']}/{stats['max_entries']} "
        f"bytes={stats['current_bytes']}/{stats['max_bytes']} "
        f"hits={stats['exact_hits']}+{stats['prefix_hits']} "
        f"misses={stats['misses']} "
        f"insertions={stats['insertions']} "
        f"evictions={stats['evictions']} "
        f"prefill_tokens_saved={stats['prefill_tokens_saved']}"
    )
    l2 = stats.get("l2")
    if l2:
        line += (
            f" l2_hits={stats.get('l2_hits', 0)} l2_misses={stats.get('l2_misses', 0)} "
            f"l2_writes={l2.get('writes', 0)} l2_bytes={l2.get('current_bytes', 0)}/{l2.get('max_bytes', 0)}"
        )
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _log_insert_failure(kind: str, exc: Exception) -> None:
    if kind == "prefill":
        msg = f"[dflash] prefix cache insert failed: {exc}"
    else:
        msg = f"[dflash] end-of-request snapshot failed: {exc}"
    sys.stderr.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    sys.stderr.flush()

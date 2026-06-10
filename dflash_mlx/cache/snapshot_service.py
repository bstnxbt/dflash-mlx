# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx

from dflash_mlx.cache.codecs import PrefixSnapshotBuilder
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.manager import RuntimeCacheManager, RuntimeCacheManagerClosed
from dflash_mlx.cache.snapshot import TargetHiddenChunks


@dataclass(frozen=True)
class SnapshotPublication:
    kind: Literal["prefill", "generation"]
    snapshot_boundary: int
    prefix_len: int
    insert_ms: float
    admitted: bool
    from_snapshot: bool = False
    snap_prefix_len: int = 0


class SnapshotService:
    def __init__(
        self,
        *,
        cache_manager: RuntimeCacheManager,
        builder: PrefixSnapshotBuilder,
    ) -> None:
        self._cache_manager: RuntimeCacheManager | None = cache_manager
        self._builder = builder
        self._insert_ms = 0.0

    @classmethod
    def from_request(
        cls,
        *,
        cache_manager: RuntimeCacheManager,
        key: DFlashPrefixKey,
        draft_model: Any,
        runtime_context: Any,
    ) -> "SnapshotService":
        runtime_config = runtime_context.runtime
        return cls(
            cache_manager=cache_manager,
            builder=PrefixSnapshotBuilder(
                key=key,
                draft_model=draft_model,
                draft_sink_size=int(runtime_config.draft_sink_size),
                draft_window_size=int(runtime_config.draft_window_size),
            ),
        )

    @property
    def insert_ms(self) -> float:
        return float(self._insert_ms)

    @property
    def active(self) -> bool:
        cache_manager = self._cache_manager
        if cache_manager is None:
            return False
        if not cache_manager.active:
            self._cache_manager = None
            return False
        return True

    def should_publish_frontier(self, prefix_len: int) -> bool:
        cache_manager = self._cache_manager
        if cache_manager is None:
            return False
        try:
            stride = cache_manager.frontier_stride()
        except RuntimeCacheManagerClosed:
            self._cache_manager = None
            return False
        return stride > 0 and int(prefix_len) > 0 and int(prefix_len) % stride == 0

    def publish(
        self,
        *,
        token_ids: list[int],
        target_cache: list[Any],
        target_hidden: mx.array | TargetHiddenChunks | None,
        last_logits: mx.array | None,
        kind: Literal["prefill", "generation"],
        require_logits: bool,
        snapshot_boundary: int,
        allow_full_attention_context: bool,
        from_snapshot: bool = False,
        snap_prefix_len: int = 0,
        l2_only: bool = False,
        adopt_cache_arrays: bool = False,
    ) -> SnapshotPublication | None:
        if target_hidden is None:
            return None
        if require_logits and last_logits is None:
            raise ValueError(f"{kind} snapshot requires last_logits")

        snapshot = self._builder.build(
            token_ids=token_ids,
            target_cache=target_cache,
            target_hidden=target_hidden,
            last_logits=last_logits,
            kind=kind,
            allow_full_attention_context=allow_full_attention_context,
            adopt_cache_arrays=adopt_cache_arrays,
        )
        cache_manager = self._cache_manager
        if cache_manager is None:
            return SnapshotPublication(
                kind=kind,
                snapshot_boundary=int(snapshot_boundary),
                prefix_len=int(snapshot.prefix_len),
                insert_ms=0.0,
                admitted=False,
                from_snapshot=bool(from_snapshot),
                snap_prefix_len=int(snap_prefix_len),
            )
        try:
            insert = (
                cache_manager.maybe_insert_snapshot_l2_only
                if l2_only
                else cache_manager.maybe_insert_snapshot
            )
            insert_result = insert(
                snapshot,
                key=self._builder.key,
                kind=kind,
                require_logits=require_logits,
            )
        except RuntimeCacheManagerClosed:
            self._cache_manager = None
            return SnapshotPublication(
                kind=kind,
                snapshot_boundary=int(snapshot_boundary),
                prefix_len=int(snapshot.prefix_len),
                insert_ms=0.0,
                admitted=False,
                from_snapshot=bool(from_snapshot),
                snap_prefix_len=int(snap_prefix_len),
            )
        self._insert_ms += float(insert_result.elapsed_ms)
        return SnapshotPublication(
            kind=kind,
            snapshot_boundary=int(snapshot_boundary),
            prefix_len=int(snapshot.prefix_len),
            insert_ms=float(insert_result.elapsed_ms),
            admitted=bool(insert_result.admitted),
            from_snapshot=bool(from_snapshot),
            snap_prefix_len=int(snap_prefix_len),
        )

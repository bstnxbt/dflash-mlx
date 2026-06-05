# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from dflash_mlx.cache.manager import (
    RuntimeCacheManagerClosed,
    RuntimeCacheManager,
    get_runtime_cache_manager,
)
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.cache.snapshot_service import SnapshotService
from dflash_mlx.server.prefix_cache_manager import (
    build_prefix_key,
    chat_template_stable_marker,
)

def compute_stable_prefix_len(
    tokens: list[int] | tuple[int, ...],
    *,
    im_start_id: Optional[int] = None,
    assistant_id: Optional[int] = None,
    boundary_offset: int = 0,
) -> int:
    if im_start_id is None or assistant_id is None:
        return len(tokens)
    n = len(tokens)
    if n < 2:
        return n
    offset = max(0, int(boundary_offset))
    for i in range(n - 2, -1, -1):
        if tokens[i] == im_start_id and tokens[i + 1] == assistant_id:
            return min(n, i + offset)
    return n


def compute_request_stable_prefix_len(
    tokens: list[int] | tuple[int, ...],
    *,
    tokenizer: Any,
    request: Any = None,
) -> int:
    im_start_id, assistant_id, boundary_offset = chat_template_stable_marker(tokenizer)
    role = _last_chat_role(request)
    if role is not None and role not in {"user", "tool"}:
        return len(tokens)
    return compute_stable_prefix_len(
        tokens,
        im_start_id=im_start_id,
        assistant_id=assistant_id,
        boundary_offset=boundary_offset,
    )

def _last_chat_role(request: Any) -> str | None:
    if getattr(request, "request_type", None) != "chat":
        return None
    messages = getattr(request, "messages", None)
    if not isinstance(messages, (list, tuple)) or not messages:
        return None
    last = messages[-1]
    role: Any
    if isinstance(last, dict):
        role = last.get("role")
    else:
        role = getattr(last, "role", None)
    if role is None:
        return None
    return str(role)

@dataclass(frozen=True)
class PrefixCacheLookupStats:
    prompt_tokens: int = 0
    stable_prefix_tokens: int = 0
    lookup_tokens: int = 0
    hit_tokens: int = 0
    lookup_ms: float = 0.0
    hit_kind: str = "inactive"
    snapshot_tokens: int = 0
    snapshot_kind: Optional[str] = None
    snapshot_nbytes: int = 0
    snapshot_has_last_logits: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "stable_prefix_tokens": int(self.stable_prefix_tokens),
            "lookup_tokens": int(self.lookup_tokens),
            "hit_tokens": int(self.hit_tokens),
            "lookup_ms": float(self.lookup_ms),
            "hit_kind": self.hit_kind,
            "snapshot_tokens": int(self.snapshot_tokens),
            "snapshot_kind": self.snapshot_kind,
            "snapshot_nbytes": int(self.snapshot_nbytes),
            "snapshot_has_last_logits": bool(self.snapshot_has_last_logits),
        }


def _lookup_stats(
    *,
    prompt: list[int],
    stable_prefix_len: int,
    lookup_ms: float,
    hit_tokens: int,
    snapshot: Optional[DFlashPrefixSnapshot],
) -> PrefixCacheLookupStats:
    lookup_tokens = min(len(prompt), stable_prefix_len)
    if hit_tokens <= 0:
        hit_kind = "miss"
    elif hit_tokens == lookup_tokens:
        hit_kind = "exact"
    else:
        hit_kind = "prefix"
    return PrefixCacheLookupStats(
        prompt_tokens=len(prompt),
        stable_prefix_tokens=stable_prefix_len,
        lookup_tokens=lookup_tokens,
        hit_tokens=hit_tokens,
        lookup_ms=lookup_ms,
        hit_kind=hit_kind,
        snapshot_tokens=0 if snapshot is None else snapshot.prefix_len,
        snapshot_kind=None if snapshot is None else snapshot.kind,
        snapshot_nbytes=0 if snapshot is None else snapshot.nbytes,
        snapshot_has_last_logits=snapshot is not None and snapshot.last_logits is not None,
    )


@dataclass
class PrefixCacheFlow:
    cache_manager: Optional[RuntimeCacheManager]
    key: Optional[DFlashPrefixKey] = None
    stable_prefix_len: Optional[int] = None
    publish_generation_snapshot: bool = True
    snapshot: Optional[DFlashPrefixSnapshot] = None
    lookup_ms: float = 0.0
    hit_tokens: int = 0
    lookup_stats: PrefixCacheLookupStats = field(default_factory=PrefixCacheLookupStats)
    snapshot_service: Optional[SnapshotService] = None

    @property
    def cache_active(self) -> bool:
        return self.cache_manager is not None

    @property
    def insert_ms(self) -> float:
        if self.snapshot_service is None:
            return 0.0
        return self.snapshot_service.insert_ms

    def prefix_cache_memory_bytes(self) -> Optional[dict[str, int]]:
        if self.cache_manager is None:
            return None
        try:
            return self.cache_manager.memory_waterfall_bytes()
        except RuntimeCacheManagerClosed:
            return None

    @classmethod
    def for_request(
        cls,
        *,
        model_provider: Any,
        draft_model: Any,
        tokenizer: Any,
        prompt: list[int],
        request: Any = None,
        request_id: int | None = None,
        runtime_context: Optional[Any] = None,
    ) -> "PrefixCacheFlow":
        if runtime_context is None:
            return cls(cache_manager=None)

        runtime_config = runtime_context.runtime
        if runtime_config.target_fa_window > 0 or not runtime_config.prefix_cache:
            get_runtime_cache_manager(runtime_context)
            return cls(cache_manager=None)

        key = build_prefix_key(model_provider, draft_model, runtime_context)
        cache_manager = get_runtime_cache_manager(runtime_context, cache_identity=key)
        if cache_manager is None:
            return cls(cache_manager=None)

        stable_prefix_len = compute_request_stable_prefix_len(
            prompt,
            tokenizer=tokenizer,
            request=request,
        )
        lookup_tokens = prompt[:stable_prefix_len]
        try:
            if request_id is None:
                lookup = cache_manager.lookup(lookup_tokens, key)
            else:
                lookup = cache_manager.lookup(
                    lookup_tokens,
                    key,
                    request_id=request_id,
                )
        except RuntimeCacheManagerClosed:
            return cls(cache_manager=None)
        hit_tokens = int(lookup.matched_tokens)
        lookup_stats = _lookup_stats(
            prompt=prompt,
            stable_prefix_len=stable_prefix_len,
            lookup_ms=lookup.elapsed_ms,
            hit_tokens=hit_tokens,
            snapshot=lookup.snapshot,
        )
        if lookup.matched_tokens > 0:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix cache hit "
                f"{hit_tokens}/{len(prompt)} tokens (stable prefix {stable_prefix_len})\n"
            )
            sys.stderr.flush()
        try:
            cache_manager.log_stats(label="lookup")
        except RuntimeCacheManagerClosed:
            cache_manager = None
        return cls(
            cache_manager=cache_manager,
            key=key,
            stable_prefix_len=stable_prefix_len,
            publish_generation_snapshot=True,
            snapshot=lookup.snapshot,
            lookup_ms=lookup.elapsed_ms,
            hit_tokens=hit_tokens,
            lookup_stats=lookup_stats,
            snapshot_service=(
                SnapshotService.from_request(
                    cache_manager=cache_manager,
                    key=key,
                    draft_model=draft_model,
                    runtime_context=runtime_context,
                )
                if cache_manager is not None
                else None
            ),
        )

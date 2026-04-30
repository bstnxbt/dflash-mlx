# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)
from __future__ import annotations

import time
from typing import Any


def arm_target_rollback_with_prefix(
    cache_entries: list[Any],
    *,
    prefix_len: int,
) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback(prefix_len=int(prefix_len))


def clear_rollback_state(cache_entry: Any) -> None:
    if hasattr(cache_entry, "clear_transients"):
        cache_entry.clear_transients()
        return
    if hasattr(cache_entry, "_armed"):
        cache_entry._armed = False
    if hasattr(cache_entry, "_tape"):
        cache_entry._tape = None
    if hasattr(cache_entry, "_tape_k"):
        cache_entry._tape_k = None
    if hasattr(cache_entry, "_tape_g"):
        cache_entry._tape_g = None
    if hasattr(cache_entry, "_tape_qkv"):
        cache_entry._tape_qkv = None
    if hasattr(cache_entry, "_snapshot"):
        cache_entry._snapshot = None


def cleanup_generation_caches(
    target_cache: list[Any],
    draft_cache: list[Any],
) -> None:
    for cache_entry in target_cache:
        if hasattr(cache_entry, "clear_transients"):
            cache_entry.clear_transients()
    draft_cache.clear()
    target_cache.clear()


def restore_target_cache_after_acceptance(
    cache_entries: list[Any],
    *,
    target_len: int,
    acceptance_length: int,
    drafted_tokens: int = 0,
    force_replay: bool = False,
) -> int:
    replay_ns_total = 0
    fully_accepted = acceptance_length == drafted_tokens and not force_replay
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "rollback"):
            if fully_accepted:
                clear_rollback_state(cache_entry)
                continue
            replay_start_ns = time.perf_counter_ns()
            cache_entry.rollback(acceptance_length)
            replay_ns_total += time.perf_counter_ns() - replay_start_ns
        elif hasattr(cache_entry, "trim"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                replay_start_ns = time.perf_counter_ns()
                cache_entry.trim(offset - target_len)
                replay_ns_total += time.perf_counter_ns() - replay_start_ns
        elif hasattr(cache_entry, "offset"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                cache_entry.offset = target_len
        elif hasattr(cache_entry, "crop"):
            cache_entry.crop(target_len)
    return replay_ns_total

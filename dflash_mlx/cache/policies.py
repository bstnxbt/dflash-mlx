from __future__ import annotations

import os
from typing import Optional


def compute_stable_prefix_len(
    tokens: list[int] | tuple[int, ...],
    *,
    im_start_id: Optional[int] = None,
    assistant_id: Optional[int] = None,
) -> int:
    if im_start_id is None or assistant_id is None:
        return len(tokens)
    n = len(tokens)
    if n < 2:
        return n
    for i in range(n - 2, -1, -1):
        if tokens[i] == im_start_id and tokens[i + 1] == assistant_id:
            return i
    return n


def prefix_cache_enabled() -> bool:
    val = os.environ.get("DFLASH_PREFIX_CACHE", "0").strip()
    return val not in ("", "0", "false", "False", "no", "NO")


def read_budget_env(
    *,
    default_entries: int = 4,
    default_bytes: int = 8 * 1024 * 1024 * 1024,
) -> tuple[int, int]:
    entries_raw = os.environ.get("DFLASH_PREFIX_CACHE_MAX_ENTRIES", str(default_entries))
    bytes_raw = os.environ.get("DFLASH_PREFIX_CACHE_MAX_BYTES", str(default_bytes))
    try:
        entries = int(entries_raw)
    except ValueError:
        entries = default_entries
    try:
        max_bytes = int(bytes_raw)
    except ValueError:
        max_bytes = default_bytes
    return max(1, entries), max(0, max_bytes)

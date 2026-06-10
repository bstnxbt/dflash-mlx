# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import mlx.core as mx

from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot

def compute_snapshot_boundary(
    prompt_len: int,
    stable_prefix_len: Optional[int],
) -> int:
    if stable_prefix_len is not None and 0 < stable_prefix_len <= prompt_len:
        return int(stable_prefix_len)
    return int(prompt_len)

def spans_cover_prefix(
    spans: Iterable[tuple[int, int]],
    prefix_len: int,
) -> bool:
    needed = int(prefix_len)
    if needed <= 0:
        return True
    covered = 0
    for start, end in sorted((int(start), int(end)) for start, end in spans):
        if start > covered:
            return False
        covered = max(covered, end)
        if covered >= needed:
            return True
    return covered >= needed

def snapshot_covers_prefix(
    prefix_snapshot: DFlashPrefixSnapshot,
    prefix_len: int,
) -> bool:
    """True when the stored feature chunk spans cover [0, prefix_len) with no gap.

    Trimmed (sink+window) snapshots leave a hole that
    init_target_hidden_from_snapshot would silently zero-fill; callers that
    need full-context features must treat such snapshots as a miss.
    """
    return spans_cover_prefix(prefix_snapshot.target_hidden_chunk_spans, prefix_len)

def init_target_hidden_from_snapshot(
    prefix_snapshot: DFlashPrefixSnapshot,
    snap_prefix_len: int,
    prompt_len: int,
) -> mx.array:
    chunks = prefix_snapshot.target_hidden_chunks
    spans = prefix_snapshot.target_hidden_chunk_spans
    if not chunks:
        raise ValueError("Snapshot has empty target_hidden_chunks")
    ref = chunks[0]
    batch = int(ref.shape[0])
    hidden_dim = int(ref.shape[-1])
    target_hidden = mx.zeros((batch, prompt_len, hidden_dim), dtype=ref.dtype)
    for chunk, (start, end) in zip(chunks, spans):
        copy_start = int(start)
        copy_end = min(int(end), int(snap_prefix_len), int(prompt_len))
        if copy_end <= copy_start:
            continue
        chunk_offset = copy_start - int(start)
        chunk_len = copy_end - copy_start
        target_hidden[:, copy_start:copy_end, :] = chunk[:, chunk_offset:chunk_offset + chunk_len, :]
    mx.eval(target_hidden)
    return target_hidden

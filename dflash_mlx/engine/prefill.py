# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

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

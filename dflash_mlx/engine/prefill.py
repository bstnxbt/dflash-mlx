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
    cached_hidden = prefix_snapshot.target_hidden
    hidden_dim = int(cached_hidden.shape[-1])
    target_hidden = mx.zeros(
        (int(cached_hidden.shape[0]), prompt_len, hidden_dim),
        dtype=cached_hidden.dtype,
    )
    copy_len = min(snap_prefix_len, int(cached_hidden.shape[1]))
    if copy_len > 0:
        target_hidden[:, :copy_len, :] = cached_hidden[:, :copy_len, :]
    mx.eval(target_hidden)
    return target_hidden

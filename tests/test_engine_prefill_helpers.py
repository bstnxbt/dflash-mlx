# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from dflash_mlx.engine.prefill import (
    compute_snapshot_boundary,
    init_target_hidden_from_snapshot,
)

def test_snapshot_boundary_defaults_to_prompt_len_when_unset():
    assert compute_snapshot_boundary(prompt_len=128, stable_prefix_len=None) == 128

def test_snapshot_boundary_clamps_to_stable_prefix_when_in_range():
    assert compute_snapshot_boundary(prompt_len=128, stable_prefix_len=64) == 64

def test_snapshot_boundary_ignores_stable_prefix_overshoot():
    assert compute_snapshot_boundary(prompt_len=64, stable_prefix_len=128) == 64

def test_snapshot_boundary_ignores_zero_or_negative_stable_prefix():
    assert compute_snapshot_boundary(prompt_len=64, stable_prefix_len=0) == 64
    assert compute_snapshot_boundary(prompt_len=64, stable_prefix_len=-1) == 64

def _full_chunk_snap(target_hidden):
    total_len = int(target_hidden.shape[1])
    return SimpleNamespace(
        target_hidden_chunks=(target_hidden,),
        target_hidden_chunk_spans=((0, total_len),),
        target_hidden_total_len=total_len,
    )

def test_init_target_hidden_copies_snapshot_rows():
    cached_hidden = mx.arange(1 * 5 * 3, dtype=mx.float32).reshape(1, 5, 3)
    snap = _full_chunk_snap(cached_hidden)
    out = init_target_hidden_from_snapshot(snap, snap_prefix_len=5, prompt_len=8)
    assert out.shape == (1, 8, 3)
    assert mx.all(out[:, :5, :] == cached_hidden).item()
    assert mx.all(out[:, 5:, :] == 0).item()

def test_init_target_hidden_clamps_copy_len_to_cache_width():
    cached_hidden = mx.ones((1, 3, 2), dtype=mx.float32)
    snap = _full_chunk_snap(cached_hidden)
    out = init_target_hidden_from_snapshot(snap, snap_prefix_len=10, prompt_len=10)
    assert out.shape == (1, 10, 2)
    assert mx.all(out[:, :3, :] == 1).item()
    assert mx.all(out[:, 3:, :] == 0).item()

def test_init_target_hidden_with_zero_copy_len():
    cached_hidden = mx.ones((1, 4, 2), dtype=mx.float32)
    snap = _full_chunk_snap(cached_hidden)
    out = init_target_hidden_from_snapshot(snap, snap_prefix_len=0, prompt_len=4)
    assert out.shape == (1, 4, 2)
    assert mx.all(out == 0).item()

def test_init_target_hidden_handles_chunked_trim():

    sink = mx.ones((1, 2, 4), dtype=mx.float32) * 7.0
    tail = mx.ones((1, 2, 4), dtype=mx.float32) * 9.0
    snap = SimpleNamespace(
        target_hidden_chunks=(sink, tail),
        target_hidden_chunk_spans=((0, 2), (10, 12)),
        target_hidden_total_len=12,
    )
    out = init_target_hidden_from_snapshot(snap, snap_prefix_len=12, prompt_len=12)
    assert out.shape == (1, 12, 4)

    assert mx.all(out[:, :2, :] == 7.0).item()

    assert mx.all(out[:, 2:10, :] == 0.0).item()

    assert mx.all(out[:, 10:12, :] == 9.0).item()

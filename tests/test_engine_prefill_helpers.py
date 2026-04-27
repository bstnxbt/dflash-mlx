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


def test_init_target_hidden_copies_snapshot_rows():
    cached_hidden = mx.arange(1 * 5 * 3, dtype=mx.float32).reshape(1, 5, 3)
    snap = SimpleNamespace(target_hidden=cached_hidden)
    out = init_target_hidden_from_snapshot(snap, snap_prefix_len=5, prompt_len=8)
    assert out.shape == (1, 8, 3)
    assert mx.all(out[:, :5, :] == cached_hidden).item()
    assert mx.all(out[:, 5:, :] == 0).item()


def test_init_target_hidden_clamps_copy_len_to_cache_width():
    cached_hidden = mx.ones((1, 3, 2), dtype=mx.float32)
    snap = SimpleNamespace(target_hidden=cached_hidden)
    out = init_target_hidden_from_snapshot(snap, snap_prefix_len=10, prompt_len=10)
    assert out.shape == (1, 10, 2)
    assert mx.all(out[:, :3, :] == 1).item()
    assert mx.all(out[:, 3:, :] == 0).item()


def test_init_target_hidden_with_zero_copy_len():
    cached_hidden = mx.ones((1, 4, 2), dtype=mx.float32)
    snap = SimpleNamespace(target_hidden=cached_hidden)
    out = init_target_hidden_from_snapshot(snap, snap_prefix_len=0, prompt_len=4)
    assert out.shape == (1, 4, 2)
    assert mx.all(out == 0).item()

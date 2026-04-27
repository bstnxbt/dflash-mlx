from __future__ import annotations

import mlx.core as mx

from dflash_mlx.engine.acceptance import match_acceptance_length


def test_empty_draft_returns_zero():
    out = match_acceptance_length(
        mx.array([], dtype=mx.uint32),
        mx.array([], dtype=mx.uint32),
    )
    assert int(out.item()) == 0


def test_full_match_returns_full_length():
    drafted = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    posterior = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    out = match_acceptance_length(drafted, posterior)
    assert int(out.item()) == 4


def test_mismatch_at_position_zero_returns_zero():
    drafted = mx.array([5, 5, 5], dtype=mx.uint32)
    posterior = mx.array([1, 5, 5], dtype=mx.uint32)
    out = match_acceptance_length(drafted, posterior)
    assert int(out.item()) == 0


def test_partial_match_stops_at_first_mismatch():
    drafted = mx.array([1, 2, 9, 4, 5], dtype=mx.uint32)
    posterior = mx.array([1, 2, 3, 4, 5], dtype=mx.uint32)
    out = match_acceptance_length(drafted, posterior)
    assert int(out.item()) == 2


def test_late_mismatch_does_not_zero_earlier_matches():
    drafted = mx.array([7, 7, 7, 0], dtype=mx.uint32)
    posterior = mx.array([7, 7, 7, 1], dtype=mx.uint32)
    out = match_acceptance_length(drafted, posterior)
    assert int(out.item()) == 3

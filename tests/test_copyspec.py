from __future__ import annotations

import pytest

from dflash_mlx.engine.copyspec import CopySpecIndex


def test_copyspec_draft_after_returns_full_prompt_copy() -> None:
    index = CopySpecIndex(
        [1, 2, 3, 4, 5, 0, 7, 8, 1, 2, 3, 4, 5],
        window_size=6,
    )

    draft = index.draft_after(0, max_tokens=2)

    assert draft == (7, 8)


def test_copyspec_rejects_partial_copy() -> None:
    index = CopySpecIndex([1, 2, 3, 4, 1, 2], window_size=3)

    assert index.draft_after(3, max_tokens=4) is None


def test_copyspec_append_committed_indexes_generated_history() -> None:
    index = CopySpecIndex([9, 9, 9, 1, 2], window_size=3)
    index.append_committed([3, 4, 1, 2])

    draft = index.draft_after(3, max_tokens=1)

    assert draft == (4,)


def test_copyspec_short_initial_prompt_stays_disabled() -> None:
    index = CopySpecIndex([1, 2], window_size=3)
    index.append_committed([3, 4, 1, 2])

    assert index.draft_after(3, max_tokens=1) is None


def test_copyspec_skips_forbidden_tokens() -> None:
    index = CopySpecIndex([1, 2, 3, 4, 5, 0, 7, 1, 2, 3, 4, 5], window_size=6)

    assert index.draft_after(0, max_tokens=1, forbidden_tokens={7}) is None


def test_copyspec_rejects_invalid_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        CopySpecIndex([1, 2, 3], window_size=0)


def test_copyspec_gate_starts_enabled() -> None:
    from dflash_mlx.engine.copyspec import CopySpecGate

    gate = CopySpecGate()

    assert gate.enabled
    assert gate.should_attempt()


def test_copyspec_gate_tolerates_misses_below_limit() -> None:
    from dflash_mlx.engine.copyspec import CopySpecGate

    gate = CopySpecGate(miss_limit=3, cooldown_blocks=4)
    gate.record_block(0)
    gate.record_block(0)

    assert gate.should_attempt()


def test_copyspec_gate_enters_cooldown_after_consecutive_misses() -> None:
    from dflash_mlx.engine.copyspec import CopySpecGate

    gate = CopySpecGate(miss_limit=3, cooldown_blocks=4)
    for _ in range(3):
        gate.record_block(0)

    assert not gate.enabled
    assert not gate.should_attempt()


def test_copyspec_gate_accepted_block_resets_miss_count() -> None:
    from dflash_mlx.engine.copyspec import CopySpecGate

    gate = CopySpecGate(miss_limit=3, cooldown_blocks=4)
    gate.record_block(0)
    gate.record_block(0)
    gate.record_block(5)
    gate.record_block(0)
    gate.record_block(0)

    assert gate.should_attempt()


def test_copyspec_gate_reenables_after_cooldown() -> None:
    from dflash_mlx.engine.copyspec import CopySpecGate

    gate = CopySpecGate(miss_limit=3, cooldown_blocks=4)
    for _ in range(3):
        gate.record_block(0)

    for _ in range(4):
        assert not gate.should_attempt()

    assert gate.should_attempt()


def test_copyspec_gate_probe_miss_reenters_cooldown() -> None:
    from dflash_mlx.engine.copyspec import CopySpecGate

    gate = CopySpecGate(miss_limit=3, cooldown_blocks=4)
    for _ in range(3):
        gate.record_block(0)
    for _ in range(4):
        gate.should_attempt()
    assert gate.should_attempt()

    gate.record_block(0)

    assert not gate.should_attempt()


def test_copyspec_gate_probe_accept_fully_restores_tolerance() -> None:
    from dflash_mlx.engine.copyspec import CopySpecGate

    gate = CopySpecGate(miss_limit=3, cooldown_blocks=4)
    for _ in range(3):
        gate.record_block(0)
    for _ in range(4):
        gate.should_attempt()
    gate.record_block(10)

    gate.record_block(0)
    gate.record_block(0)

    assert gate.should_attempt()


def test_copyspec_gate_rejects_invalid_miss_limit() -> None:
    from dflash_mlx.engine.copyspec import CopySpecGate

    with pytest.raises(ValueError, match="miss_limit"):
        CopySpecGate(miss_limit=0)


def test_copyspec_gate_rejects_invalid_cooldown() -> None:
    from dflash_mlx.engine.copyspec import CopySpecGate

    with pytest.raises(ValueError, match="cooldown_blocks"):
        CopySpecGate(cooldown_blocks=0)

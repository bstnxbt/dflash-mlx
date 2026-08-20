from __future__ import annotations

import pytest

from dflash_mlx.engine.copyspec import CopySpecAutoGate, CopySpecIndex


def test_copyspec_returns_a_full_prompt_copy() -> None:
    index = CopySpecIndex(
        [1, 2, 3, 4, 5, 0, 7, 8, 1, 2, 3, 4, 5],
        window_size=6,
    )

    assert index.draft_after(0, max_tokens=2) == (7, 8)


def test_copyspec_rejects_partial_or_forbidden_copies() -> None:
    partial = CopySpecIndex([1, 2, 3, 4, 1, 2], window_size=3)
    forbidden = CopySpecIndex(
        [1, 2, 3, 4, 5, 0, 7, 1, 2, 3, 4, 5],
        window_size=6,
    )

    assert partial.draft_after(3, max_tokens=4) is None
    assert forbidden.draft_after(0, max_tokens=1, forbidden_tokens={7}) is None


def test_copyspec_indexes_committed_history() -> None:
    index = CopySpecIndex([9, 9, 9, 1, 2], window_size=3)
    index.append_committed([3, 4, 1, 2])

    assert index.draft_after(3, max_tokens=1) == (4,)


def test_copyspec_short_initial_prompt_stays_disabled() -> None:
    index = CopySpecIndex([1, 2], window_size=3)
    index.append_committed([3, 4, 1, 2])

    assert index.draft_after(3, max_tokens=1) is None


def _gate(**overrides) -> CopySpecAutoGate:
    config = {
        "probe_off_cycles": 4,
        "probe_on_min_copy": 2,
        "probe_on_max_cycles": 8,
        "latch_cycles": 3,
        "margin": 1.0,
    }
    config.update(overrides)
    return CopySpecAutoGate(**config)


def _record(
    gate: CopySpecAutoGate,
    *,
    source: str,
    cycles: int,
    cost_ns: int,
) -> None:
    for _ in range(cycles):
        gate.record_cycle(
            source=source,
            commit_count=4,
            cycle_cost_ns=cost_ns,
        )


@pytest.mark.parametrize(
    "baseline_ns,copy_ns,expected_phase,expected_engaged,expected_reset",
    [
        (1_000_000, 50_000, "engaged", True, False),
        (50_000, 1_000_000, "dormant", False, True),
    ],
)
def test_auto_gate_selects_the_faster_path(
    baseline_ns: int,
    copy_ns: int,
    expected_phase: str,
    expected_engaged: bool,
    expected_reset: bool,
) -> None:
    gate = _gate()
    assert not gate.engage_copy()

    _record(gate, source="dflash", cycles=4, cost_ns=baseline_ns)
    assert gate.phase == "measure_on"
    assert gate.engage_copy()

    _record(gate, source="copy", cycles=2, cost_ns=copy_ns)
    assert gate.phase == expected_phase
    assert gate.engage_copy() is expected_engaged
    assert gate.take_reset() is expected_reset
    assert gate.metrics()["engages" if expected_engaged else "disengages"] == 1


def test_auto_gate_rebaselines_after_engaged_latch() -> None:
    gate = _gate(latch_cycles=3)
    _record(gate, source="dflash", cycles=4, cost_ns=1_000_000)
    _record(gate, source="copy", cycles=2, cost_ns=50_000)
    assert gate.phase == "engaged"

    _record(gate, source="copy", cycles=3, cost_ns=50_000)

    assert gate.phase == "measure_off"
    assert gate.take_reset()


def test_auto_gate_abandons_probe_without_copy_blocks() -> None:
    gate = _gate(probe_on_max_cycles=3)
    _record(gate, source="dflash", cycles=4, cost_ns=500_000)
    _record(gate, source="dflash", cycles=3, cost_ns=500_000)

    assert gate.phase == "dormant"
    assert gate.take_reset()


@pytest.mark.parametrize(
    "invalid",
    [
        {"probe_off_cycles": 0},
        {"probe_on_min_copy": 0},
        {"probe_on_max_cycles": 0},
        {"latch_cycles": 0},
        {"margin": 0.0},
    ],
)
def test_auto_gate_rejects_invalid_configuration(invalid: dict) -> None:
    with pytest.raises(ValueError):
        _gate(**invalid)

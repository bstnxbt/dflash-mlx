from __future__ import annotations

import pytest

from dflash_mlx.engine.copyspec import CopySpecIndex, CopySpecAutoGate


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


# ────────────────────────────────────────────────────────────────
# CopySpecAutoGate tests
# ────────────────────────────────────────────────────────────────

def _make_gate(**kwargs) -> CopySpecAutoGate:
    """Tiny helper: gate with small windows so tests run in <10 cycles."""
    defaults = dict(
        probe_off_cycles=4,
        probe_on_min_copy=2,
        probe_on_max_cycles=8,
        latch_cycles=6,
        margin=1.0,
    )
    defaults.update(kwargs)
    return CopySpecAutoGate(**defaults)


def _dflash(gate: CopySpecAutoGate, *, commits: int = 4, cost_ns: int | None = 500_000) -> None:
    """Feed one dflash cycle into the gate."""
    gate.record_cycle(source="dflash", commit_count=commits, cycle_cost_ns=cost_ns)


def _copy(gate: CopySpecAutoGate, *, commits: int = 4, cost_ns: int | None = 500_000) -> None:
    """Feed one copy cycle into the gate."""
    gate.record_cycle(source="copy", commit_count=commits, cycle_cost_ns=cost_ns)


# 1. Initial state ────────────────────────────────────────────────

def test_starts_in_measure_off() -> None:
    gate = _make_gate()
    assert gate.phase == "measure_off"


def test_engage_copy_false_in_measure_off() -> None:
    gate = _make_gate()
    assert gate.engage_copy() is False


# 2. measure_off → measure_on transition ──────────────────────────

def test_after_probe_off_cycles_transitions_to_measure_on() -> None:
    gate = _make_gate(probe_off_cycles=4)
    for _ in range(4):
        _dflash(gate)
    assert gate.phase == "measure_on"
    assert gate.engage_copy() is True


def test_off_rate_captured_after_measure_off() -> None:
    gate = _make_gate(probe_off_cycles=4)
    # 4 cycles × 8 commits each, cost 500_000 ns each → rate = 32 / (4×500_000/1e9)
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=8, cycle_cost_ns=500_000)
    expected = 32 / (4 * 500_000 / 1e9)  # 16_000_000.0
    assert gate._off_rate == pytest.approx(expected)


def test_off_rate_not_set_before_measure_off_ends() -> None:
    gate = _make_gate(probe_off_cycles=4)
    _dflash(gate)
    assert gate._off_rate is None


# 3. measure_on → engaged (copy wins) ────────────────────────────

def test_copy_wins_transitions_to_engaged() -> None:
    """Copy cycles with tiny cost (high tps) beat the dflash off-rate."""
    gate = _make_gate(probe_off_cycles=4, probe_on_min_copy=2, margin=1.0)
    # Off phase: 4 cycles, 4 commits, 1_000_000 ns each → off_rate = 16/(4e-3) = 4_000.0 tps
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=1_000_000)
    assert gate.phase == "measure_on"
    # On phase: 2 copy cycles, 4 commits, 50_000 ns each → on_rate = 8/(100_000/1e9) = 80_000 tps
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    assert gate.phase == "engaged"
    assert gate.engage_copy() is True


def test_copy_wins_no_reset_pending() -> None:
    gate = _make_gate(probe_off_cycles=4, probe_on_min_copy=2, margin=1.0)
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=1_000_000)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    assert gate._pending_reset is False
    assert gate.take_reset() is False


def test_copy_wins_engages_counter() -> None:
    gate = _make_gate(probe_off_cycles=4, probe_on_min_copy=2, margin=1.0)
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=1_000_000)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    assert gate.metrics()["engages"] == 1


# 4. measure_on → dormant (copy loses) ───────────────────────────

def test_copy_loses_transitions_to_dormant() -> None:
    """Copy cycles with huge cost (low tps) lose to the dflash off-rate."""
    gate = _make_gate(probe_off_cycles=4, probe_on_min_copy=2, margin=1.0)
    # Off: 4 commits × 50_000 ns → off_rate = 16/(4×50_000/1e9) = 80_000 tps (fast baseline)
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=50_000)
    assert gate.phase == "measure_on"
    # On: 2 copy cycles × 4 commits × 1_000_000 ns → on_rate = 8/(2e-3) = 4_000 tps (slow)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=1_000_000)
    assert gate.phase == "dormant"
    assert gate.engage_copy() is False


def test_copy_loses_take_reset_true_once() -> None:
    gate = _make_gate(probe_off_cycles=4, probe_on_min_copy=2, margin=1.0)
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=50_000)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=1_000_000)
    assert gate.take_reset() is True
    assert gate.take_reset() is False  # consumed


def test_copy_loses_disengages_counter() -> None:
    gate = _make_gate(probe_off_cycles=4, probe_on_min_copy=2, margin=1.0)
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=50_000)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=1_000_000)
    assert gate.metrics()["disengages"] == 1


# 5. measure_on max_cycles with zero copy blocks → dormant + reset ─

def test_measure_on_max_cycles_no_copy_goes_dormant() -> None:
    gate = _make_gate(probe_off_cycles=4, probe_on_min_copy=2, probe_on_max_cycles=5)
    for _ in range(4):
        _dflash(gate)
    assert gate.phase == "measure_on"
    # 5 dflash cycles (no copy) to hit max_cycles
    for _ in range(5):
        _dflash(gate)
    assert gate.phase == "dormant"
    assert gate.take_reset() is True


# 6. Engaged latch expiry → measure_off + reset ───────────────────

def test_engaged_latch_expiry_returns_to_measure_off_with_reset() -> None:
    gate = _make_gate(
        probe_off_cycles=4, probe_on_min_copy=2, latch_cycles=3, margin=1.0
    )
    # Drive to engaged
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=1_000_000)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    assert gate.phase == "engaged"
    # Expire latch (3 cycles)
    for _ in range(3):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    assert gate.phase == "measure_off"
    assert gate.take_reset() is True


def test_engaged_cycles_counter_increments() -> None:
    gate = _make_gate(
        probe_off_cycles=4, probe_on_min_copy=2, latch_cycles=3, margin=1.0
    )
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=1_000_000)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    # 2 more engaged cycles (latch=3, not yet expired)
    gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    assert gate.metrics()["engaged_cycles"] == 2


# 7. Dormant latch expiry → measure_off, NO reset ─────────────────

def test_dormant_latch_expiry_returns_to_measure_off_no_reset() -> None:
    gate = _make_gate(
        probe_off_cycles=4, probe_on_min_copy=2, latch_cycles=3, margin=1.0
    )
    # Drive to dormant
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=50_000)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=1_000_000)
    assert gate.phase == "dormant"
    # Consume the reset from entering dormant
    gate.take_reset()
    # Expire latch
    for _ in range(3):
        _dflash(gate)
    assert gate.phase == "measure_off"
    assert gate.take_reset() is False


def test_dormant_cycles_counter_increments() -> None:
    gate = _make_gate(
        probe_off_cycles=4, probe_on_min_copy=2, latch_cycles=5, margin=1.0
    )
    for _ in range(4):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=50_000)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=1_000_000)
    assert gate.phase == "dormant"
    for _ in range(2):
        _dflash(gate)
    assert gate.metrics()["dormant_cycles"] == 2


# 8. _win_rate: tps vs tpc fallback ──────────────────────────────

def test_win_rate_uses_tps_when_cost_present() -> None:
    gate = _make_gate(probe_off_cycles=100)  # won't transition during test
    gate.record_cycle(source="dflash", commit_count=16, cycle_cost_ns=1_000_000)
    # 16 commits / (1_000_000 / 1e9 seconds) = 16_000.0 tps
    assert gate._win_rate() == pytest.approx(16_000.0)


def test_win_rate_falls_back_to_tpc_when_no_cost() -> None:
    gate = _make_gate(probe_off_cycles=100)
    gate.record_cycle(source="dflash", commit_count=8, cycle_cost_ns=None)
    gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=None)
    # 12 commits / 2 cycles = 6.0 tpc
    assert gate._win_rate() == pytest.approx(6.0)


def test_win_rate_returns_zero_when_empty() -> None:
    gate = _make_gate()
    assert gate._win_rate() == 0.0


# 9. Full lifecycle: measure_off → measure_on(win) → engaged → (latch) → measure_off

def test_full_lifecycle_phase_sequence_and_reset_signals() -> None:
    gate = _make_gate(
        probe_off_cycles=3,
        probe_on_min_copy=2,
        probe_on_max_cycles=8,
        latch_cycles=4,
        margin=1.0,
    )
    phases = [gate.phase]

    # measure_off: 3 dflash cycles
    for _ in range(3):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=2_000_000)
    phases.append(gate.phase)  # should be measure_on
    assert gate.take_reset() is False  # no reset on transition to measure_on

    # measure_on: 2 copy cycles (fast → win)
    for _ in range(2):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    phases.append(gate.phase)  # should be engaged
    assert gate.take_reset() is False  # no reset on engage

    # engaged: run latch_cycles to expire
    for _ in range(4):
        gate.record_cycle(source="copy", commit_count=4, cycle_cost_ns=50_000)
    phases.append(gate.phase)  # should be measure_off
    assert gate.take_reset() is True  # reset when re-baselining

    assert phases == ["measure_off", "measure_on", "engaged", "measure_off"]


# 10. __init__ validation ──────────────────────────────────────────

@pytest.mark.parametrize("bad_kwargs,match", [
    ({"probe_off_cycles": 0}, "probe_off_cycles"),
    ({"probe_on_min_copy": 0}, "probe_on_min_copy"),
    ({"probe_on_max_cycles": 0}, "probe_on_max_cycles"),
    ({"latch_cycles": 0}, "latch_cycles"),
    ({"margin": 0.0}, "margin"),
    ({"margin": -1.0}, "margin"),
])
def test_init_validation(bad_kwargs: dict, match: str) -> None:
    valid = dict(
        probe_off_cycles=4,
        probe_on_min_copy=2,
        probe_on_max_cycles=8,
        latch_cycles=6,
        margin=1.0,
    )
    valid.update(bad_kwargs)
    with pytest.raises(ValueError, match=match):
        CopySpecAutoGate(**valid)


# 11. metrics() shape ─────────────────────────────────────────────

def test_metrics_returns_expected_keys() -> None:
    gate = _make_gate()
    m = gate.metrics()
    assert set(m.keys()) == {
        "state", "off_rate", "engaged_cycles",
        "dormant_cycles", "probes", "engages", "disengages",
    }
    assert m["state"] == "measure_off"
    assert m["off_rate"] is None
    assert m["engaged_cycles"] == 0


# 12. probes counter ──────────────────────────────────────────────

def test_probes_counter_increments_each_measure_on_entry() -> None:
    gate = _make_gate(probe_off_cycles=2, probe_on_min_copy=1, latch_cycles=2, margin=1.0)
    # First probe cycle: off → on → dormant → off
    for _ in range(2):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=50_000)
    assert gate.metrics()["probes"] == 1
    # Drain measure_on with 1 copy that loses (high cost)
    gate.record_cycle(source="copy", commit_count=1, cycle_cost_ns=5_000_000)
    # wait latch to expire
    for _ in range(2):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=50_000)
    # Re-baseline complete → second probe
    for _ in range(2):
        gate.record_cycle(source="dflash", commit_count=4, cycle_cost_ns=50_000)
    assert gate.metrics()["probes"] == 2

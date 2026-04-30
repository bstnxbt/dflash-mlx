from __future__ import annotations

from dflash_mlx.engine.adaptive_fallback import (
    AdaptiveFallbackConfig,
    AdaptiveFallbackState,
    resolve_adaptive_fallback_config,
)


def test_adaptive_fallback_disabled_by_default():
    cfg = resolve_adaptive_fallback_config({})
    state = AdaptiveFallbackState(config=cfg, initial_block_tokens=8)

    assert cfg.enabled is False
    assert state.block_tokens_for_cycle(32) == 8
    assert state.record_cycle(block_len=8, commit_count=1) is None
    assert state.summary_fields()["adaptive_fallback_enabled"] is False


def test_low_tokens_per_cycle_enters_target_ar_cooldown():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=3.0,
            probe_cycles=2,
            bad_probe_windows=1,
            cooldown_tokens=4,
        ),
        initial_block_tokens=8,
    )

    assert state.record_cycle(block_len=8, commit_count=2) is None
    decision = state.record_cycle(block_len=8, commit_count=1)

    assert decision is not None
    assert decision.action == "fallback"
    assert decision.tokens_per_cycle == 1.5
    assert decision.cooldown_tokens == 4
    assert decision.block_tokens == 8
    assert decision.next_block_tokens == 4
    assert state.in_cooldown is True
    assert state.block_tokens_for_cycle(32) == 1
    assert state.summary_fields()["adaptive_fallback_triggered"] is True


def test_cooldown_completion_starts_reprobe_with_smaller_block():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=3.0,
            probe_cycles=1,
            bad_probe_windows=1,
            cooldown_tokens=2,
        ),
        initial_block_tokens=8,
    )

    fallback = state.record_cycle(block_len=8, commit_count=1)
    assert fallback is not None
    assert fallback.action == "fallback"

    assert state.record_cycle(block_len=1, commit_count=1) is None
    reprobe = state.record_cycle(block_len=1, commit_count=1)

    assert reprobe is not None
    assert reprobe.action == "reprobe"
    assert reprobe.block_tokens == 8
    assert reprobe.next_block_tokens == 4
    assert state.in_cooldown is False
    assert state.block_tokens_for_cycle(32) == 4
    summary = state.summary_fields()
    assert summary["adaptive_fallback_tokens"] == 2
    assert summary["adaptive_reprobe_count"] == 1


def test_configured_reprobe_block_is_used_and_clamped():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=9.0,
            probe_cycles=1,
            bad_probe_windows=1,
            cooldown_tokens=1,
            reprobe_block_tokens=6,
        ),
        initial_block_tokens=16,
    )

    fallback = state.record_cycle(block_len=16, commit_count=1)
    assert fallback is not None
    assert fallback.next_block_tokens == 6
    reprobe = state.record_cycle(block_len=1, commit_count=1)
    assert reprobe is not None
    assert state.block_tokens_for_cycle(32) == 6

    clamped_cfg = AdaptiveFallbackConfig(
        enabled=True,
        min_tokens_per_cycle=9.0,
        probe_cycles=1,
        bad_probe_windows=1,
        cooldown_tokens=1,
        reprobe_block_tokens=64,
    )
    clamped = AdaptiveFallbackState(config=clamped_cfg, initial_block_tokens=16)
    clamped_fallback = clamped.record_cycle(block_len=16, commit_count=1)
    assert clamped_fallback is None
    assert clamped.summary_fields()["adaptive_fallback_count"] == 0


def test_probe_window_pass_resets_without_fallback():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=2,
            cooldown_tokens=4,
        ),
        initial_block_tokens=8,
    )

    assert state.record_cycle(block_len=8, commit_count=2) is None
    assert state.record_cycle(block_len=8, commit_count=2) is None

    summary = state.summary_fields()
    assert summary["adaptive_fallback_triggered"] is False
    assert summary["adaptive_last_probe_tokens_per_cycle"] == 2.0
    assert summary["adaptive_pending_probe_cycles"] == 0
    assert state.block_tokens_for_cycle(32) == 8


def test_single_bad_probe_window_waits_for_confirmation():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=2,
            bad_probe_windows=2,
            cooldown_tokens=4,
        ),
        initial_block_tokens=8,
    )

    assert state.record_cycle(block_len=8, commit_count=1) is None
    assert state.record_cycle(block_len=8, commit_count=1) is None

    summary = state.summary_fields()
    assert summary["adaptive_fallback_count"] == 0
    assert summary["adaptive_pending_bad_probe_windows"] == 1

    assert state.record_cycle(block_len=8, commit_count=2) is None
    assert state.record_cycle(block_len=8, commit_count=2) is None

    summary = state.summary_fields()
    assert summary["adaptive_fallback_count"] == 0
    assert summary["adaptive_pending_bad_probe_windows"] == 0
    assert state.block_tokens_for_cycle(32) == 8


def test_sustained_bad_probe_windows_enter_fallback():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=2,
            bad_probe_windows=2,
            cooldown_tokens=4,
        ),
        initial_block_tokens=8,
    )

    assert state.record_cycle(block_len=8, commit_count=1) is None
    assert state.record_cycle(block_len=8, commit_count=1) is None
    assert state.record_cycle(block_len=8, commit_count=1) is None
    decision = state.record_cycle(block_len=8, commit_count=1)

    assert decision is not None
    assert decision.action == "fallback"
    assert decision.tokens_per_cycle == 1.0
    assert state.in_cooldown is True


def test_default_cooldown_holds_target_ar_for_coding_length_outputs():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=2,
            bad_probe_windows=2,
        ),
        initial_block_tokens=16,
    )

    for _ in range(3):
        assert state.record_cycle(block_len=16, commit_count=1) is None
    fallback = state.record_cycle(block_len=16, commit_count=1)
    assert fallback is not None
    assert fallback.action == "fallback"
    assert fallback.cooldown_tokens == 4096

    for _ in range(1000):
        assert state.record_cycle(block_len=1, commit_count=1) is None

    assert state.in_cooldown is True
    summary = state.summary_fields()
    assert summary["adaptive_reprobe_count"] == 0
    assert summary["adaptive_fallback_tokens"] == 1000


def test_debug_trace_transient_bad_window_does_not_fallback():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=8,
            bad_probe_windows=2,
            cooldown_tokens=32,
        ),
        initial_block_tokens=16,
    )
    debug_acceptance_history = [
        4, 0, 4, 1, 3, 1, 4, 0,
        1, 5, 0, 2, 1, 1, 1, 1,
        0, 0, 2, 1, 3, 0, 3, 1,
        2, 2, 0, 0, 0, 0, 3, 1,
        1, 0, 1, 1, 1, 1, 0, 3,
        3, 4, 2, 3, 1, 1, 1, 3,
        3, 0, 3, 0, 1, 0, 0, 2,
        0, 4, 4, 0, 2, 1, 0, 1,
        0, 1, 0, 1, 1, 0, 1, 1,
        3, 2, 1, 1, 0, 0, 0, 2,
    ]

    for acceptance_len in debug_acceptance_history:
        decision = state.record_cycle(
            block_len=16,
            commit_count=1 + acceptance_len,
        )
        assert decision is None

    summary = state.summary_fields()
    assert summary["adaptive_fallback_count"] == 0
    assert state.in_cooldown is False


def test_final_probe_cycle_does_not_enter_fallback():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=9.0,
            probe_cycles=1,
            cooldown_tokens=4,
        ),
        initial_block_tokens=8,
    )

    decision = state.record_cycle(
        block_len=8,
        commit_count=1,
        can_continue=False,
    )

    assert decision is None
    assert state.in_cooldown is False
    summary = state.summary_fields()
    assert summary["adaptive_fallback_triggered"] is False
    assert summary["adaptive_fallback_count"] == 0
    assert summary["adaptive_last_probe_tokens_per_cycle"] == 1.0


def test_final_cooldown_cycle_does_not_record_reprobe():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=9.0,
            probe_cycles=1,
            bad_probe_windows=1,
            cooldown_tokens=1,
        ),
        initial_block_tokens=8,
    )

    fallback = state.record_cycle(block_len=8, commit_count=1)
    assert fallback is not None
    assert fallback.action == "fallback"

    decision = state.record_cycle(
        block_len=1,
        commit_count=1,
        can_continue=False,
    )

    assert decision is None
    summary = state.summary_fields()
    assert summary["adaptive_fallback_triggered"] is True
    assert summary["adaptive_reprobe_count"] == 0
    assert summary["adaptive_fallback_tokens"] == 1


def test_cooldown_restores_reference_when_target_ar_is_not_faster():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=1,
            bad_probe_windows=1,
            cooldown_tokens=2,
        ),
        initial_block_tokens=8,
    )

    fallback = state.record_cycle(block_len=8, commit_count=1, cycle_us=80_000)
    assert fallback is not None
    assert fallback.action == "fallback"

    assert state.record_cycle(block_len=1, commit_count=1, cycle_us=90_000) is None
    restore = state.record_cycle(block_len=1, commit_count=1, cycle_us=90_000)

    assert restore is not None
    assert restore.action == "restore"
    assert state.block_tokens_for_cycle(32) == 8
    summary = state.summary_fields()
    assert summary["adaptive_latency_locked"] is True
    assert summary["adaptive_latency_reject_count"] == 1
    assert summary["adaptive_reprobe_count"] == 0
    assert summary["adaptive_ar_ms_per_token"] == 90.0


def test_reprobe_restores_reference_when_smaller_block_is_not_faster():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=1,
            bad_probe_windows=1,
            cooldown_tokens=1,
        ),
        initial_block_tokens=8,
    )

    fallback = state.record_cycle(block_len=8, commit_count=1, cycle_us=80_000)
    assert fallback is not None
    reprobe = state.record_cycle(block_len=1, commit_count=1, cycle_us=50_000)
    assert reprobe is not None
    assert reprobe.action == "reprobe"

    restore = state.record_cycle(block_len=4, commit_count=1, cycle_us=90_000)

    assert restore is not None
    assert restore.action == "restore"
    assert state.block_tokens_for_cycle(32) == 8
    summary = state.summary_fields()
    assert summary["adaptive_latency_locked"] is True
    assert summary["adaptive_reference_block_tokens"] == 8
    assert summary["adaptive_reference_ms_per_token"] == 80.0
    assert summary["adaptive_last_probe_ms_per_token"] == 90.0


def test_reprobe_can_continue_when_smaller_block_is_faster():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=1,
            bad_probe_windows=1,
            cooldown_tokens=1,
        ),
        initial_block_tokens=8,
    )

    fallback = state.record_cycle(block_len=8, commit_count=1, cycle_us=80_000)
    assert fallback is not None
    reprobe = state.record_cycle(block_len=1, commit_count=1, cycle_us=50_000)
    assert reprobe is not None

    second_fallback = state.record_cycle(block_len=4, commit_count=1, cycle_us=40_000)

    assert second_fallback is not None
    assert second_fallback.action == "fallback"
    assert second_fallback.block_tokens == 4
    assert second_fallback.next_block_tokens == 2
    summary = state.summary_fields()
    assert summary["adaptive_fallback_count"] == 2
    assert summary["adaptive_reference_block_tokens"] == 4
    assert summary["adaptive_reference_ms_per_token"] == 40.0


def test_does_not_enter_cooldown_when_block_cannot_shrink():
    state = AdaptiveFallbackState(
        config=AdaptiveFallbackConfig(
            enabled=True,
            min_tokens_per_cycle=2.0,
            probe_cycles=1,
            bad_probe_windows=1,
            cooldown_tokens=4,
        ),
        initial_block_tokens=2,
    )

    decision = state.record_cycle(block_len=2, commit_count=1, cycle_us=50_000)

    assert decision is None
    assert state.in_cooldown is False
    assert state.summary_fields()["adaptive_fallback_count"] == 0


def test_env_resolver_requires_opt_in():
    disabled = resolve_adaptive_fallback_config({"DFLASH_ADAPTIVE_FALLBACK": "0"})
    enabled_default = resolve_adaptive_fallback_config({"DFLASH_ADAPTIVE_FALLBACK": "1"})
    enabled = resolve_adaptive_fallback_config(
        {
            "DFLASH_ADAPTIVE_FALLBACK": "1",
            "DFLASH_ADAPTIVE_MIN_TOKENS_PER_CYCLE": "2.5",
            "DFLASH_ADAPTIVE_PROBE_CYCLES": "3",
            "DFLASH_ADAPTIVE_BAD_PROBE_WINDOWS": "4",
            "DFLASH_ADAPTIVE_TARGET_AR_COOLDOWN": "5",
            "DFLASH_ADAPTIVE_LATENCY_MARGIN": "1.2",
            "DFLASH_ADAPTIVE_REPROBE_BLOCK_TOKENS": "7",
        }
    )

    assert disabled.enabled is False
    assert enabled_default.enabled is True
    assert enabled_default.bad_probe_windows == 2
    assert enabled_default.cooldown_tokens == 4096
    assert enabled_default.latency_margin == 1.0
    assert enabled_default.reprobe_block_tokens is None
    assert enabled.enabled is True
    assert enabled.min_tokens_per_cycle == 2.5
    assert enabled.probe_cycles == 3
    assert enabled.bad_probe_windows == 4
    assert enabled.cooldown_tokens == 5
    assert enabled.latency_margin == 1.2
    assert enabled.reprobe_block_tokens == 7

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


_FALSE_VALUES = {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class AdaptiveFallbackConfig:
    enabled: bool = False
    min_tokens_per_cycle: float = 2.0
    probe_cycles: int = 8
    bad_probe_windows: int = 2
    # Conservative hold after fallback.
    cooldown_tokens: int = 4096
    latency_margin: float = 1.0
    reprobe_block_tokens: Optional[int] = None


@dataclass
class AdaptiveDecision:
    action: str
    reason: str
    tokens_per_cycle: float
    probe_cycles: int
    probe_tokens: int
    cooldown_tokens: int
    block_tokens: int
    next_block_tokens: int
    fallback_count: int
    reprobe_count: int

    def as_event(self) -> dict[str, object]:
        return {
            "event": "adaptive_fallback",
            "action": self.action,
            "reason": self.reason,
            "tokens_per_cycle": self.tokens_per_cycle,
            "probe_cycles": self.probe_cycles,
            "probe_tokens": self.probe_tokens,
            "cooldown_tokens": self.cooldown_tokens,
            "block_tokens": self.block_tokens,
            "next_block_tokens": self.next_block_tokens,
            "fallback_count": self.fallback_count,
            "reprobe_count": self.reprobe_count,
        }


@dataclass
class AdaptiveFallbackState:
    config: AdaptiveFallbackConfig
    initial_block_tokens: int
    active_block_tokens: int = field(init=False)
    mode: str = field(default="probe", init=False)
    cooldown_remaining: int = field(default=0, init=False)
    cooldown_total_us: float = field(default=0.0, init=False)
    cooldown_observed_tokens: int = field(default=0, init=False)
    pending_reprobe_block_tokens: Optional[int] = field(default=None, init=False)
    probe_cycles_seen: int = field(default=0, init=False)
    probe_tokens: int = field(default=0, init=False)
    probe_total_us: float = field(default=0.0, init=False)
    bad_probe_windows_seen: int = field(default=0, init=False)
    last_probe_tokens_per_cycle: float = field(default=0.0, init=False)
    last_probe_us_per_token: Optional[float] = field(default=None, init=False)
    ar_us_per_token: Optional[float] = field(default=None, init=False)
    reference_block_tokens: Optional[int] = field(default=None, init=False)
    reference_us_per_token: Optional[float] = field(default=None, init=False)
    latency_reject_count: int = field(default=0, init=False)
    latency_locked: bool = field(default=False, init=False)
    fallback_count: int = field(default=0, init=False)
    reprobe_count: int = field(default=0, init=False)
    fallback_tokens: int = field(default=0, init=False)
    fallback_reason: Optional[str] = field(default=None, init=False)
    decisions: list[AdaptiveDecision] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.initial_block_tokens = max(1, int(self.initial_block_tokens))
        self.active_block_tokens = self.initial_block_tokens

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def in_cooldown(self) -> bool:
        return self.enabled and self.mode == "cooldown"

    def block_tokens_for_cycle(self, remaining_tokens: int) -> int:
        remaining_tokens = max(0, int(remaining_tokens))
        if remaining_tokens <= 0:
            return 0
        if self.in_cooldown:
            return 1
        return max(1, min(self.active_block_tokens, remaining_tokens))

    def record_cycle(
        self,
        *,
        block_len: int,
        commit_count: int,
        cycle_us: float = 0.0,
        can_continue: bool = True,
    ) -> Optional[AdaptiveDecision]:
        if not self.enabled:
            return None

        block_len = int(block_len)
        commit_count = max(0, int(commit_count))

        if self.in_cooldown:
            self.fallback_tokens += commit_count
            if cycle_us > 0.0 and commit_count > 0:
                self.cooldown_total_us += float(cycle_us)
                self.cooldown_observed_tokens += commit_count
            self.cooldown_remaining = max(0, self.cooldown_remaining - commit_count)
            if self.cooldown_remaining > 0 or not can_continue:
                return None
            if self._cooldown_loses_to_reference():
                return self._restore_reference_block(
                    reason="target_ar_not_faster_than_reference"
                )
            return self._begin_reprobe()

        if block_len <= 1 or self.latency_locked:
            return None

        self.probe_cycles_seen += 1
        self.probe_tokens += commit_count
        if cycle_us > 0.0:
            self.probe_total_us += float(cycle_us)
        if self.probe_cycles_seen < self.config.probe_cycles:
            return None

        tokens_per_cycle = self.probe_tokens / max(1, self.probe_cycles_seen)
        probe_us_per_token = (
            self.probe_total_us / self.probe_tokens
            if self.probe_total_us > 0.0 and self.probe_tokens > 0
            else None
        )
        self.last_probe_tokens_per_cycle = tokens_per_cycle
        self.last_probe_us_per_token = probe_us_per_token
        probe_cycles = self.probe_cycles_seen
        probe_tokens = self.probe_tokens
        self._reset_probe()

        if self._smaller_probe_loses_to_reference(block_len, probe_us_per_token):
            return self._restore_reference_block(
                reason="reprobe_not_faster_than_reference"
            )
        if block_len < self.initial_block_tokens and probe_us_per_token is not None:
            self.reference_block_tokens = block_len
            self.reference_us_per_token = probe_us_per_token

        if tokens_per_cycle >= self.config.min_tokens_per_cycle or not can_continue:
            self.bad_probe_windows_seen = 0
            return None

        # Require sustained bad probe windows before changing policy.
        self.bad_probe_windows_seen += 1
        if self.bad_probe_windows_seen < max(1, int(self.config.bad_probe_windows)):
            return None

        next_block_tokens = self._next_reprobe_block_tokens()
        if next_block_tokens >= self.active_block_tokens:
            return None
        self.bad_probe_windows_seen = 0
        if probe_us_per_token is not None:
            self.reference_block_tokens = self.active_block_tokens
            self.reference_us_per_token = probe_us_per_token
        self.mode = "cooldown"
        self.cooldown_remaining = max(1, int(self.config.cooldown_tokens))
        self.pending_reprobe_block_tokens = next_block_tokens
        self.fallback_count += 1
        self.fallback_reason = (
            f"tokens_per_cycle={tokens_per_cycle:.3f} "
            f"< {self.config.min_tokens_per_cycle:.3f}"
        )
        decision = AdaptiveDecision(
            action="fallback",
            reason=self.fallback_reason,
            tokens_per_cycle=tokens_per_cycle,
            probe_cycles=probe_cycles,
            probe_tokens=probe_tokens,
            cooldown_tokens=self.cooldown_remaining,
            block_tokens=self.active_block_tokens,
            next_block_tokens=next_block_tokens,
            fallback_count=self.fallback_count,
            reprobe_count=self.reprobe_count,
        )
        self.decisions.append(decision)
        return decision

    def summary_fields(self) -> dict[str, object]:
        return {
            "adaptive_fallback_enabled": bool(self.config.enabled),
            "adaptive_fallback_triggered": self.fallback_count > 0,
            "adaptive_fallback_count": int(self.fallback_count),
            "adaptive_reprobe_count": int(self.reprobe_count),
            "adaptive_fallback_tokens": int(self.fallback_tokens),
            "adaptive_fallback_reason": self.fallback_reason,
            "adaptive_min_tokens_per_cycle": float(self.config.min_tokens_per_cycle),
            "adaptive_probe_window": int(self.config.probe_cycles),
            "adaptive_bad_probe_windows": int(self.config.bad_probe_windows),
            "adaptive_cooldown_tokens": int(self.config.cooldown_tokens),
            "adaptive_latency_margin": float(self.config.latency_margin),
            "adaptive_initial_block_tokens": int(self.initial_block_tokens),
            "adaptive_final_block_tokens": int(self.active_block_tokens),
            "adaptive_last_probe_tokens_per_cycle": float(self.last_probe_tokens_per_cycle),
            "adaptive_last_probe_ms_per_token": (
                self.last_probe_us_per_token / 1_000.0
                if self.last_probe_us_per_token is not None
                else None
            ),
            "adaptive_ar_ms_per_token": (
                self.ar_us_per_token / 1_000.0
                if self.ar_us_per_token is not None
                else None
            ),
            "adaptive_reference_block_tokens": self.reference_block_tokens,
            "adaptive_reference_ms_per_token": (
                self.reference_us_per_token / 1_000.0
                if self.reference_us_per_token is not None
                else None
            ),
            "adaptive_latency_reject_count": int(self.latency_reject_count),
            "adaptive_latency_locked": bool(self.latency_locked),
            "adaptive_pending_bad_probe_windows": int(self.bad_probe_windows_seen),
            "adaptive_pending_probe_cycles": int(self.probe_cycles_seen),
            "adaptive_pending_probe_tokens": int(self.probe_tokens),
            "adaptive_events": [
                {k: v for k, v in decision.as_event().items() if k != "event"}
                for decision in self.decisions
            ],
        }

    def _begin_reprobe(self) -> AdaptiveDecision:
        self._finish_cooldown_observation()
        next_block_tokens = max(
            1,
            min(
                int(self.pending_reprobe_block_tokens or self.active_block_tokens),
                self.initial_block_tokens,
            ),
        )
        previous_block_tokens = self.active_block_tokens
        self.active_block_tokens = next_block_tokens
        self.pending_reprobe_block_tokens = None
        self.mode = "probe"
        self.reprobe_count += 1
        self._reset_probe()
        decision = AdaptiveDecision(
            action="reprobe",
            reason="target_ar_cooldown_complete",
            tokens_per_cycle=self.last_probe_tokens_per_cycle,
            probe_cycles=0,
            probe_tokens=0,
            cooldown_tokens=0,
            block_tokens=previous_block_tokens,
            next_block_tokens=self.active_block_tokens,
            fallback_count=self.fallback_count,
            reprobe_count=self.reprobe_count,
        )
        self.decisions.append(decision)
        return decision

    def _reset_probe(self) -> None:
        self.probe_cycles_seen = 0
        self.probe_tokens = 0
        self.probe_total_us = 0.0

    def _finish_cooldown_observation(self) -> None:
        if self.cooldown_total_us > 0.0 and self.cooldown_observed_tokens > 0:
            self.ar_us_per_token = (
                self.cooldown_total_us / self.cooldown_observed_tokens
            )
        self.cooldown_total_us = 0.0
        self.cooldown_observed_tokens = 0

    def _cooldown_loses_to_reference(self) -> bool:
        self._finish_cooldown_observation()
        if self.ar_us_per_token is None or self.reference_us_per_token is None:
            return False
        return not self._latency_beats_reference(self.ar_us_per_token)

    def _smaller_probe_loses_to_reference(
        self,
        block_len: int,
        probe_us_per_token: Optional[float],
    ) -> bool:
        if block_len >= self.initial_block_tokens:
            return False
        if probe_us_per_token is None or self.reference_us_per_token is None:
            return False
        return not self._latency_beats_reference(probe_us_per_token)

    def _latency_beats_reference(self, us_per_token: float) -> bool:
        assert self.reference_us_per_token is not None
        return us_per_token <= (
            self.reference_us_per_token * float(self.config.latency_margin)
        )

    def _restore_reference_block(self, *, reason: str) -> AdaptiveDecision:
        previous_block_tokens = self.active_block_tokens
        restored_block_tokens = max(
            1,
            min(
                int(self.reference_block_tokens or self.initial_block_tokens),
                self.initial_block_tokens,
            ),
        )
        self.active_block_tokens = restored_block_tokens
        self.pending_reprobe_block_tokens = None
        self.cooldown_remaining = 0
        self.mode = "probe"
        self.latency_locked = True
        self.latency_reject_count += 1
        self._reset_probe()
        decision = AdaptiveDecision(
            action="restore",
            reason=reason,
            tokens_per_cycle=self.last_probe_tokens_per_cycle,
            probe_cycles=0,
            probe_tokens=0,
            cooldown_tokens=0,
            block_tokens=previous_block_tokens,
            next_block_tokens=self.active_block_tokens,
            fallback_count=self.fallback_count,
            reprobe_count=self.reprobe_count,
        )
        self.decisions.append(decision)
        return decision

    def _next_reprobe_block_tokens(self) -> int:
        configured = self.config.reprobe_block_tokens
        if configured is not None:
            return max(1, min(int(configured), self.initial_block_tokens))
        if self.active_block_tokens <= 2:
            return self.active_block_tokens
        return max(2, self.active_block_tokens // 2)


def resolve_adaptive_fallback_config(
    env: Mapping[str, str],
) -> AdaptiveFallbackConfig:
    enabled_raw = str(env.get("DFLASH_ADAPTIVE_FALLBACK", "")).strip().lower()
    enabled = enabled_raw not in _FALSE_VALUES
    if not enabled:
        return AdaptiveFallbackConfig(enabled=False)
    return AdaptiveFallbackConfig(
        enabled=True,
        min_tokens_per_cycle=_float_env(
            env,
            "DFLASH_ADAPTIVE_MIN_TOKENS_PER_CYCLE",
            2.0,
            min_value=1.0,
        ),
        probe_cycles=_int_env(env, "DFLASH_ADAPTIVE_PROBE_CYCLES", 8, min_value=1),
        bad_probe_windows=_int_env(
            env,
            "DFLASH_ADAPTIVE_BAD_PROBE_WINDOWS",
            2,
            min_value=1,
        ),
        cooldown_tokens=_int_env(
            env,
            "DFLASH_ADAPTIVE_TARGET_AR_COOLDOWN",
            4096,
            min_value=1,
        ),
        latency_margin=_float_env(
            env,
            "DFLASH_ADAPTIVE_LATENCY_MARGIN",
            1.0,
            min_value=0.0,
        ),
        reprobe_block_tokens=_optional_int_env(
            env,
            "DFLASH_ADAPTIVE_REPROBE_BLOCK_TOKENS",
            min_value=1,
        ),
    )


def _int_env(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    min_value: int,
) -> int:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(int(min_value), value)


def _optional_int_env(
    env: Mapping[str, str],
    name: str,
    *,
    min_value: int,
) -> Optional[int]:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(int(min_value), value)


def _float_env(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    min_value: float,
) -> float:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return max(float(min_value), value)

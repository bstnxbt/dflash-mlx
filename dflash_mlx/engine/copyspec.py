# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from collections.abc import Sequence

COPYSPEC_WINDOW_SIZE = 6

_FNV_OFFSET_BASIS = 14695981039346656037
_FNV_PRIME = 1099511628211
_U64_MASK = (1 << 64) - 1


class CopySpecIndex:
    def __init__(
        self,
        prompt_tokens: Sequence[int],
        *,
        window_size: int = COPYSPEC_WINDOW_SIZE,
    ) -> None:
        if window_size <= 0:
            raise ValueError("copyspec window_size must be positive")
        self.window_size = int(window_size)
        self._tokens = [int(token) for token in prompt_tokens]
        self._index: dict[int, int | list[int]] = {}
        self._enabled = len(self._tokens) > self.window_size
        self._build()

    def draft_after(
        self,
        staged_first: int,
        *,
        max_tokens: int,
        forbidden_tokens: set[int] | None = None,
    ) -> tuple[int, ...] | None:
        max_tokens = int(max_tokens)
        if not self._enabled or max_tokens <= 0:
            return None
        window = self._tail_window(int(staged_first))
        if window is None:
            return None

        best_pos = -1
        best_available = 0
        for source_pos in self._positions_for(_hash_tokens(window)):
            if not self._matches_window(source_pos, window):
                continue
            available = len(self._tokens) - int(source_pos)
            if available > best_available:
                best_pos = int(source_pos)
                best_available = int(available)

        if best_pos < 0 or best_available <= 0:
            return None
        token_count = min(max_tokens, best_available)
        if token_count < max_tokens:
            return None
        copied = tuple(self._tokens[best_pos : best_pos + token_count])
        if forbidden_tokens is not None and any(token in forbidden_tokens for token in copied):
            return None
        return copied

    def append_committed(self, token_ids: Sequence[int]) -> None:
        for token_id in token_ids:
            self._tokens.append(int(token_id))
            if not self._enabled:
                continue
            self._index_latest_window()

    def _build(self) -> None:
        if not self._enabled or len(self._tokens) < self.window_size:
            return
        for start in range(0, len(self._tokens) - self.window_size + 1):
            self._index_window(start)

    def _index_latest_window(self) -> None:
        if len(self._tokens) < self.window_size:
            return
        self._index_window(len(self._tokens) - self.window_size)

    def _index_window(self, start: int) -> None:
        end = int(start) + self.window_size
        token_hash = _hash_tokens(self._tokens[start:end])
        previous = self._index.get(token_hash)
        if previous is None:
            self._index[token_hash] = end
        elif isinstance(previous, int):
            self._index[token_hash] = [previous, end]
        else:
            previous.append(end)

    def _tail_window(self, staged_first: int) -> tuple[int, ...] | None:
        if len(self._tokens) + 1 < self.window_size:
            return None
        if self.window_size == 1:
            return (int(staged_first),)
        return tuple(self._tokens[-(self.window_size - 1) :]) + (int(staged_first),)

    def _matches_window(self, source_pos: int, window: tuple[int, ...]) -> bool:
        start = int(source_pos) - self.window_size
        if start < 0 or source_pos > len(self._tokens):
            return False
        return tuple(self._tokens[start:source_pos]) == window

    def _positions_for(self, token_hash: int) -> tuple[int, ...] | list[int]:
        positions = self._index.get(token_hash)
        if positions is None:
            return ()
        if isinstance(positions, int):
            return (positions,)
        return positions


COPYSPEC_PROBE_OFF_CYCLES = 24
COPYSPEC_PROBE_ON_MIN_COPY = 6
COPYSPEC_PROBE_ON_MAX_CYCLES = 48
COPYSPEC_LATCH_CYCLES = 200
COPYSPEC_MARGIN = 1.0
COPYSPEC_BACKOFF_CAP = 4  # max dormant-period doublings (latch_cycles * 2^cap)


class CopySpecAutoGate:
    """Windowed A/B latch that decides per-cycle whether copyspec should draft.

    Four-phase state machine:
      measure_off  → gather a baseline (copyspec disabled)
      measure_on   → probe with copyspec enabled
      engaged      → copyspec is winning; stay on for latch_cycles
      dormant      → copyspec is losing; stay off for latch_cycles
    """

    def __init__(
        self,
        *,
        probe_off_cycles: int = COPYSPEC_PROBE_OFF_CYCLES,
        probe_on_min_copy: int = COPYSPEC_PROBE_ON_MIN_COPY,
        probe_on_max_cycles: int = COPYSPEC_PROBE_ON_MAX_CYCLES,
        latch_cycles: int = COPYSPEC_LATCH_CYCLES,
        margin: float = COPYSPEC_MARGIN,
        backoff_cap: int = COPYSPEC_BACKOFF_CAP,
    ) -> None:
        if probe_off_cycles < 1:
            raise ValueError("probe_off_cycles must be >= 1")
        if probe_on_min_copy < 1:
            raise ValueError("probe_on_min_copy must be >= 1")
        if probe_on_max_cycles < 1:
            raise ValueError("probe_on_max_cycles must be >= 1")
        if latch_cycles < 1:
            raise ValueError("latch_cycles must be >= 1")
        if margin <= 0:
            raise ValueError("margin must be > 0")
        if backoff_cap < 0:
            raise ValueError("backoff_cap must be >= 0")

        self._probe_off_cycles = probe_off_cycles
        self._probe_on_min_copy = probe_on_min_copy
        self._probe_on_max_cycles = probe_on_max_cycles
        self._latch_cycles = latch_cycles
        self._margin = margin
        self._backoff_cap = backoff_cap

        # Phase
        self.phase: str = "measure_off"

        # Window accumulators
        self._win_commits: int = 0
        self._win_cost_ns: int = 0
        self._win_cycles: int = 0
        self._win_copy_blocks: int = 0

        # State
        self._off_rate: float | None = None
        self._latch_remaining: int = 0
        self._pending_reset: bool = False
        # Consecutive disengages → exponentially longer dormant periods, so a
        # workload where copyspec never wins stops paying repeated probe cost
        # (each probe engages copyspec briefly and is wasted there). Reset to
        # 0 on any engage, so a workload that turns copy-heavy stays responsive.
        self._consecutive_losses: int = 0

        # Metrics
        self._engaged_cycles: int = 0
        self._dormant_cycles: int = 0
        self._probes: int = 0
        self._engages: int = 0
        self._disengages: int = 0

    def engage_copy(self) -> bool:
        """Return True iff copyspec should draft this cycle."""
        return self.phase in ("measure_on", "engaged")

    def record_cycle(
        self,
        *,
        source: str,
        commit_count: int,
        cycle_cost_ns: int | None,
    ) -> None:
        """Called once per cycle that ran a real draft (block_len > 1)."""
        # Accumulate
        self._win_commits += commit_count
        if isinstance(cycle_cost_ns, int) and cycle_cost_ns > 0:
            self._win_cost_ns += cycle_cost_ns
        self._win_cycles += 1
        if source == "copy":
            self._win_copy_blocks += 1

        # Advance phase machine
        if self.phase == "measure_off":
            if self._win_cycles >= self._probe_off_cycles:
                self._off_rate = self._win_rate()
                self._reset_window()
                self.phase = "measure_on"
                self._probes += 1

        elif self.phase == "measure_on":
            ended = (
                self._win_copy_blocks >= self._probe_on_min_copy
                or self._win_cycles >= self._probe_on_max_cycles
            )
            if ended:
                on_rate = self._win_rate()
                copy_seen = self._win_copy_blocks
                self._reset_window()
                if (
                    copy_seen == 0
                    or self._off_rate is None
                    or on_rate < self._off_rate * self._margin
                ):
                    self.phase = "dormant"
                    self._latch_remaining = self._latch_cycles * (
                        2 ** min(self._consecutive_losses, self._backoff_cap)
                    )
                    self._consecutive_losses += 1
                    self._pending_reset = True
                    self._disengages += 1
                else:
                    self.phase = "engaged"
                    self._latch_remaining = self._latch_cycles
                    self._consecutive_losses = 0
                    self._engages += 1

        elif self.phase == "engaged":
            self._engaged_cycles += 1
            self._latch_remaining -= 1
            if self._latch_remaining <= 0:
                self._pending_reset = True
                self.phase = "measure_off"
                self._reset_window()

        elif self.phase == "dormant":
            self._dormant_cycles += 1
            self._latch_remaining -= 1
            if self._latch_remaining <= 0:
                self.phase = "measure_off"
                self._reset_window()

    def take_reset(self) -> bool:
        """Return True if the adaptive policy should be reset, then clear the flag."""
        result = self._pending_reset
        self._pending_reset = False
        return result

    def _win_rate(self) -> float:
        """Realized throughput for the current window."""
        if self._win_cost_ns > 0:
            return self._win_commits / (self._win_cost_ns / 1e9)
        if self._win_cycles > 0:
            return self._win_commits / self._win_cycles
        return 0.0

    def _reset_window(self) -> None:
        self._win_commits = 0
        self._win_cost_ns = 0
        self._win_cycles = 0
        self._win_copy_blocks = 0

    def metrics(self) -> dict:
        return {
            "state": self.phase,
            "off_rate": self._off_rate,
            "engaged_cycles": int(self._engaged_cycles),
            "dormant_cycles": int(self._dormant_cycles),
            "probes": int(self._probes),
            "engages": int(self._engages),
            "disengages": int(self._disengages),
        }


def _hash_tokens(tokens: Sequence[int]) -> int:
    value = _FNV_OFFSET_BASIS
    for token in tokens:
        value ^= int(token) & _U64_MASK
        value = (value * _FNV_PRIME) & _U64_MASK
    return value

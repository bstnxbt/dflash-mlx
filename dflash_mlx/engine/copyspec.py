# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from collections.abc import Sequence

COPYSPEC_WINDOW_SIZE = 6
COPYSPEC_MISS_LIMIT = 3
COPYSPEC_COOLDOWN_BLOCKS = 64

_FNV_OFFSET_BASIS = 14695981039346656037
_FNV_PRIME = 1099511628211
_U64_MASK = (1 << 64) - 1


class CopySpecGate:
    """Tolerance gate deciding whether copyspec drafting may run.

    Replaces the previous one-strike permanent disable. Copyspec pauses
    only after ``miss_limit`` consecutive zero-acceptance copy blocks,
    and then only for ``cooldown_blocks`` draft cycles before probing
    again. An accepted copy block fully restores the miss tolerance, so
    a few coincidental window matches early in a generation (e.g. during
    a reasoning phase) no longer forfeit copy drafting for the rest of
    the request when the output later turns copy-heavy.
    """

    def __init__(
        self,
        *,
        miss_limit: int = COPYSPEC_MISS_LIMIT,
        cooldown_blocks: int = COPYSPEC_COOLDOWN_BLOCKS,
    ) -> None:
        if miss_limit <= 0:
            raise ValueError("copyspec miss_limit must be positive")
        if cooldown_blocks <= 0:
            raise ValueError("copyspec cooldown_blocks must be positive")
        self.miss_limit = int(miss_limit)
        self.cooldown_blocks = int(cooldown_blocks)
        self._consecutive_misses = 0
        self._cooldown_remaining = 0

    @property
    def enabled(self) -> bool:
        """Non-consuming view of the gate state (for tests/diagnostics)."""
        return self._cooldown_remaining == 0

    def should_attempt(self) -> bool:
        """Whether copyspec may draft this cycle. Ticks the cooldown."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False
        return True

    def record_block(self, accepted_tokens: int) -> None:
        if int(accepted_tokens) > 0:
            self._consecutive_misses = 0
            return
        self._consecutive_misses += 1
        if self._consecutive_misses >= self.miss_limit:
            self._cooldown_remaining = self.cooldown_blocks
            # Leave one remaining strike so a failed post-cooldown probe
            # re-enters cooldown immediately instead of burning the full
            # miss budget again.
            self._consecutive_misses = self.miss_limit - 1


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


def _hash_tokens(tokens: Sequence[int]) -> int:
    value = _FNV_OFFSET_BASIS
    for token in tokens:
        value ^= int(token) & _U64_MASK
        value = (value * _FNV_PRIME) & _U64_MASK
    return value

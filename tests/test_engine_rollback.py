from __future__ import annotations

from types import SimpleNamespace

from dflash_mlx.engine.rollback import (
    arm_target_rollback_with_prefix,
    cleanup_generation_caches,
    clear_rollback_state,
    restore_target_cache_after_acceptance,
)


class _ArmableCache:
    def __init__(self):
        self.armed_at: list[int] = []
        self.rolled_back_to: list[int] = []
        self._tape = "x"
        self._tape_k = "y"
        self._tape_g = "z"
        self._tape_qkv = "q"
        self._snapshot = "s"
        self._armed = True

    def arm_rollback(self, *, prefix_len: int) -> None:
        self.armed_at.append(int(prefix_len))

    def rollback(self, accepted: int) -> None:
        self.rolled_back_to.append(int(accepted))

    def clear_transients(self) -> None:
        self._tape = None
        self._tape_k = None
        self._tape_g = None
        self._tape_qkv = None
        self._snapshot = None
        self._armed = False


class _OffsetCache:
    def __init__(self, offset: int):
        self.offset = int(offset)
        self.trim_calls: list[int] = []

    def trim(self, n: int) -> None:
        self.trim_calls.append(int(n))
        self.offset -= int(n)


class _CropCache:
    def __init__(self):
        self.crop_calls: list[int] = []

    def crop(self, target_len: int) -> None:
        self.crop_calls.append(int(target_len))


def test_arm_calls_arm_rollback_with_int_prefix():
    a, b = _ArmableCache(), _ArmableCache()
    arm_target_rollback_with_prefix([a, b, "not_armable"], prefix_len=42)
    assert a.armed_at == [42]
    assert b.armed_at == [42]


def test_clear_rollback_state_uses_clear_transients_when_present():
    c = _ArmableCache()
    clear_rollback_state(c)
    assert c._armed is False
    assert c._tape is None and c._tape_k is None and c._snapshot is None


def test_clear_rollback_state_falls_back_to_attribute_writes():
    obj = SimpleNamespace(
        _armed=True, _tape=1, _tape_k=2, _tape_g=3, _tape_qkv=4, _snapshot=5
    )
    clear_rollback_state(obj)
    assert obj._armed is False
    assert obj._tape is None and obj._snapshot is None


def test_full_acceptance_skips_rollback_replay():
    c = _ArmableCache()
    replay_ns = restore_target_cache_after_acceptance(
        [c],
        target_len=100,
        acceptance_length=4,
        drafted_tokens=4,
    )
    assert c.rolled_back_to == []
    assert c._tape is None
    assert replay_ns == 0


def test_force_replay_overrides_full_acceptance_fast_path():
    c = _ArmableCache()
    replay_ns = restore_target_cache_after_acceptance(
        [c],
        target_len=100,
        acceptance_length=0,
        drafted_tokens=0,
        force_replay=True,
    )

    assert c.rolled_back_to == [0]
    assert replay_ns >= 0


def test_partial_acceptance_calls_rollback_with_acceptance_length():
    c = _ArmableCache()
    replay_ns = restore_target_cache_after_acceptance(
        [c],
        target_len=100,
        acceptance_length=2,
        drafted_tokens=4,
    )
    assert c.rolled_back_to == [2]
    assert replay_ns >= 0


def test_offset_trim_path_when_offset_exceeds_target_len():
    c = _OffsetCache(offset=110)
    restore_target_cache_after_acceptance(
        [c],
        target_len=100,
        acceptance_length=0,
        drafted_tokens=0,
    )
    assert c.trim_calls == [10]


def test_offset_no_trim_when_already_within_target():
    c = _OffsetCache(offset=80)
    restore_target_cache_after_acceptance(
        [c],
        target_len=100,
        acceptance_length=0,
        drafted_tokens=0,
    )
    assert c.trim_calls == []


def test_crop_fallback_path():
    c = _CropCache()
    restore_target_cache_after_acceptance(
        [c],
        target_len=64,
        acceptance_length=0,
        drafted_tokens=0,
    )
    assert c.crop_calls == [64]


def test_cleanup_clears_target_transients_and_empties_lists():
    c = _ArmableCache()
    target = [c]
    draft = ["draft_entry"]
    cleanup_generation_caches(target, draft)
    assert target == []
    assert draft == []
    assert c._tape is None

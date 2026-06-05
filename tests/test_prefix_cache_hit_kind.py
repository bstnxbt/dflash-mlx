# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""Tests for hit_kind classification in PrefixCacheLookupResult.

Design: hit_kind is exposed only through the public RuntimeCacheManager.lookup()
API (which returns PrefixCacheLookupResult).  Internal _l1.lookup() and
_store.lookup() keep their existing 2-tuple signatures — they do not return
hit_kind.  The classification is side-effected into _last_hit_kind state that
the manager reads after the lookup under _state_lock.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from dflash_mlx.cache.codecs import build_snapshot
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.manager import HitKind, PrefixCacheLookupResult, RuntimeCacheManager
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.store import PrefixSnapshotStore
from mlx_lm.models.cache import KVCache


def _make_key() -> DFlashPrefixKey:
    return DFlashPrefixKey(
        target_model_id="test/target-v1",
        draft_model_id="test/draft-v1",
        capture_layer_ids=(10, 20),
        draft_sink_size=16,
        draft_window_size=2048,
        template_hash="a" * 64,
        prompt_policy_hash="b" * 64,
    )


def _make_snapshot(
    token_ids: list[int],
    key: DFlashPrefixKey,
    *,
    kind: str = "prefill",
    with_logits: bool = True,
    hidden_dim: int = 8,
    vocab: int = 32,
):
    n = max(1, len(token_ids))
    kv = KVCache()
    kv.keys = mx.zeros((1, 2, n, hidden_dim), dtype=mx.float32)
    kv.values = mx.zeros((1, 2, n, hidden_dim), dtype=mx.float32)
    kv.offset = n
    mx.eval(kv.keys, kv.values)
    target_hidden = mx.zeros((1, n, hidden_dim), dtype=mx.float32)
    last_logits = mx.zeros((1, vocab), dtype=mx.float32) if with_logits else None
    snap = build_snapshot(
        token_ids=token_ids,
        target_cache=[kv],
        target_hidden=target_hidden,
        last_logits=last_logits,
        key=key,
        kind=kind,
    )
    return snap


def _make_manager(max_entries: int = 8) -> RuntimeCacheManager:
    l1 = DFlashPrefixCache(max_entries=max_entries, max_bytes=8 * 1024**3)
    store = PrefixSnapshotStore(l1=l1)
    return RuntimeCacheManager(store)


class TestHitKindTypeSystem:
    """The HitKind Literal covers all possible values."""

    def test_hit_kind_literal_values(self):
        assert set(HitKind.__args__) == {
            "miss", "l1_exact", "l1_prefix", "l2_exact", "l2_prefix"
        }

    def test_result_is_dataclass_not_tuple(self):
        mgr = _make_manager()
        result = mgr.lookup([1, 2, 3], _make_key())
        assert isinstance(result, PrefixCacheLookupResult)
        # Must NOT be unpackable as a tuple (keeps old 2-tuple callers safe).
        with pytest.raises((TypeError, ValueError)):
            _, _ = result  # type: ignore[misc]

    def test_default_hit_kind_is_miss(self):
        result = PrefixCacheLookupResult(matched_tokens=0, snapshot=None, elapsed_ms=0.0)
        assert result.hit_kind == "miss"


class TestHitKindL1:
    """hit_kind classification with L1-only cache."""

    def test_cold_miss(self):
        mgr = _make_manager()
        result = mgr.lookup([1, 2, 3], _make_key())
        assert result.hit_kind == "miss"
        assert result.matched_tokens == 0

    def test_prefix_hit(self):
        mgr = _make_manager()
        key = _make_key()
        snap = _make_snapshot([1, 2, 3], key, kind="prefill", with_logits=True)
        mgr._store._l1.insert(snap)
        result = mgr.lookup([1, 2, 3, 4, 5], key)
        assert result.hit_kind == "l1_prefix"
        assert result.matched_tokens == 3

    def test_exact_hit(self):
        mgr = _make_manager()
        key = _make_key()
        snap = _make_snapshot([1, 2, 3], key, kind="prefill", with_logits=True)
        mgr._store._l1.insert(snap)
        result = mgr.lookup([1, 2, 3], key)
        assert result.hit_kind == "l1_exact"
        assert result.matched_tokens == 3

    def test_sequential_lookups_are_independent(self):
        mgr = _make_manager()
        key = _make_key()
        snap = _make_snapshot([1, 2, 3], key, kind="prefill", with_logits=True)
        mgr._store._l1.insert(snap)

        assert mgr.lookup([1, 2, 3], key).hit_kind == "l1_exact"
        assert mgr.lookup([9, 9, 9], key).hit_kind == "miss"
        assert mgr.lookup([1, 2, 3, 4, 5], key).hit_kind == "l1_prefix"

    def test_probe_does_not_overwrite_last_hit_kind(self):
        """Internal record=False probes must not change _last_hit_kind."""
        mgr = _make_manager()
        key = _make_key()
        snap = _make_snapshot([1, 2, 3], key, kind="prefill", with_logits=True)
        mgr._store._l1.insert(snap)

        # Establish a recorded hit.
        mgr.lookup([1, 2, 3], key)
        assert mgr._store._l1._last_hit_kind == "l1_exact"

        # Probe (record=False) must not overwrite state.
        mgr._store._l1.lookup([1, 2, 3], key, record=False)
        assert mgr._store._l1._last_hit_kind == "l1_exact"

    def test_generation_snapshot_without_logits_reports_prefix_not_exact(self):
        """Generation snapshot without last_logits can only serve as a prefix hit."""
        mgr = _make_manager()
        key = _make_key()
        snap = _make_snapshot([1, 2, 3], key, kind="generation", with_logits=False)
        mgr._store._l1.insert(snap)
        result = mgr.lookup([1, 2, 3], key)
        # Without logits the snapshot is not usable for exact-match serving.
        assert result.hit_kind in ("miss", "l1_prefix")
        assert result.matched_tokens in (0, 3)

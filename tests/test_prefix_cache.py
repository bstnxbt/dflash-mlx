# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from __future__ import annotations

import threading

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from dflash_mlx.cache.codecs import (
    build_snapshot,
    hydrate_target_cache,
    serialize_target_cache,
    target_cache_is_serializable,
)
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
from dflash_mlx.server.prefix_cache_flow import compute_stable_prefix_len

def _make_kv_cache_populated(n_tokens: int = 4, hkv: int = 2, d: int = 8) -> KVCache:
    cache = KVCache()
    keys = mx.arange(1 * hkv * n_tokens * d, dtype=mx.float32).reshape(1, hkv, n_tokens, d)
    vals = (mx.arange(1 * hkv * n_tokens * d, dtype=mx.float32) + 1000.0).reshape(1, hkv, n_tokens, d)
    cache.keys = keys
    cache.values = vals
    cache.offset = n_tokens
    mx.eval(cache.keys, cache.values)
    return cache

def _make_gdn_cache_populated(size: int = 3, conv_k: int = 4) -> RecurrentRollbackCache:
    cache = RecurrentRollbackCache(size=size, conv_kernel_size=conv_k)
    cache.cache[0] = mx.arange(12, dtype=mx.float32).reshape(1, conv_k, 3)
    cache.cache[1] = (mx.arange(24, dtype=mx.float32) + 100.0).reshape(1, 2, 4, 3)
    mx.eval(cache.cache[0], cache.cache[1])
    return cache

def _make_mixed_template(n_fa: int = 2, n_gdn: int = 2) -> list:
    out = []
    for _ in range(n_fa):
        out.append(_make_kv_cache_populated())
    for _ in range(n_gdn):
        out.append(_make_gdn_cache_populated())
    return out

def _make_key(**overrides) -> DFlashPrefixKey:
    base = dict(
        target_model_id="test/target-v1",
        draft_model_id="test/draft-v1",
        capture_layer_ids=(10, 20),
        draft_sink_size=16,
        draft_window_size=2048,
    )
    base.update(overrides)
    return DFlashPrefixKey(**base)

def _make_synthetic_snapshot(
    token_ids: list[int],
    key: DFlashPrefixKey,
    hidden_dim: int = 8,
    vocab: int = 32,
    kind: str = "prefill",
) -> DFlashPrefixSnapshot:
    prefix_len = len(token_ids)
    kv_cache = _make_kv_cache_populated(n_tokens=max(1, prefix_len))
    gdn_cache = _make_gdn_cache_populated()
    target_hidden = mx.zeros((1, prefix_len, hidden_dim), dtype=mx.float32)
    last_logits = mx.zeros((1, vocab), dtype=mx.float32)
    return build_snapshot(
        token_ids=token_ids,
        target_cache=[kv_cache, gdn_cache],
        target_hidden=target_hidden,
        last_logits=last_logits,
        key=key,
        kind=kind,
    )

def test_l1_generation_snapshot_without_logits_is_prefix_only():
    cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
    key = _make_key()
    snapshot = _make_synthetic_snapshot([1, 2, 3], key, kind="generation")
    snapshot.last_logits = None
    cache.insert(snapshot)

    exact_len, exact = cache.lookup([1, 2, 3], key)
    assert exact_len == 0
    assert exact is None

    prefix_len, prefix = cache.lookup([1, 2, 3, 4], key)
    assert prefix_len == 3
    assert prefix is snapshot

class TestSerializeHydrate:
    def test_kv_only_round_trip(self):
        src = [_make_kv_cache_populated(n_tokens=5)]
        assert target_cache_is_serializable(src)
        fa, gdn = serialize_target_cache(src)
        assert fa[0] is not None
        assert gdn[0] is None
        k, v, offset = fa[0]
        assert offset == 5

        template = [KVCache()]
        snapshot = DFlashPrefixSnapshot.from_full_target_hidden(
            token_ids=(1, 2, 3, 4, 5),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 5, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        hydrated = hydrate_target_cache(snapshot, template)
        assert isinstance(hydrated[0], KVCache)
        assert hydrated[0].offset == 5

        src_k, src_v = src[0].state
        h_k, h_v = hydrated[0].state
        assert mx.all(src_k == h_k).item()
        assert mx.all(src_v == h_v).item()

    def test_gdn_only_round_trip(self):
        src = [_make_gdn_cache_populated(size=3, conv_k=4)]
        assert target_cache_is_serializable(src)
        fa, gdn = serialize_target_cache(src)
        assert fa[0] is None
        assert gdn[0] is not None
        assert len(gdn[0]) == 3
        template = [RecurrentRollbackCache(size=3, conv_kernel_size=4)]
        snapshot = DFlashPrefixSnapshot.from_full_target_hidden(
            token_ids=(7, 8, 9),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 3, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        hydrated = hydrate_target_cache(snapshot, template)
        assert isinstance(hydrated[0], RecurrentRollbackCache)
        assert hydrated[0].conv_kernel_size == 4
        for a, b in zip(src[0].cache, hydrated[0].cache):
            if a is None:
                assert b is None
            else:
                assert mx.all(a == b).item()

    def test_mixed_round_trip(self):
        src = _make_mixed_template(n_fa=2, n_gdn=2)
        assert target_cache_is_serializable(src)
        fa, gdn = serialize_target_cache(src)

        assert fa[0] is not None and gdn[0] is None
        assert fa[1] is not None and gdn[1] is None
        assert fa[2] is None and gdn[2] is not None
        assert fa[3] is None and gdn[3] is not None

        template = [KVCache(), KVCache(),
                    RecurrentRollbackCache(size=3, conv_kernel_size=4),
                    RecurrentRollbackCache(size=3, conv_kernel_size=4)]
        snapshot = DFlashPrefixSnapshot.from_full_target_hidden(
            token_ids=(1, 2),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 2, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        hydrated = hydrate_target_cache(snapshot, template)
        assert len(hydrated) == 4
        assert isinstance(hydrated[0], KVCache)
        assert isinstance(hydrated[2], RecurrentRollbackCache)

    def test_unknown_cache_type_rejected(self):
        class WeirdCache:
            pass
        src = [WeirdCache()]
        assert target_cache_is_serializable(src) is False
        with pytest.raises(TypeError):
            serialize_target_cache(src)

    def test_hydrate_size_mismatch_raises(self):
        src = [_make_kv_cache_populated()]
        fa, gdn = serialize_target_cache(src)
        snapshot = DFlashPrefixSnapshot.from_full_target_hidden(
            token_ids=(1,),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 1, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        template = [KVCache(), KVCache()]
        with pytest.raises(ValueError, match="Template cache length"):
            hydrate_target_cache(snapshot, template)

    def test_hydrate_type_mismatch_raises(self):
        src = [_make_kv_cache_populated()]
        fa, gdn = serialize_target_cache(src)
        snapshot = DFlashPrefixSnapshot.from_full_target_hidden(
            token_ids=(1,),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 1, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        template = [RecurrentRollbackCache(size=3, conv_kernel_size=4)]
        with pytest.raises(ValueError, match="Snapshot missing GDN state"):
            hydrate_target_cache(snapshot, template)

class TestMutationIsolation:
    def test_kv_clone_is_independent(self):
        src = [_make_kv_cache_populated(n_tokens=3)]
        fa, gdn = serialize_target_cache(src)

        src[0].keys = mx.zeros_like(src[0].keys)
        mx.eval(src[0].keys)

        snap_k, _, _ = fa[0]
        assert not mx.all(snap_k == 0).item(), \
            "Snapshot shares buffer with live cache — deep-copy is broken"

    def test_gdn_clone_is_independent(self):
        src = [_make_gdn_cache_populated(size=3, conv_k=4)]
        fa, gdn = serialize_target_cache(src)

        src[0].cache[1] = mx.zeros_like(src[0].cache[1])
        mx.eval(src[0].cache[1])

        assert not mx.all(gdn[0][1] == 0).item(), \
            "GDN snapshot shares buffer with live cache — deep-copy is broken"

    def test_hydrate_returns_independent_cache(self):
        src = [_make_kv_cache_populated(n_tokens=3)]
        fa, gdn = serialize_target_cache(src)
        snapshot = DFlashPrefixSnapshot.from_full_target_hidden(
            token_ids=(1, 2, 3),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 3, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        template = [KVCache()]
        hydrated1 = hydrate_target_cache(snapshot, template)
        hydrated2 = hydrate_target_cache(snapshot, template)

        hydrated1[0].keys = mx.zeros_like(hydrated1[0].keys)
        mx.eval(hydrated1[0].keys)

        h2_k, _ = hydrated2[0].state
        assert not mx.all(h2_k == 0).item()
        snap_k, _, _ = snapshot.fa_states[0]
        assert not mx.all(snap_k == 0).item()

class TestSnapshot:
    def test_prefix_len_matches_token_ids(self):
        snap = _make_synthetic_snapshot(token_ids=[1, 2, 3, 4], key=_make_key())
        assert snap.prefix_len == 4

    def test_nbytes_positive_and_accurate(self):
        snap = _make_synthetic_snapshot(token_ids=[1, 2, 3], key=_make_key())

        target_hidden_bytes = sum(int(c.nbytes) for c in snap.target_hidden_chunks)
        expected_min = target_hidden_bytes + int(snap.last_logits.nbytes)
        assert snap.nbytes >= expected_min

        assert snap.nbytes > 0

class TestLRUBehavior:
    def test_empty_lookup_is_miss(self):
        cache = DFlashPrefixCache(max_entries=4)
        matched, snap = cache.lookup([1, 2, 3], _make_key())
        assert matched == 0
        assert snap is None
        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["exact_hits"] == 0

    def test_exact_hit(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        snap = _make_synthetic_snapshot([10, 20, 30], key)
        cache.insert(snap)
        matched, found = cache.lookup([10, 20, 30], key)
        assert matched == 3
        assert found is snap
        stats = cache.stats()
        assert stats["exact_hits"] == 1
        assert stats["prefix_hits"] == 0

    def test_prefix_hit(self):

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        snap = _make_synthetic_snapshot([10, 20, 30], key)
        cache.insert(snap)
        matched, found = cache.lookup([10, 20, 30, 40, 50], key)
        assert matched == 3
        assert found is snap
        stats = cache.stats()
        assert stats["prefix_hits"] == 1
        assert stats["exact_hits"] == 0

    def test_longer_request_returns_stored_len(self):

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        snap = _make_synthetic_snapshot([10, 20, 30], key)
        cache.insert(snap)
        matched, found = cache.lookup([10, 20, 30, 99, 42], key)
        assert matched == 3
        assert found is snap

    def test_divergent_tokens_refused(self):

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        snap = _make_synthetic_snapshot([10, 20, 30, 40], key)
        cache.insert(snap)
        matched, found = cache.lookup([10, 20, 99, 40], key)
        assert matched == 0
        assert found is None

    def test_request_shorter_than_snapshot_refused(self):

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        snap = _make_synthetic_snapshot([10, 20, 30, 40, 50], key)
        cache.insert(snap)
        matched, found = cache.lookup([10, 20, 30], key)
        assert matched == 0
        assert found is None

    def test_fingerprint_mismatch_is_miss(self):
        cache = DFlashPrefixCache(max_entries=4)
        key_a = _make_key()
        key_b = _make_key(target_model_id="other-target")
        snap = _make_synthetic_snapshot([10, 20, 30], key_a)
        cache.insert(snap)
        matched, found = cache.lookup([10, 20, 30], key_b)
        assert matched == 0
        assert found is None

    def test_longest_of_multiple_matches_wins(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        short = _make_synthetic_snapshot([10, 20], key)
        long = _make_synthetic_snapshot([10, 20, 30, 40], key)
        cache.insert(short)
        cache.insert(long)
        matched, found = cache.lookup([10, 20, 30, 40], key)
        assert matched == 4
        assert found is long

    def test_evicts_on_entries_limit(self):
        cache = DFlashPrefixCache(max_entries=2)
        key = _make_key()
        s1 = _make_synthetic_snapshot([1], key)
        s2 = _make_synthetic_snapshot([2], key)
        s3 = _make_synthetic_snapshot([3], key)
        cache.insert(s1)
        cache.insert(s2)
        cache.insert(s3)
        stats = cache.stats()
        assert stats["current_entries"] == 2
        assert stats["evictions"] >= 1

        matched1, _ = cache.lookup([1], key)
        matched3, _ = cache.lookup([3], key)
        assert matched1 == 0
        assert matched3 == 1

    def test_evicts_on_bytes_limit(self):

        per_entry_nbytes = _make_synthetic_snapshot([1, 2, 3], _make_key()).nbytes
        cache = DFlashPrefixCache(max_entries=1000, max_bytes=per_entry_nbytes)
        key = _make_key()
        cache.insert(_make_synthetic_snapshot([1], key))
        cache.insert(_make_synthetic_snapshot([2], key))
        stats = cache.stats()
        assert stats["current_bytes"] <= per_entry_nbytes + 1
        assert stats["evictions"] >= 1

    def test_lru_touch_on_lookup(self):

        cache = DFlashPrefixCache(max_entries=2)
        key = _make_key()
        s1 = _make_synthetic_snapshot([1], key)
        s2 = _make_synthetic_snapshot([2], key)
        cache.insert(s1)
        cache.insert(s2)
        cache.lookup([1], key)
        s3 = _make_synthetic_snapshot([3], key)
        cache.insert(s3)

        m1, _ = cache.lookup([1], key)
        m2, _ = cache.lookup([2], key)
        m3, _ = cache.lookup([3], key)
        assert m1 == 1 and m3 == 1
        assert m2 == 0

    def test_clear_empties(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        cache.insert(_make_synthetic_snapshot([1, 2], key))
        assert cache.stats()["current_entries"] == 1
        cache.clear()
        assert cache.stats()["current_entries"] == 0
        matched, _ = cache.lookup([1, 2], key)
        assert matched == 0

    def test_stats_counters(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        cache.lookup([1, 2], key)
        cache.insert(_make_synthetic_snapshot([1, 2], key))
        cache.lookup([1, 2], key)
        cache.lookup([1, 2, 3, 4], key)
        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["exact_hits"] == 1
        assert stats["prefix_hits"] == 1
        assert stats["prefill_tokens_saved"] == 2 + 2
        assert stats["insertions"] == 1

class TestConcurrency:
    def test_concurrent_inserts_do_not_crash(self):
        cache = DFlashPrefixCache(max_entries=16)
        key = _make_key()
        errors: list[Exception] = []

        def worker(base: int):
            try:
                for i in range(10):
                    snap = _make_synthetic_snapshot([base, base + i], key)
                    cache.insert(snap)
                    cache.lookup([base, base + i], key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(b,)) for b in (100, 200, 300)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent access raised: {errors}"

class TestStablePrefixLen:

    IM_START = 900
    ASST = 901

    def test_no_markers_returns_full_length(self):

        assert compute_stable_prefix_len([1, 2, 3, 4], im_start_id=None, assistant_id=None) == 4
        assert compute_stable_prefix_len([], im_start_id=None, assistant_id=None) == 0

    def test_no_assistant_marker_returns_full_length(self):

        tokens = [1, 2, 3, self.IM_START, 77, 5]
        assert compute_stable_prefix_len(
            tokens, im_start_id=self.IM_START, assistant_id=self.ASST
        ) == len(tokens)

    def test_trailing_tail_stripped(self):

        tokens = [10, 11, 12, 13, self.IM_START, self.ASST, 100, 200, 300]

        assert compute_stable_prefix_len(
            tokens, im_start_id=self.IM_START, assistant_id=self.ASST
        ) == 4

    def test_last_occurrence_wins(self):

        tokens = [
            10,
            self.IM_START, self.ASST, 20, 21,
            22, 23,
            self.IM_START, self.ASST, 100, 200,
        ]

        assert compute_stable_prefix_len(
            tokens, im_start_id=self.IM_START, assistant_id=self.ASST
        ) == 7

    def test_stripped_matches_across_turns(self):

        turn1 = [10, 11, 12, 13, self.IM_START, self.ASST, 100, 200]

        turn2 = [
            10, 11, 12, 13,
            self.IM_START, self.ASST, 50, 51, 52,
            902, 903,
            self.IM_START, 600, 601,
            self.IM_START, self.ASST, 700, 701,
        ]
        s1 = compute_stable_prefix_len(turn1, im_start_id=self.IM_START, assistant_id=self.ASST)
        s2 = compute_stable_prefix_len(turn2, im_start_id=self.IM_START, assistant_id=self.ASST)
        assert turn1[:s1] == turn2[:s1], "stripped turn 1 should be a prefix of stripped turn 2"

        assert s2 >= s1

    def test_tuple_input(self):

        tokens = (10, 11, self.IM_START, self.ASST, 300)
        assert compute_stable_prefix_len(
            tokens, im_start_id=self.IM_START, assistant_id=self.ASST
        ) == 2

    def test_short_input(self):

        assert compute_stable_prefix_len([], im_start_id=1, assistant_id=2) == 0
        assert compute_stable_prefix_len([5], im_start_id=1, assistant_id=2) == 1

class TestPrefixPruning:
    def test_strict_prefix_pruned_on_insert(self):
        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        short = _make_synthetic_snapshot([1, 2, 3], key)
        longer = _make_synthetic_snapshot([1, 2, 3, 4, 5], key)

        cache.insert(short)
        assert cache.stats()["current_entries"] == 1

        cache.insert(longer)

        assert cache.stats()["current_entries"] == 1
        assert cache.stats()["prefix_prunes"] == 1

        matched_len, snap = cache.lookup([1, 2, 3, 4, 5], key)
        assert matched_len == 5
        assert snap is longer

    def test_equal_token_ids_pruned(self):
        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        first = _make_synthetic_snapshot([7, 8, 9], key)
        second = _make_synthetic_snapshot([7, 8, 9], key)

        cache.insert(first)
        cache.insert(second)

        assert cache.stats()["current_entries"] == 1
        assert cache.stats()["prefix_prunes"] == 1

    def test_unrelated_tokens_not_pruned(self):
        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        a = _make_synthetic_snapshot([1, 2, 3], key)
        b = _make_synthetic_snapshot([9, 8, 7, 6], key)

        cache.insert(a)
        cache.insert(b)

        assert cache.stats()["current_entries"] == 2
        assert cache.stats()["prefix_prunes"] == 0

    def test_different_key_not_pruned(self):
        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key_a = _make_key(target_model_id="model-A")
        key_b = _make_key(target_model_id="model-B")
        short = _make_synthetic_snapshot([1, 2, 3], key_a)
        longer = _make_synthetic_snapshot([1, 2, 3, 4, 5], key_b)

        cache.insert(short)
        cache.insert(longer)

        assert cache.stats()["current_entries"] == 2
        assert cache.stats()["prefix_prunes"] == 0

    def test_longer_then_shorter_no_prune(self):
        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        longer = _make_synthetic_snapshot([1, 2, 3, 4, 5], key)
        short = _make_synthetic_snapshot([1, 2, 3], key)

        cache.insert(longer)
        cache.insert(short)

        assert cache.stats()["current_entries"] == 2
        assert cache.stats()["prefix_prunes"] == 0

    def test_pruning_clears_lru_slot(self):
        cache = DFlashPrefixCache(max_entries=2, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        short = _make_synthetic_snapshot([1, 2, 3], key)
        longer = _make_synthetic_snapshot([1, 2, 3, 4], key)
        unrelated = _make_synthetic_snapshot([9, 9, 9, 9], key)

        cache.insert(short)
        cache.insert(longer)

        cache.insert(unrelated)
        assert cache.stats()["current_entries"] == 2
        assert cache.stats()["prefix_prunes"] == 1
        assert cache.stats()["evictions"] == 0

class TestCrossKindPrune:
    def test_generation_prunes_dominated_prefill(self):

        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        prefill = _make_synthetic_snapshot([1, 2, 3], key, kind="prefill")
        generation = _make_synthetic_snapshot([1, 2, 3, 4, 5], key, kind="generation")

        cache.insert(prefill)
        cache.insert(generation)
        assert cache.stats()["current_entries"] == 1
        assert cache.stats()["cross_kind_prunes"] == 1

    def test_disabled_keeps_both_kinds(self):
        cache = DFlashPrefixCache(
            max_entries=8,
            max_bytes=8 * 1024 * 1024 * 1024,
            cross_kind_prune=False,
        )
        key = _make_key()
        prefill = _make_synthetic_snapshot([1, 2, 3], key, kind="prefill")
        generation = _make_synthetic_snapshot([1, 2, 3, 4, 5], key, kind="generation")

        cache.insert(prefill)
        cache.insert(generation)
        assert cache.stats()["current_entries"] == 2
        assert cache.stats()["cross_kind_prunes"] == 0

    def test_does_not_cross_keys(self):
        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key_a = _make_key(target_model_id="A")
        key_b = _make_key(target_model_id="B")
        prefill = _make_synthetic_snapshot([1, 2, 3], key_a, kind="prefill")
        generation = _make_synthetic_snapshot([1, 2, 3, 4, 5], key_b, kind="generation")

        cache.insert(prefill)
        cache.insert(generation)
        assert cache.stats()["current_entries"] == 2
        assert cache.stats()["cross_kind_prunes"] == 0

class TestSkipLongSnapshot:
    def test_long_snapshot_skipped(self):
        cache = DFlashPrefixCache(
            max_entries=8,
            max_bytes=8 * 1024 * 1024 * 1024,
            max_snapshot_tokens=5,
        )
        key = _make_key()
        too_long = _make_synthetic_snapshot([1, 2, 3, 4, 5, 6], key)
        cache.insert(too_long)
        stats = cache.stats()
        assert stats["current_entries"] == 0
        assert stats["skipped_too_long"] == 1
        assert stats["insertions"] == 0

    def test_short_snapshot_inserted(self):
        cache = DFlashPrefixCache(
            max_entries=8,
            max_bytes=8 * 1024 * 1024 * 1024,
            max_snapshot_tokens=5,
        )
        key = _make_key()
        ok = _make_synthetic_snapshot([1, 2, 3, 4, 5], key)
        cache.insert(ok)
        stats = cache.stats()
        assert stats["current_entries"] == 1
        assert stats["skipped_too_long"] == 0
        assert stats["insertions"] == 1

    def test_disabled_with_zero(self):
        cache = DFlashPrefixCache(
            max_entries=8,
            max_bytes=8 * 1024 * 1024 * 1024,
            max_snapshot_tokens=0,
        )
        key = _make_key()
        snap = _make_synthetic_snapshot(list(range(100)), key)
        cache.insert(snap)
        assert cache.stats()["current_entries"] == 1
        assert cache.stats()["skipped_too_long"] == 0

    def test_cap_bypassed_when_l2_wired(self):

        class _StubL2:
            def __init__(self):
                self.inserts = []

            def insert_async(self, snap):
                self.inserts.append(snap)

            def lookup(self, tokens, key):
                return None

            def stats(self):
                return {}

            def clear(self):
                pass

        l2 = _StubL2()
        cache = DFlashPrefixCache(
            max_entries=8,
            max_bytes=8 * 1024 * 1024 * 1024,
            l2=l2,
            max_snapshot_tokens=5,
        )
        key = _make_key()
        too_long = _make_synthetic_snapshot([1, 2, 3, 4, 5, 6], key)
        cache.insert(too_long)
        stats = cache.stats()
        assert stats["current_entries"] == 1
        assert stats["skipped_too_long"] == 0
        assert stats["insertions"] == 1

class TestByteBudget:
    def test_byte_budget_evicts(self):

        probe = _make_synthetic_snapshot([1], _make_key())
        per_entry = probe.nbytes

        cache = DFlashPrefixCache(max_entries=999, max_bytes=int(per_entry * 1.5))
        key = _make_key()
        cache.insert(_make_synthetic_snapshot([1], key))
        cache.insert(_make_synthetic_snapshot([2], key))
        stats = cache.stats()
        assert stats["current_entries"] == 1
        assert stats["evictions"] == 1
        assert stats["byte_budget_evictions"] == 1
        assert stats["current_bytes"] <= cache._max_bytes

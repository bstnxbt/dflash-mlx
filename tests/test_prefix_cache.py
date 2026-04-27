
from __future__ import annotations

import os
import threading

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from dflash_mlx.prefix_cache import (
    DFlashPrefixCache,
    DFlashPrefixKey,
    DFlashPrefixSnapshot,
    build_snapshot,
    compute_stable_prefix_len,
    hydrate_target_cache,
    prefix_cache_enabled,
    read_budget_env,
    serialize_target_cache,
    target_cache_is_serializable,
)
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


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
    )


# ----------------------------------------------------------------------
# Serialize / hydrate round-trip
# ----------------------------------------------------------------------


class TestSerializeHydrate:
    def test_kv_only_round_trip(self):
        src = [_make_kv_cache_populated(n_tokens=5)]
        assert target_cache_is_serializable(src)
        fa, gdn = serialize_target_cache(src)
        assert fa[0] is not None
        assert gdn[0] is None
        k, v, offset = fa[0]
        assert offset == 5
        # Build template of matching types, hydrate, compare
        template = [KVCache()]
        snapshot = DFlashPrefixSnapshot(
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
        # Values must be element-wise equal
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
        snapshot = DFlashPrefixSnapshot(
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
        # Layers 0, 1 should be FA; layers 2, 3 should be GDN.
        assert fa[0] is not None and gdn[0] is None
        assert fa[1] is not None and gdn[1] is None
        assert fa[2] is None and gdn[2] is not None
        assert fa[3] is None and gdn[3] is not None

        template = [KVCache(), KVCache(),
                    RecurrentRollbackCache(size=3, conv_kernel_size=4),
                    RecurrentRollbackCache(size=3, conv_kernel_size=4)]
        snapshot = DFlashPrefixSnapshot(
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
        snapshot = DFlashPrefixSnapshot(
            token_ids=(1,),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 1, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        template = [KVCache(), KVCache()]  # wrong size
        with pytest.raises(ValueError, match="Template cache length"):
            hydrate_target_cache(snapshot, template)

    def test_hydrate_type_mismatch_raises(self):
        src = [_make_kv_cache_populated()]
        fa, gdn = serialize_target_cache(src)
        snapshot = DFlashPrefixSnapshot(
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


# ----------------------------------------------------------------------
# Mutation isolation
# ----------------------------------------------------------------------


class TestMutationIsolation:
    def test_kv_clone_is_independent(self):
        src = [_make_kv_cache_populated(n_tokens=3)]
        fa, gdn = serialize_target_cache(src)
        # Mutate source AFTER snapshot.
        src[0].keys = mx.zeros_like(src[0].keys)
        mx.eval(src[0].keys)
        # Snapshot should be unchanged.
        snap_k, _, _ = fa[0]
        assert not mx.all(snap_k == 0).item(), \
            "Snapshot shares buffer with live cache — deep-copy is broken"

    def test_gdn_clone_is_independent(self):
        src = [_make_gdn_cache_populated(size=3, conv_k=4)]
        fa, gdn = serialize_target_cache(src)
        # Mutate source.
        src[0].cache[1] = mx.zeros_like(src[0].cache[1])
        mx.eval(src[0].cache[1])
        # Snapshot should be unchanged.
        assert not mx.all(gdn[0][1] == 0).item(), \
            "GDN snapshot shares buffer with live cache — deep-copy is broken"

    def test_hydrate_returns_independent_cache(self):
        src = [_make_kv_cache_populated(n_tokens=3)]
        fa, gdn = serialize_target_cache(src)
        snapshot = DFlashPrefixSnapshot(
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
        # Mutate first hydration.
        hydrated1[0].keys = mx.zeros_like(hydrated1[0].keys)
        mx.eval(hydrated1[0].keys)
        # Second hydration and snapshot must both be unchanged.
        h2_k, _ = hydrated2[0].state
        assert not mx.all(h2_k == 0).item()
        snap_k, _, _ = snapshot.fa_states[0]
        assert not mx.all(snap_k == 0).item()


# ----------------------------------------------------------------------
# Snapshot properties
# ----------------------------------------------------------------------


class TestSnapshot:
    def test_prefix_len_matches_token_ids(self):
        snap = _make_synthetic_snapshot(token_ids=[1, 2, 3, 4], key=_make_key())
        assert snap.prefix_len == 4

    def test_nbytes_positive_and_accurate(self):
        snap = _make_synthetic_snapshot(token_ids=[1, 2, 3], key=_make_key())
        # nbytes must at least cover target_hidden and last_logits.
        expected_min = int(snap.target_hidden.nbytes) + int(snap.last_logits.nbytes)
        assert snap.nbytes >= expected_min
        # And non-zero (caches are populated).
        assert snap.nbytes > 0


# ----------------------------------------------------------------------
# LRU cache behavior
# ----------------------------------------------------------------------


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
        # Strict semantic: stored snapshot is a prefix of the longer request.
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
        # Cache has 3 tokens. Request has 5. Matching prefix = 3.
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        snap = _make_synthetic_snapshot([10, 20, 30], key)
        cache.insert(snap)
        matched, found = cache.lookup([10, 20, 30, 99, 42], key)
        assert matched == 3
        assert found is snap

    def test_divergent_tokens_refused(self):
        # Strict semantic: a snapshot that diverges past the match point
        # cannot be partially used — its cache is too far advanced.
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        snap = _make_synthetic_snapshot([10, 20, 30, 40], key)
        cache.insert(snap)
        matched, found = cache.lookup([10, 20, 99, 40], key)
        assert matched == 0
        assert found is None

    def test_request_shorter_than_snapshot_refused(self):
        # Stored snapshot is [10,20,30,40,50]. Request is [10,20,30].
        # Snapshot is NOT a prefix of request (it's longer). Refuse.
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
        # s1 (oldest) should be gone.
        matched1, _ = cache.lookup([1], key)
        matched3, _ = cache.lookup([3], key)
        assert matched1 == 0
        assert matched3 == 1

    def test_evicts_on_bytes_limit(self):
        # Set a small byte budget so each insert forces eviction.
        per_entry_nbytes = _make_synthetic_snapshot([1, 2, 3], _make_key()).nbytes
        cache = DFlashPrefixCache(max_entries=1000, max_bytes=per_entry_nbytes)
        key = _make_key()
        cache.insert(_make_synthetic_snapshot([1], key))
        cache.insert(_make_synthetic_snapshot([2], key))
        stats = cache.stats()
        assert stats["current_bytes"] <= per_entry_nbytes + 1
        assert stats["evictions"] >= 1

    def test_lru_touch_on_lookup(self):
        # After touching s1 via lookup, inserting s3 should evict s2 (now oldest),
        # not s1.
        cache = DFlashPrefixCache(max_entries=2)
        key = _make_key()
        s1 = _make_synthetic_snapshot([1], key)
        s2 = _make_synthetic_snapshot([2], key)
        cache.insert(s1)
        cache.insert(s2)
        cache.lookup([1], key)  # touch s1
        s3 = _make_synthetic_snapshot([3], key)
        cache.insert(s3)
        # s2 evicted, s1 and s3 remain.
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
        cache.lookup([1, 2], key)  # miss (empty)
        cache.insert(_make_synthetic_snapshot([1, 2], key))
        cache.lookup([1, 2], key)         # exact hit (full snapshot)
        cache.lookup([1, 2, 3, 4], key)    # prefix hit (snapshot is prefix of longer request)
        stats = cache.stats()
        assert stats["misses"] == 1
        assert stats["exact_hits"] == 1
        assert stats["prefix_hits"] == 1
        assert stats["prefill_tokens_saved"] == 2 + 2  # 2 exact + 2 prefix
        assert stats["insertions"] == 1


# ----------------------------------------------------------------------
# Thread safety smoke
# ----------------------------------------------------------------------


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
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(b,)) for b in (100, 200, 300)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Concurrent access raised: {errors}"


# ----------------------------------------------------------------------
# Env gate
# ----------------------------------------------------------------------


class TestStablePrefixLen:

    IM_START = 900
    ASST = 901

    def test_no_markers_returns_full_length(self):
        # Unknown tokenizer (no marker IDs) falls back to full length.
        assert compute_stable_prefix_len([1, 2, 3, 4], im_start_id=None, assistant_id=None) == 4
        assert compute_stable_prefix_len([], im_start_id=None, assistant_id=None) == 0

    def test_no_assistant_marker_returns_full_length(self):
        # No `<|im_start|> assistant` pair present.
        tokens = [1, 2, 3, self.IM_START, 77, 5]  # im_start but not followed by assistant
        assert compute_stable_prefix_len(
            tokens, im_start_id=self.IM_START, assistant_id=self.ASST
        ) == len(tokens)

    def test_trailing_tail_stripped(self):
        # sys + user + <|im_start|> assistant \n <think> \n
        tokens = [10, 11, 12, 13, self.IM_START, self.ASST, 100, 200, 300]
        # Strip from position 4 onwards (the <|im_start|>).
        assert compute_stable_prefix_len(
            tokens, im_start_id=self.IM_START, assistant_id=self.ASST
        ) == 4

    def test_last_occurrence_wins(self):
        # Multiple <|im_start|> assistant pairs — strip from the LAST.
        tokens = [
            10,
            self.IM_START, self.ASST, 20, 21,      # historical assistant turn
            22, 23,                                # user turn markers omitted for test
            self.IM_START, self.ASST, 100, 200,    # current generation prompt + thinking
        ]
        # Last pair is at position 7. Strip from there.
        assert compute_stable_prefix_len(
            tokens, im_start_id=self.IM_START, assistant_id=self.ASST
        ) == 7

    def test_stripped_matches_across_turns(self):
        # Turn 1: sys + u1 + <|im_start|> assistant \n <think> \n  (tail at 4)
        turn1 = [10, 11, 12, 13, self.IM_START, self.ASST, 100, 200]
        # Turn 2: sys + u1 + <|im_start|> assistant \n a1 content <|im_end|> \n +
        #         <|im_start|> user \n u2 <|im_end|> \n + <|im_start|> assistant \n <think> \n
        turn2 = [
            10, 11, 12, 13,                        # stable prefix (sys + u1)
            self.IM_START, self.ASST, 50, 51, 52,   # historical a1 (different from turn 1's tail)
            902, 903,                               # user markers
            self.IM_START, 600, 601,                # user block (not assistant)
            self.IM_START, self.ASST, 700, 701,     # new generation prompt
        ]
        s1 = compute_stable_prefix_len(turn1, im_start_id=self.IM_START, assistant_id=self.ASST)
        s2 = compute_stable_prefix_len(turn2, im_start_id=self.IM_START, assistant_id=self.ASST)
        assert turn1[:s1] == turn2[:s1], "stripped turn 1 should be a prefix of stripped turn 2"
        # Turn 2 has extra content between s1 and s2.
        assert s2 >= s1

    def test_tuple_input(self):
        # Accepts tuple as well as list.
        tokens = (10, 11, self.IM_START, self.ASST, 300)
        assert compute_stable_prefix_len(
            tokens, im_start_id=self.IM_START, assistant_id=self.ASST
        ) == 2

    def test_short_input(self):
        # Tokens shorter than 2 — return length unchanged.
        assert compute_stable_prefix_len([], im_start_id=1, assistant_id=2) == 0
        assert compute_stable_prefix_len([5], im_start_id=1, assistant_id=2) == 1


class TestEnvGate:
    def test_default_disabled(self, monkeypatch):
        monkeypatch.delenv("DFLASH_PREFIX_CACHE", raising=False)
        assert prefix_cache_enabled() is False

    def test_enabled_values(self, monkeypatch):
        for val in ("1", "true", "True", "yes", "on"):
            monkeypatch.setenv("DFLASH_PREFIX_CACHE", val)
            assert prefix_cache_enabled() is True, f"{val!r} should enable"

    def test_disabled_values(self, monkeypatch):
        for val in ("0", "false", "False", "no", "NO", ""):
            monkeypatch.setenv("DFLASH_PREFIX_CACHE", val)
            assert prefix_cache_enabled() is False, f"{val!r} should disable"

    def test_budget_env_defaults(self, monkeypatch):
        monkeypatch.delenv("DFLASH_PREFIX_CACHE_MAX_ENTRIES", raising=False)
        monkeypatch.delenv("DFLASH_PREFIX_CACHE_MAX_BYTES", raising=False)
        entries, mbytes = read_budget_env(default_entries=3, default_bytes=1024)
        assert entries == 3
        assert mbytes == 1024

    def test_budget_env_override(self, monkeypatch):
        monkeypatch.setenv("DFLASH_PREFIX_CACHE_MAX_ENTRIES", "7")
        monkeypatch.setenv("DFLASH_PREFIX_CACHE_MAX_BYTES", "2048")
        entries, mbytes = read_budget_env()
        assert entries == 7
        assert mbytes == 2048

    def test_budget_env_invalid_values_fall_back(self, monkeypatch):
        monkeypatch.setenv("DFLASH_PREFIX_CACHE_MAX_ENTRIES", "not-a-number")
        monkeypatch.setenv("DFLASH_PREFIX_CACHE_MAX_BYTES", "also-bad")
        entries, mbytes = read_budget_env(default_entries=5, default_bytes=100)
        assert entries == 5
        assert mbytes == 100


class TestPrefixPruning:
    def test_strict_prefix_pruned_on_insert(self):
        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        short = _make_synthetic_snapshot([1, 2, 3], key)
        longer = _make_synthetic_snapshot([1, 2, 3, 4, 5], key)

        cache.insert(short)
        assert cache.stats()["current_entries"] == 1

        cache.insert(longer)
        # short is dominated by longer with the same key → pruned.
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
        # Equal tokens with same key: first is dominated by second.
        assert cache.stats()["current_entries"] == 1
        assert cache.stats()["prefix_prunes"] == 1

    def test_unrelated_tokens_not_pruned(self):
        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        a = _make_synthetic_snapshot([1, 2, 3], key)
        b = _make_synthetic_snapshot([9, 8, 7, 6], key)

        cache.insert(a)
        cache.insert(b)
        # No prefix relation → both kept.
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
        # Different keys: pruning rule does not cross fingerprints.
        assert cache.stats()["current_entries"] == 2
        assert cache.stats()["prefix_prunes"] == 0

    def test_longer_then_shorter_no_prune(self):
        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        longer = _make_synthetic_snapshot([1, 2, 3, 4, 5], key)
        short = _make_synthetic_snapshot([1, 2, 3], key)

        cache.insert(longer)
        cache.insert(short)
        # Inserting a shorter snapshot after a longer one: the longer one
        # is NOT a prefix of the shorter, so no prune happens. The shorter
        # is also not "dominated" — we only prune existing entries that are
        # prefixes of the incoming.
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
        # Pruning short should free an LRU slot, so unrelated fits without
        # forcing a budget eviction of longer.
        cache.insert(unrelated)
        assert cache.stats()["current_entries"] == 2
        assert cache.stats()["prefix_prunes"] == 1
        assert cache.stats()["evictions"] == 0

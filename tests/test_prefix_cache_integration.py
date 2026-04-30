# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from dflash_mlx.cache.codecs import (
    build_snapshot,
    hydrate_target_cache,
    target_cache_is_serializable,
)
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
from dflash_mlx.runtime_context import build_runtime_context, runtime_config_from_profile

def _runtime_context(**overrides):
    values = dict(
        profile="balanced",
        prefix_cache=True,
        prefix_cache_l2=False,
        prefix_cache_l2_dir="/tmp/dflash-prefix-l2-test",
    )
    values.update(overrides)
    return build_runtime_context(runtime_config_from_profile(**values))

def _make_populated_target_cache(n_tokens: int = 8):
    caches = []

    kv = KVCache()
    kv.keys = mx.arange(1 * 2 * n_tokens * 4, dtype=mx.float32).reshape(1, 2, n_tokens, 4)
    kv.values = (mx.arange(1 * 2 * n_tokens * 4, dtype=mx.float32) + 100.0).reshape(1, 2, n_tokens, 4)
    kv.offset = n_tokens
    caches.append(kv)

    gdn = RecurrentRollbackCache(size=3, conv_kernel_size=4)
    gdn.cache[0] = mx.arange(12, dtype=mx.float32).reshape(1, 4, 3)
    gdn.cache[1] = (mx.arange(24, dtype=mx.float32) + 1000.0).reshape(1, 2, 4, 3)
    caches.append(gdn)
    mx.eval(kv.keys, kv.values, gdn.cache[0], gdn.cache[1])
    return caches

def _make_key(**overrides) -> DFlashPrefixKey:
    base = dict(
        target_model_id="test/target",
        draft_model_id="test/draft",
        capture_layer_ids=(10, 20),
        draft_sink_size=64,
        draft_window_size=1024,
    )
    base.update(overrides)
    return DFlashPrefixKey(**base)

def _simulate_serve_insert(
    cache: DFlashPrefixCache,
    key: DFlashPrefixKey,
    prompt_tokens: list[int],
    n_cached_tokens: int,
):
    live_cache = _make_populated_target_cache(n_tokens=n_cached_tokens)
    hidden_dim = 6
    target_hidden = mx.zeros((1, len(prompt_tokens), hidden_dim), dtype=mx.float32)

    target_hidden = mx.arange(1 * len(prompt_tokens) * hidden_dim, dtype=mx.float32).reshape(
        1, len(prompt_tokens), hidden_dim
    )
    last_logits = mx.arange(32, dtype=mx.float32).reshape(1, 32)
    mx.eval(target_hidden, last_logits)

    assert target_cache_is_serializable(live_cache)
    snap = build_snapshot(
        token_ids=prompt_tokens,
        target_cache=live_cache,
        target_hidden=target_hidden,
        last_logits=last_logits,
        key=key,
    )
    cache.insert(snap)
    return snap, live_cache

class TestEndToEnd:
    def test_insert_then_lookup_exact(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        snap, _ = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=8)

        matched, found = cache.lookup(prompt, key)
        assert matched == len(prompt)
        assert found is snap

    def test_insert_then_lookup_prefix(self):

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt_turn1 = [100, 101, 102, 103, 104]
        _simulate_serve_insert(cache, key, prompt_turn1, n_cached_tokens=5)

        prompt_turn2 = prompt_turn1 + [200, 201, 202]
        matched, found = cache.lookup(prompt_turn2, key)
        assert matched == 5
        assert found is not None

    def test_hydrate_after_lookup_preserves_state(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [1, 2, 3, 4]
        snap, live_cache = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=4)

        _, found = cache.lookup(prompt, key)
        assert found is snap

        template = _make_populated_target_cache(n_tokens=1)
        hydrated = hydrate_target_cache(found, template)

        src_k, src_v = live_cache[0].state
        h_k, h_v = hydrated[0].state
        assert mx.all(src_k == h_k).item()
        assert mx.all(src_v == h_v).item()
        assert hydrated[0].offset == live_cache[0].offset

        for a, b in zip(live_cache[1].cache, hydrated[1].cache):
            if a is None:
                assert b is None
            else:
                assert mx.all(a == b).item()

    def test_mutation_of_live_cache_does_not_affect_snapshot(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [7, 8, 9]
        snap, live_cache = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=3)

        live_cache[0].keys = mx.zeros_like(live_cache[0].keys)
        live_cache[1].cache[1] = mx.zeros_like(live_cache[1].cache[1])
        mx.eval(live_cache[0].keys, live_cache[1].cache[1])

        _, found = cache.lookup(prompt, key)
        assert found is not None
        snap_k, _, _ = found.fa_states[0]
        assert not mx.all(snap_k == 0).item()
        assert not mx.all(found.gdn_states[1][1] == 0).item()

    def test_different_key_refuses_match(self):
        cache = DFlashPrefixCache(max_entries=4)
        key_a = _make_key(target_model_id="model-a")
        key_b = _make_key(target_model_id="model-b")
        prompt = [1, 2, 3]
        _simulate_serve_insert(cache, key_a, prompt, n_cached_tokens=3)

        matched, found = cache.lookup(prompt, key_b)
        assert matched == 0
        assert found is None

    def test_cache_handles_monotone_turns_via_pruning(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        _simulate_serve_insert(cache, key, [1, 2], n_cached_tokens=2)
        _simulate_serve_insert(cache, key, [1, 2, 3], n_cached_tokens=3)
        _simulate_serve_insert(cache, key, [1, 2, 3, 4], n_cached_tokens=4)
        stats = cache.stats()

        assert stats["current_entries"] == 1
        assert stats["prefix_prunes"] == 2
        assert stats["evictions"] == 0
        matched, _ = cache.lookup([1, 2, 3, 4], key)
        assert matched == 4

    def test_cache_lru_eviction_on_unrelated_turns(self):
        cache = DFlashPrefixCache(max_entries=2)
        key = _make_key()
        _simulate_serve_insert(cache, key, [1, 2], n_cached_tokens=2)
        _simulate_serve_insert(cache, key, [9, 8], n_cached_tokens=2)
        _simulate_serve_insert(cache, key, [5, 6, 7], n_cached_tokens=3)
        stats = cache.stats()
        assert stats["current_entries"] == 2
        assert stats["evictions"] >= 1
        assert stats["prefix_prunes"] == 0
        matched, _ = cache.lookup([5, 6, 7], key)
        assert matched == 3

class TestServeHelperShapes:

    def test_get_cache_disabled(self, monkeypatch):

        import dflash_mlx.serve as serve_mod
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_SINGLETON", None)
        assert serve_mod._get_dflash_prefix_cache(
            _runtime_context(prefix_cache=False)
        ) is None

    def test_get_cache_enabled_returns_singleton(self, monkeypatch):
        import dflash_mlx.serve as serve_mod
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_SINGLETON", None)
        monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_CONFIG_KEY", None)
        context = _runtime_context(prefix_cache=True)
        first = serve_mod._get_dflash_prefix_cache(context)
        second = serve_mod._get_dflash_prefix_cache(context)
        assert first is not None
        assert first is second

    def test_build_prefix_key_from_provider(self):
        import dflash_mlx.serve as serve_mod

        class FakeProvider:
            model_key = ("target/x", None, "draft/y")

        class FakeDraft:
            target_layer_ids = [3, 7]

        key = serve_mod._build_prefix_key(FakeProvider(), FakeDraft())
        assert key.target_model_id == "target/x"
        assert key.draft_model_id == "draft/y"
        assert key.capture_layer_ids == (3, 7)
        assert isinstance(key.draft_sink_size, int)
        assert isinstance(key.draft_window_size, int)

    def test_build_prefix_key_uses_runtime_draft_window(self):
        import dflash_mlx.serve as serve_mod

        class FakeProvider:
            model_key = ("target/x", None, "draft/y")

        class FakeDraft:
            target_layer_ids = [3, 7]

        key_default = serve_mod._build_prefix_key(
            FakeProvider(),
            FakeDraft(),
            _runtime_context(draft_sink_size=64, draft_window_size=1024),
        )
        key_windowed = serve_mod._build_prefix_key(
            FakeProvider(),
            FakeDraft(),
            _runtime_context(draft_sink_size=32, draft_window_size=512),
        )

        assert key_default.draft_sink_size == 64
        assert key_default.draft_window_size == 1024
        assert key_windowed.draft_sink_size == 32
        assert key_windowed.draft_window_size == 512
        assert key_default != key_windowed

    def test_build_prefix_key_handles_missing(self):
        import dflash_mlx.serve as serve_mod

        class FakeProvider:
            pass

        class FakeDraft:
            target_layer_ids = ()

        key = serve_mod._build_prefix_key(FakeProvider(), FakeDraft())
        assert key.target_model_id == ""
        assert key.draft_model_id == ""
        assert key.capture_layer_ids == ()

class TestContextConfigExposedCorrectly:

    def test_prefix_cache_config_affects_singleton(self, monkeypatch):
        import dflash_mlx.serve as serve_mod
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_SINGLETON", None)
        assert serve_mod._get_dflash_prefix_cache(
            _runtime_context(prefix_cache=False)
        ) is None

        assert serve_mod._get_dflash_prefix_cache(
            _runtime_context(prefix_cache=True)
        ) is not None

    def test_singleton_rebuilds_when_config_changes(self, monkeypatch):
        import dflash_mlx.serve as serve_mod
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_SINGLETON", None)
        monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_CONFIG_KEY", None)
        first = serve_mod._get_dflash_prefix_cache(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=2)
        )
        second = serve_mod._get_dflash_prefix_cache(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=7)
        )

        assert first is not None
        assert second is not None
        assert first is not second
        assert second.stats()["max_entries"] == 7

    def test_budgets_propagate_to_cache(self, monkeypatch):
        import dflash_mlx.serve as serve_mod
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_SINGLETON", None)
        monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_CONFIG_KEY", None)
        cache = serve_mod._get_dflash_prefix_cache(
            _runtime_context(
                prefix_cache=True,
                prefix_cache_max_entries=7,
                prefix_cache_max_bytes=1234567,
            )
        )
        assert cache is not None
        stats = cache.stats()
        assert stats["max_entries"] == 7
        assert stats["max_bytes"] == 1234567

    def test_prefix_flow_lookup_records_hit(self, monkeypatch):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        class FakeProvider:
            model_key = ("target/x", None, "draft/y")

        class FakeDraft:
            target_layer_ids = [3, 7]

        class FakeTokenizer:
            unk_token_id = -1

            def convert_tokens_to_ids(self, tokens):
                return [-1 for _ in tokens]

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key(
            target_model_id="target/x",
            draft_model_id="draft/y",
            capture_layer_ids=(3, 7),
        )
        prompt = [11, 12, 13, 14]
        snap, _ = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=4)

        monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_SINGLETON", cache)
        monkeypatch.setattr(flow_mod, "build_prefix_key", lambda *_args: key)

        flow = flow_mod.PrefixCacheFlow.for_request(
            model_provider=FakeProvider(),
            draft_model=FakeDraft(),
            tokenizer=FakeTokenizer(),
            prompt=prompt,
            runtime_context=_runtime_context(prefix_cache=True),
        )

        assert flow.hit_tokens == len(prompt)
        assert flow.snapshot is snap
        assert flow.stable_prefix_len == len(prompt)

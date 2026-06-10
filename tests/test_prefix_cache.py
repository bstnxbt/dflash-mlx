# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from __future__ import annotations

import json
from types import SimpleNamespace
import threading

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache

from dflash_mlx.cache.codecs import (
    PrefixSnapshotBuilder,
    build_snapshot,
    hydrate_target_cache,
    serialize_target_cache,
)
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.cache.store import PrefixSnapshotStore
from dflash_mlx.diagnostics import TraceConfig
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
from dflash_mlx.server.prefix_cache_flow import (
    compute_request_stable_prefix_len,
    compute_stable_prefix_len,
)

def _make_kv_cache_populated(n_tokens: int = 4, hkv: int = 2, d: int = 8) -> KVCache:
    cache = KVCache()
    keys = mx.arange(1 * hkv * n_tokens * d, dtype=mx.float32).reshape(1, hkv, n_tokens, d)
    vals = (mx.arange(1 * hkv * n_tokens * d, dtype=mx.float32) + 1000.0).reshape(1, hkv, n_tokens, d)
    cache.keys = keys
    cache.values = vals
    cache.offset = n_tokens
    mx.eval(cache.keys, cache.values)
    return cache

def _make_rotating_cache_populated(
    n_tokens: int = 7,
    *,
    max_size: int = 4,
    keep: int = 1,
    hkv: int = 2,
    d: int = 8,
) -> RotatingKVCache:
    cache = RotatingKVCache(max_size=max_size, keep=keep)
    for token in range(n_tokens):
        base = token * hkv * d
        keys = (
            mx.arange(hkv * d, dtype=mx.float32).reshape(1, hkv, 1, d)
            + float(base)
        )
        vals = keys + 1000.0
        cache.update_and_fetch(keys, vals)
    mx.eval(cache.keys, cache.values)
    return cache

def _append_rotating_token(
    cache: RotatingKVCache,
    token: int,
    *,
    hkv: int = 2,
    d: int = 8,
) -> None:
    base = token * hkv * d
    keys = (
        mx.arange(hkv * d, dtype=mx.float32).reshape(1, hkv, 1, d)
        + float(base)
    )
    vals = keys + 1000.0
    cache.update_and_fetch(keys, vals)
    mx.eval(cache.keys, cache.values)

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
        template_hash="a" * 64,
        prompt_policy_hash="b" * 64,
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

def _make_full_hidden_snapshot(
    *,
    token_ids,
    fa_states,
    gdn_states,
    target_hidden: mx.array,
    last_logits,
    key: DFlashPrefixKey,
    kind: str = "prefill",
) -> DFlashPrefixSnapshot:
    total_len = int(target_hidden.shape[1])
    return DFlashPrefixSnapshot(
        token_ids=tuple(token_ids),
        fa_states=fa_states,
        gdn_states=gdn_states,
        target_hidden_chunks=(target_hidden,),
        target_hidden_chunk_spans=((0, total_len),),
        target_hidden_total_len=total_len,
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


def test_l1_prunes_dominated_prefill_snapshots_without_frontier_stride():
    cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
    key = _make_key()
    cache.insert(_make_synthetic_snapshot([1, 2, 3, 4], key))
    cache.insert(_make_synthetic_snapshot([1, 2, 3, 4, 5, 6, 7, 8], key))
    cache.insert(_make_synthetic_snapshot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key))

    matched_len, snapshot = cache.lookup([1, 2, 3, 4, 99], key)

    assert matched_len == 0
    assert snapshot is None
    assert cache.stats()["current_entries"] == 1


def test_l1_preserves_aligned_prefill_frontiers():
    cache = DFlashPrefixCache(
        max_entries=8,
        max_bytes=8 * 1024 * 1024 * 1024,
        frontier_stride=4,
    )
    key = _make_key()
    frontier = _make_synthetic_snapshot([1, 2, 3, 4], key)
    cache.insert(frontier)
    cache.insert(_make_synthetic_snapshot([1, 2, 3, 4, 5, 6, 7, 8], key))
    cache.insert(_make_synthetic_snapshot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key))

    matched_len, snapshot = cache.lookup([1, 2, 3, 4, 99], key)

    assert matched_len == 4
    assert snapshot is frontier
    assert cache.stats()["current_entries"] == 3


def test_l1_aligned_frontier_duplicate_replaces_old_snapshot():
    cache = DFlashPrefixCache(
        max_entries=8,
        max_bytes=8 * 1024 * 1024 * 1024,
        frontier_stride=4,
    )
    key = _make_key()
    old_frontier = _make_synthetic_snapshot([1, 2, 3, 4], key)
    new_frontier = _make_synthetic_snapshot([1, 2, 3, 4], key)
    cache.insert(old_frontier)
    cache.insert(_make_synthetic_snapshot([1, 2, 3, 4, 5, 6, 7, 8], key))
    cache.insert(new_frontier)

    matched_len, snapshot = cache.lookup([1, 2, 3, 4, 99], key)

    assert matched_len == 4
    assert snapshot is new_frontier
    assert cache.stats()["current_entries"] == 2


def test_l1_lookup_cache_event_records_request_id(tmp_path):
    cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
    cache.set_trace_config(TraceConfig(log_dir=tmp_path))
    key = _make_key()
    cache.insert(_make_synthetic_snapshot([1, 2, 3], key))

    matched_len, snapshot = cache.lookup([1, 2, 3, 4], key, request_id=42)

    assert matched_len == 3
    assert snapshot is not None
    rows = [
        json.loads(line)
        for line in (tmp_path / "cache_events.jsonl").read_text().splitlines()
    ]
    lookup = [row for row in rows if row.get("op") == "lookup"][-1]
    assert lookup["request_id"] == 42
    assert lookup["result"] == "prefix_hit"


def test_store_l2_miss_cache_event_records_request_id(tmp_path):
    class _MissL2:
        def lookup(self, _tokens, _key, **_kwargs):
            return None

        def stats(self):
            return {}

    key = _make_key()
    store = PrefixSnapshotStore(
        l1=DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024),
        l2=_MissL2(),
    )
    store.set_trace_config(TraceConfig(log_dir=tmp_path))

    matched_len, snapshot = store.lookup([9, 10], key, request_id=99)

    assert matched_len == 0
    assert snapshot is None
    rows = [
        json.loads(line)
        for line in (tmp_path / "cache_events.jsonl").read_text().splitlines()
    ]
    lookup = [row for row in rows if row.get("op") == "lookup"][-1]
    assert lookup["request_id"] == 99
    assert lookup["result"] == "miss"


def test_store_l2_hit_cache_event_records_request_id(tmp_path):
    key = _make_key()
    l2_snapshot = _make_synthetic_snapshot([1, 2, 3], key)

    class _HitL2:
        def __init__(self, snapshot):
            self.snapshot = snapshot
            self.inserts = []

        def lookup(self, _tokens, _key, **_kwargs):
            return self.snapshot

        def insert_async(self, snapshot):
            self.inserts.append(snapshot)
            return True

        def stats(self):
            return {}

    store = PrefixSnapshotStore(
        l1=DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024),
        l2=_HitL2(l2_snapshot),
    )
    store.set_trace_config(TraceConfig(log_dir=tmp_path))

    matched_len, snapshot = store.lookup([1, 2, 3, 4], key, request_id=123)

    assert matched_len == 3
    assert snapshot is l2_snapshot
    rows = [
        json.loads(line)
        for line in (tmp_path / "cache_events.jsonl").read_text().splitlines()
    ]
    lookup = [row for row in rows if row.get("op") == "lookup"][-1]
    assert lookup["request_id"] == 123
    assert lookup["result"] == "l2_hit"


def test_store_promotes_l2_prefix_hit_to_l1_for_next_lookup():
    key = _make_key()
    l2_snapshot = _make_synthetic_snapshot([1, 2, 3], key)

    class _HitL2:
        def __init__(self, snapshot):
            self.snapshot = snapshot
            self.lookups = []
            self.inserts = []

        def lookup(self, _tokens, _key, **kwargs):
            self.lookups.append(kwargs)
            min_token_len = int(kwargs.get("min_token_len", 0))
            if len(self.snapshot.token_ids) <= min_token_len:
                return None
            return self.snapshot

        def insert_async(self, snapshot):
            self.inserts.append(snapshot)
            return True

        def stats(self):
            return {}

    l2 = _HitL2(l2_snapshot)
    store = PrefixSnapshotStore(
        l1=DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024),
        l2=l2,
    )

    first_len, first = store.lookup([1, 2, 3, 4], key)
    second_len, second = store.lookup([1, 2, 3, 5], key)

    assert first_len == 3
    assert first is l2_snapshot
    assert second_len == 3
    assert second is l2_snapshot
    assert len(l2.lookups) == 2
    assert l2.lookups[0].get("min_token_len", 0) == 0
    assert l2.lookups[1]["min_token_len"] == 3


def test_prefix_snapshot_builder_matches_build_snapshot_shape():
    key = _make_key()
    target_cache = [_make_kv_cache_populated(n_tokens=3), _make_gdn_cache_populated()]
    target_hidden = mx.arange(1 * 3 * 4, dtype=mx.float32).reshape(1, 3, 4)
    last_logits = mx.arange(16, dtype=mx.float32).reshape(1, 16)
    builder = PrefixSnapshotBuilder(
        key=key,
        draft_sink_size=16,
        draft_window_size=2048,
    )

    snapshot = builder.build(
        token_ids=[1, 2, 3],
        target_cache=target_cache,
        target_hidden=target_hidden,
        last_logits=last_logits,
        kind="prefill",
    )

    assert snapshot.key == key
    assert snapshot.kind == "prefill"
    assert snapshot.token_ids == (1, 2, 3)
    assert len(snapshot.fa_states) == 2
    assert len(snapshot.gdn_states) == 2
    assert snapshot.last_logits is not None

    target_cache[0].keys = mx.zeros_like(target_cache[0].keys)
    mx.eval(target_cache[0].keys)
    snap_k, _, _ = snapshot.fa_states[0]
    assert not mx.all(snap_k == 0).item()

class TestSerializeHydrate:
    def test_kv_only_round_trip(self):
        src = [_make_kv_cache_populated(n_tokens=5)]
        fa, gdn = serialize_target_cache(src)
        assert fa[0] is not None
        assert gdn[0] is None
        k, v, offset = fa[0]
        assert offset == 5

        template = [KVCache()]
        snapshot = _make_full_hidden_snapshot(
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

    def test_rotating_kv_round_trip_uses_temporal_order(self):
        src = [_make_rotating_cache_populated(n_tokens=7, max_size=4, keep=1)]
        fa, gdn = serialize_target_cache(src)
        assert fa[0] is not None
        assert gdn[0] is None
        k, v, offset = fa[0][:3]
        assert offset == 7
        assert k.shape[2] == 4
        assert fa[0][3] == src[0]._idx

        template = [RotatingKVCache(max_size=4, keep=1)]
        snapshot = _make_full_hidden_snapshot(
            token_ids=tuple(range(7)),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 7, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        hydrated = hydrate_target_cache(snapshot, template)
        assert isinstance(hydrated[0], RotatingKVCache)
        assert hydrated[0].offset == 7
        assert hydrated[0].max_size == 4
        assert hydrated[0].keep == 1
        src_k = src[0]._temporal_order(src[0].keys)
        src_v = src[0]._temporal_order(src[0].values)
        h_k = hydrated[0]._temporal_order(hydrated[0].keys)
        h_v = hydrated[0]._temporal_order(hydrated[0].values)
        assert hydrated[0]._idx == src[0]._idx
        assert mx.all(src_k == h_k).item()
        assert mx.all(src_v == h_v).item()

        _append_rotating_token(src[0], 7)
        _append_rotating_token(hydrated[0], 7)
        src_k = src[0]._temporal_order(src[0].keys)
        h_k = hydrated[0]._temporal_order(hydrated[0].keys)
        assert hydrated[0].offset == src[0].offset
        assert mx.all(src_k == h_k).item()

    def test_rotating_kv_round_trip_before_window_wrap(self):
        src = [_make_rotating_cache_populated(n_tokens=3, max_size=8, keep=1)]
        fa, gdn = serialize_target_cache(src)
        k, _v, offset = fa[0][:3]
        assert offset == 3
        assert k.shape[2] == 8
        snapshot = _make_full_hidden_snapshot(
            token_ids=(0, 1, 2),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 3, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        hydrated = hydrate_target_cache(
            snapshot,
            [RotatingKVCache(max_size=8, keep=1)],
        )
        assert isinstance(hydrated[0], RotatingKVCache)
        assert hydrated[0].offset == 3
        assert hydrated[0]._idx == 3
        assert mx.all(
            src[0]._temporal_order(src[0].keys)
            == hydrated[0]._temporal_order(hydrated[0].keys)
        ).item()
        assert mx.all(
            src[0]._temporal_order(src[0].values)
            == hydrated[0]._temporal_order(hydrated[0].values)
        ).item()

    def test_rotating_kv_round_trip_preserves_ring_index_for_masks(self):
        src = [_make_rotating_cache_populated(n_tokens=8, max_size=4, keep=1)]
        fa, gdn = serialize_target_cache(src)
        assert fa[0][3] == src[0]._idx
        snapshot = _make_full_hidden_snapshot(
            token_ids=tuple(range(8)),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 8, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        hydrated = hydrate_target_cache(
            snapshot,
            [RotatingKVCache(max_size=4, keep=1)],
        )
        assert isinstance(hydrated[0], RotatingKVCache)
        assert hydrated[0]._idx == src[0]._idx
        src_mask = src[0].make_mask(1, window_size=2)
        hydrated_mask = hydrated[0].make_mask(1, window_size=2)
        assert mx.all(src_mask == hydrated_mask).item()

    def test_rotating_kv_round_trip_clone_isolated(self):
        src = [_make_rotating_cache_populated(n_tokens=7, max_size=4, keep=1)]
        fa, gdn = serialize_target_cache(src)
        snap_k, snap_v = fa[0][0], fa[0][1]
        src[0].keys = mx.zeros_like(src[0].keys)
        src[0].values = mx.zeros_like(src[0].values)
        mx.eval(src[0].keys, src[0].values)
        assert not mx.all(snap_k == 0).item()
        assert not mx.all(snap_v == 0).item()
        snapshot = _make_full_hidden_snapshot(
            token_ids=tuple(range(7)),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 7, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        hydrated = hydrate_target_cache(
            snapshot,
            [RotatingKVCache(max_size=4, keep=1)],
        )
        assert not mx.all(hydrated[0].keys == 0).item()
        assert not mx.all(hydrated[0].values == 0).item()

    def test_rotating_and_gdn_mixed_round_trip(self):
        src = [
            _make_rotating_cache_populated(n_tokens=7, max_size=4, keep=1),
            _make_gdn_cache_populated(),
        ]
        fa, gdn = serialize_target_cache(src)
        snapshot = _make_full_hidden_snapshot(
            token_ids=tuple(range(7)),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 7, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        hydrated = hydrate_target_cache(
            snapshot,
            [
                RotatingKVCache(max_size=4, keep=1),
                RecurrentRollbackCache(size=3, conv_kernel_size=4),
            ],
        )
        assert isinstance(hydrated[0], RotatingKVCache)
        assert isinstance(hydrated[1], RecurrentRollbackCache)
        assert mx.all(
            src[0]._temporal_order(src[0].keys)
            == hydrated[0]._temporal_order(hydrated[0].keys)
        ).item()
        for a, b in zip(src[1].cache, hydrated[1].cache):
            if a is None:
                assert b is None
            else:
                assert mx.all(a == b).item()

    def test_rotating_hydrate_missing_state_fails_fast(self):
        snapshot = _make_full_hidden_snapshot(
            token_ids=(1, 2, 3),
            fa_states=(None,),
            gdn_states=(None,),
            target_hidden=mx.zeros((1, 3, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        with pytest.raises(ValueError, match="missing rotating FA state"):
            hydrate_target_cache(
                snapshot,
                [RotatingKVCache(max_size=4, keep=1)],
            )

    def test_rotating_hydrate_missing_ring_index_fails_fast(self):
        src = [_make_rotating_cache_populated(n_tokens=7, max_size=4, keep=1)]
        fa, gdn = serialize_target_cache(src)
        k, v, offset = fa[0][:3]
        snapshot = _make_full_hidden_snapshot(
            token_ids=tuple(range(7)),
            fa_states=((k, v, offset),),
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 7, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )

        with pytest.raises(ValueError, match="missing rotating FA ring index"):
            hydrate_target_cache(
                snapshot,
                [RotatingKVCache(max_size=4, keep=1)],
            )

    def test_mixed_full_attention_draft_snapshot_keeps_full_target_hidden(self):
        target_hidden = mx.zeros((1, 100, 4), dtype=mx.float32)
        draft_model = SimpleNamespace(
            args=SimpleNamespace(
                layer_types=("sliding_attention", "full_attention"),
                sliding_window=16,
            )
        )
        snapshot = build_snapshot(
            token_ids=list(range(100)),
            target_cache=[_make_kv_cache_populated(n_tokens=100)],
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 10), dtype=mx.float32),
            key=_make_key(),
            draft_model=draft_model,
            draft_sink_size=4,
            draft_window_size=16,
            allow_full_attention_context=True,
        )

        assert len(snapshot.target_hidden_chunks) == 1
        assert snapshot.target_hidden_chunk_spans == ((0, 100),)
        assert snapshot.target_hidden_total_len == 100

    def test_mixed_full_attention_draft_snapshot_stays_trimmed_when_not_allowed(self):
        target_hidden = mx.zeros((1, 100, 4), dtype=mx.float32)
        draft_model = SimpleNamespace(
            args=SimpleNamespace(
                layer_types=("sliding_attention", "full_attention"),
                sliding_window=16,
            )
        )
        snapshot = build_snapshot(
            token_ids=list(range(100)),
            target_cache=[_make_kv_cache_populated(n_tokens=100)],
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 10), dtype=mx.float32),
            key=_make_key(),
            draft_model=draft_model,
            draft_sink_size=4,
            draft_window_size=16,
            allow_full_attention_context=False,
        )

        assert len(snapshot.target_hidden_chunks) == 2
        assert snapshot.target_hidden_chunk_spans == ((0, 4), (84, 100))
        assert snapshot.target_hidden_total_len == 100

    def test_snapshot_target_hidden_is_clamped_to_token_prefix(self):
        target_hidden = mx.zeros((1, 8, 4), dtype=mx.float32)

        snapshot = build_snapshot(
            token_ids=list(range(5)),
            target_cache=[_make_kv_cache_populated(n_tokens=5)],
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 10), dtype=mx.float32),
            key=_make_key(),
            trim_target_hidden=False,
        )

        assert snapshot.target_hidden_total_len == 5
        assert snapshot.target_hidden_chunk_spans == ((0, 5),)
        assert snapshot.target_hidden_chunks[0].shape[1] == 5

    def test_build_snapshot_rejects_short_target_hidden(self):
        with pytest.raises(
            ValueError,
            match="target_hidden length 3 < token prefix length 5",
        ):
            build_snapshot(
                token_ids=list(range(5)),
                target_cache=[_make_kv_cache_populated(n_tokens=5)],
                target_hidden=mx.zeros((1, 3, 4), dtype=mx.float32),
                last_logits=mx.zeros((1, 10), dtype=mx.float32),
                key=_make_key(),
            )

    def test_build_snapshot_rejects_fa_cache_offset_mismatch(self):
        with pytest.raises(ValueError, match="FA cache offset 8 at layer 0"):
            build_snapshot(
                token_ids=list(range(5)),
                target_cache=[_make_kv_cache_populated(n_tokens=8)],
                target_hidden=mx.zeros((1, 5, 4), dtype=mx.float32),
                last_logits=mx.zeros((1, 10), dtype=mx.float32),
                key=_make_key(),
            )

    def test_gdn_only_round_trip(self):
        src = [_make_gdn_cache_populated(size=3, conv_k=4)]
        fa, gdn = serialize_target_cache(src)
        assert fa[0] is None
        assert gdn[0] is not None
        assert len(gdn[0]) == 3
        template = [RecurrentRollbackCache(size=3, conv_kernel_size=4)]
        snapshot = _make_full_hidden_snapshot(
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
        fa, gdn = serialize_target_cache(src)

        assert fa[0] is not None and gdn[0] is None
        assert fa[1] is not None and gdn[1] is None
        assert fa[2] is None and gdn[2] is not None
        assert fa[3] is None and gdn[3] is not None

        template = [KVCache(), KVCache(),
                    RecurrentRollbackCache(size=3, conv_kernel_size=4),
                    RecurrentRollbackCache(size=3, conv_kernel_size=4)]
        snapshot = _make_full_hidden_snapshot(
            token_ids=(1, 2, 3, 4),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 4, 4)),
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
        with pytest.raises(TypeError):
            serialize_target_cache(src)

    def test_hydrate_size_mismatch_raises(self):
        src = [_make_kv_cache_populated(n_tokens=1)]
        fa, gdn = serialize_target_cache(src)
        snapshot = _make_full_hidden_snapshot(
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

    def test_hydrate_rejects_fa_cache_offset_mismatch(self):
        src = [_make_kv_cache_populated(n_tokens=4)]
        fa, gdn = serialize_target_cache(src)
        snapshot = _make_full_hidden_snapshot(
            token_ids=(1, 2),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 2, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        with pytest.raises(ValueError, match="FA cache offset 4 at layer 0"):
            hydrate_target_cache(snapshot, [KVCache()])

    def test_hydrate_type_mismatch_raises(self):
        src = [_make_kv_cache_populated(n_tokens=1)]
        fa, gdn = serialize_target_cache(src)
        snapshot = _make_full_hidden_snapshot(
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
        snapshot = _make_full_hidden_snapshot(
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
        snapshots = {
            base: [
                _make_synthetic_snapshot([base, base + i], key)
                for i in range(10)
            ]
            for base in (100, 200, 300)
        }
        errors: list[Exception] = []

        def worker(base: int):
            try:
                for i, snap in enumerate(snapshots[base]):
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
    GEMMA_TURN = 105
    GEMMA_MODEL = 4368
    GEMMA_NEWLINE = 107

    class QwenTokenizer:
        unk_token_id = 3

        def convert_tokens_to_ids(self, tokens):
            if tokens == ["<|im_start|>", "assistant"]:
                return [TestStablePrefixLen.IM_START, TestStablePrefixLen.ASST]
            return [3 for _ in tokens]

    class GemmaTokenizer:
        unk_token_id = 3

        def convert_tokens_to_ids(self, tokens):
            if tokens == ["<|im_start|>", "assistant"]:
                return [3, 111457]
            if tokens == ["<|turn>", "model"]:
                return [
                    TestStablePrefixLen.GEMMA_TURN,
                    TestStablePrefixLen.GEMMA_MODEL,
                ]
            return [3 for _ in tokens]

        def encode(self, text):
            if text == "<|turn>model\n":
                return [
                    TestStablePrefixLen.GEMMA_TURN,
                    TestStablePrefixLen.GEMMA_MODEL,
                    TestStablePrefixLen.GEMMA_NEWLINE,
                ]
            return [3]

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

    def test_gemma_turn_model_boundary_matches_continuation(self):
        turn = 105
        model = 4368
        newline = 107
        thought_channel_start = 100
        thought = 45518
        channel_end = 101
        turn1 = [
            10,
            11,
            turn,
            model,
            newline,
            thought_channel_start,
            thought,
            newline,
            channel_end,
        ]
        turn2 = [10, 11, turn, model, newline, 2021, 1586, 784]

        s1 = compute_stable_prefix_len(turn1, im_start_id=turn, assistant_id=model)
        s2 = compute_stable_prefix_len(turn2, im_start_id=turn, assistant_id=model)

        assert s1 == 2
        assert s2 == 2
        assert turn1[:s1] == turn2[:s2]

    def test_gemma_turn_model_boundary_offset_keeps_role_line(self):
        turn = 105
        model = 4368
        newline = 107
        turn1 = [10, 11, turn, model, newline, 100, 45518, newline, 101]
        turn2 = [10, 11, turn, model, newline, 2021, 1586, 784]

        s1 = compute_stable_prefix_len(
            turn1,
            im_start_id=turn,
            assistant_id=model,
            boundary_offset=3,
        )
        s2 = compute_stable_prefix_len(
            turn2,
            im_start_id=turn,
            assistant_id=model,
            boundary_offset=3,
        )

        assert s1 == 5
        assert s2 == 5
        assert turn1[:s1] == turn2[:s2]

    def test_tuple_input(self):

        tokens = (10, 11, self.IM_START, self.ASST, 300)
        assert compute_stable_prefix_len(
            tokens, im_start_id=self.IM_START, assistant_id=self.ASST
        ) == 2

    def test_short_input(self):

        assert compute_stable_prefix_len([], im_start_id=1, assistant_id=2) == 0
        assert compute_stable_prefix_len([5], im_start_id=1, assistant_id=2) == 1

    def test_request_stable_prefix_tool_turn_strips_next_assistant_prompt(self):
        tokens = [
            10,
            11,
            self.IM_START,
            self.ASST,
            100,
            200,
            300,
            400,
            self.IM_START,
            self.ASST,
        ]
        request = SimpleNamespace(
            request_type="chat",
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "tool_calls": []},
                {"role": "tool", "content": "ok"},
            ],
        )

        assert (
            compute_request_stable_prefix_len(
                tokens,
                tokenizer=self.QwenTokenizer(),
                request=request,
            )
            == 8
        )

    def test_request_stable_prefix_user_turn_uses_marker_boundary(self):
        tokens = [
            10,
            11,
            self.GEMMA_TURN,
            self.GEMMA_MODEL,
            self.GEMMA_NEWLINE,
            100,
            200,
            300,
            400,
        ]
        request = SimpleNamespace(
            request_type="chat",
            messages=[
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
            ],
        )

        assert (
            compute_request_stable_prefix_len(
                tokens,
                tokenizer=self.GemmaTokenizer(),
                request=request,
            )
            == 5
        )

    def test_request_stable_prefix_non_chat_uses_marker_boundary(self):
        tokens = [
            10,
            11,
            self.GEMMA_TURN,
            self.GEMMA_MODEL,
            self.GEMMA_NEWLINE,
            100,
            200,
        ]
        request = SimpleNamespace(request_type="completion")

        assert (
            compute_request_stable_prefix_len(
                tokens,
                tokenizer=self.GemmaTokenizer(),
                request=request,
            )
            == 5
        )

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
    def test_generation_keeps_dominated_prefill_for_exact_prompt_reuse(self):

        cache = DFlashPrefixCache(max_entries=8, max_bytes=8 * 1024 * 1024 * 1024)
        key = _make_key()
        prefill = _make_synthetic_snapshot([1, 2, 3], key, kind="prefill")
        generation = _make_synthetic_snapshot([1, 2, 3, 4, 5], key, kind="generation")

        cache.insert(prefill)
        cache.insert(generation)
        assert cache.stats()["current_entries"] == 2
        assert cache.stats()["cross_kind_prunes"] == 0

        exact_len, exact_snapshot = cache.lookup([1, 2, 3], key)
        assert exact_len == 3
        assert exact_snapshot is prefill

        prefix_len, prefix_snapshot = cache.lookup([1, 2, 3, 4, 5, 6], key)
        assert prefix_len == 5
        assert prefix_snapshot is generation

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

    def test_generation_snapshot_can_exceed_token_cap(self):
        cache = DFlashPrefixCache(
            max_entries=8,
            max_bytes=8 * 1024 * 1024 * 1024,
            max_snapshot_tokens=5,
        )
        key = _make_key()
        generation = _make_synthetic_snapshot(
            [1, 2, 3, 4, 5, 6],
            key,
            kind="generation",
        )

        cache.insert(generation)

        stats = cache.stats()
        assert stats["current_entries"] == 1
        assert stats["skipped_too_long"] == 0
        assert stats["insertions"] == 1
        prefix_len, prefix = cache.lookup([1, 2, 3, 4, 5, 6, 7], key)
        assert prefix_len == 6
        assert prefix is generation

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

    def test_cap_limits_l1_when_l2_wired_but_prefill_persists_to_l2(self):

        class _StubL2:
            def __init__(self):
                self.inserts = []

            def insert_async(self, snap):
                self.inserts.append(snap)
                return True

            def lookup(self, tokens, key):
                return None

            def stats(self):
                return {}

            def clear(self):
                pass

        l2 = _StubL2()
        cache = PrefixSnapshotStore(
            l1=DFlashPrefixCache(
                max_entries=8,
                max_bytes=8 * 1024 * 1024 * 1024,
                max_snapshot_tokens=5,
            ),
            l2=l2,
        )
        key = _make_key()
        too_long = _make_synthetic_snapshot([1, 2, 3, 4, 5, 6], key)
        assert cache.insert(too_long) is True
        stats = cache.stats()
        assert stats["current_entries"] == 0
        assert stats["skipped_too_long"] == 1
        assert stats["insertions"] == 0
        assert l2.inserts == [too_long]

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

    def test_generation_yields_to_matching_prefill_under_byte_pressure(self):
        key = _make_key()
        prefill = _make_synthetic_snapshot([1, 2, 3], key, kind="prefill")
        generation = _make_synthetic_snapshot(
            [1, 2, 3, 4, 5],
            key,
            kind="generation",
        )
        cache = DFlashPrefixCache(
            max_entries=999,
            max_bytes=max(prefill.nbytes, generation.nbytes) + 1,
        )

        assert cache.insert(prefill)
        result = cache.insert_with_evictions(generation)

        assert result.admitted is False
        assert result.inserted_evicted_snapshot is generation
        exact_len, exact = cache.lookup([1, 2, 3], key)
        assert exact_len == 3
        assert exact is prefill
        prefix_len, prefix = cache.lookup([1, 2, 3, 9], key)
        assert prefix_len == 3
        assert prefix is prefill
        stats = cache.stats()
        assert stats["current_entries"] == 1
        assert stats["evictions"] == 1
        assert stats["byte_budget_evictions"] == 1

    def test_generation_coexists_with_prefill_when_unrelated_entry_can_evict(self):
        key = _make_key()
        prefill = _make_synthetic_snapshot([1, 2, 3], key, kind="prefill")
        generation = _make_synthetic_snapshot(
            [1, 2, 3, 4, 5],
            key,
            kind="generation",
        )
        unrelated = _make_synthetic_snapshot([9, 9, 9], key, kind="prefill")
        cache = DFlashPrefixCache(
            max_entries=999,
            max_bytes=prefill.nbytes + generation.nbytes + 1,
        )

        assert cache.insert(unrelated)
        assert cache.insert(prefill)
        result = cache.insert_with_evictions(generation)

        assert result.admitted is True
        assert result.inserted_evicted_snapshot is None
        exact_len, exact = cache.lookup([1, 2, 3], key)
        assert exact_len == 3
        assert exact is prefill
        prefix_len, prefix = cache.lookup([1, 2, 3, 4, 5, 6], key)
        assert prefix_len == 5
        assert prefix is generation
        stats = cache.stats()
        assert stats["current_entries"] == 2
        assert stats["evictions"] == 1
        assert stats["byte_budget_evictions"] == 1

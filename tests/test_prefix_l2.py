# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.prefix_l2 import (
    L2_FILE_SUFFIX,
    L2_LAYOUT_ROOT,
    L2_SCHEMA_VERSION,
    DFlashPrefixL2Cache,
    _deserialize,
    _format_filename,
    _key_hash,
    _runtime_layout_hash,
    _serialize,
    _token_hash,
)
from tests.test_prefix_cache import (
    _make_key,
    _make_synthetic_snapshot,
)

def _all_snapshot_files(cache_dir: Path) -> list[Path]:
    root = cache_dir / L2_LAYOUT_ROOT
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.name.endswith(L2_FILE_SUFFIX):
            out.append(p)
    return out

def _bucket_dir(cache_dir: Path, key) -> Path:
    kh = _key_hash(key)
    return cache_dir / L2_LAYOUT_ROOT / _runtime_layout_hash() / kh[:2] / kh

def _wait_writes(l2: DFlashPrefixL2Cache, expected: int, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if l2.stats()["writes"] >= expected:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"L2 writer did not catch up: writes={l2.stats()['writes']} expected>={expected}"
    )

class TestBfloat16RoundTrip:
    def test_bf16_arrays_preserved_through_save_load(self, tmp_path):
        from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
        n_layers = 4
        head, head_dim, seq = 8, 32, 16

        ref = mx.random.normal((1, head, seq, head_dim))
        bf = ref.astype(mx.bfloat16)
        fa_states = tuple(
            (bf, bf, 0) for _ in range(n_layers)
        )
        last_logits_bf = mx.random.normal((1, 1024)).astype(mx.bfloat16)
        snap = DFlashPrefixSnapshot(
            token_ids=tuple(range(seq)),
            fa_states=fa_states,
            gdn_states=tuple([None] * n_layers),
            target_hidden_chunks=tuple(),
            target_hidden_chunk_spans=tuple(),
            target_hidden_total_len=0,
            last_logits=last_logits_bf,
            key=_make_key(),
            kind="prefill",
        )
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            l2._write_one(snap)
            loaded = l2.lookup(tuple(range(seq)), snap.key)
            assert loaded is not None
            assert loaded.fa_states[0][0].dtype == mx.bfloat16
            assert loaded.last_logits.dtype == mx.bfloat16

            assert mx.all(loaded.fa_states[0][0] == bf).item()
            assert mx.all(loaded.last_logits == last_logits_bf).item()
        finally:
            l2.shutdown()

class TestSerializationRoundTrip:
    def test_full_snapshot_roundtrip(self, tmp_path):
        snap = _make_synthetic_snapshot([1, 2, 3, 4, 5], _make_key())
        arrays, meta_dict = _serialize(snap)
        path = tmp_path / "rt.safetensors"
        mx.save_safetensors(str(path), arrays, metadata=meta_dict)
        loaded_arrays, loaded_meta_dict = mx.load(
            str(path), format="safetensors", return_metadata=True
        )
        loaded_meta = json.loads(loaded_meta_dict["dflash_meta"])
        rehydrated = _deserialize(loaded_arrays, loaded_meta)
        assert rehydrated.token_ids == snap.token_ids
        assert rehydrated.kind == snap.kind
        assert rehydrated.key == snap.key
        assert rehydrated.target_hidden_total_len == snap.target_hidden_total_len
        assert len(rehydrated.fa_states) == len(snap.fa_states)
        assert len(rehydrated.gdn_states) == len(snap.gdn_states)

    def test_post_restore_mutation_isolation(self, tmp_path):
        snap = _make_synthetic_snapshot([1, 2, 3], _make_key())
        arrays, meta_dict = _serialize(snap)
        path = tmp_path / "iso.safetensors"
        mx.save_safetensors(str(path), arrays, metadata=meta_dict)
        loaded_arrays, loaded_meta_dict = mx.load(
            str(path), format="safetensors", return_metadata=True
        )
        loaded_meta = json.loads(loaded_meta_dict["dflash_meta"])
        snap1 = _deserialize(loaded_arrays, loaded_meta)
        snap2 = _deserialize(loaded_arrays, loaded_meta)
        k1, _v1, _ = snap1.fa_states[0]
        before = float(k1[0, 0, 0, 0].item())
        del snap2
        del loaded_arrays
        assert float(k1[0, 0, 0, 0].item()) == before

class TestL2Lifecycle:
    def test_disabled_when_no_l2_attached(self):
        cache = DFlashPrefixCache(max_entries=2, max_bytes=10**9)
        snap = _make_synthetic_snapshot([1, 2, 3], _make_key())
        cache.insert(snap)
        stats = cache.stats()
        assert "l2" not in stats

    def test_evict_promotes_to_l2_disk(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            cache = DFlashPrefixCache(max_entries=1, max_bytes=10**9, l2=l2)
            key = _make_key()
            cache.insert(_make_synthetic_snapshot([1, 2, 3], key))
            cache.insert(_make_synthetic_snapshot([4, 5, 6], key))
            _wait_writes(l2, expected=1)
            assert len(_all_snapshot_files(tmp_path)) == 1
        finally:
            l2.shutdown()

    def test_l2_lookup_promotes_to_l1(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([7, 8, 9, 10], key))
            cache = DFlashPrefixCache(max_entries=4, max_bytes=10**9, l2=l2)
            matched, hydrated = cache.lookup([7, 8, 9, 10], key)
            assert matched == 4
            assert hydrated is not None
            assert hydrated.token_ids == (7, 8, 9, 10)
            stats = cache.stats()
            assert stats["l2_hits"] == 1
            assert stats["current_entries"] == 1
        finally:
            l2.shutdown()

    def test_l2_lookup_miss_when_key_mismatch(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key_a = _make_key(target_model_id="model-a")
            key_b = _make_key(target_model_id="model-b")
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key_a))
            cache = DFlashPrefixCache(max_entries=4, max_bytes=10**9, l2=l2)
            matched, hydrated = cache.lookup([1, 2, 3], key_b)
            assert matched == 0
            assert hydrated is None
            stats = cache.stats()
            assert stats["l2_hits"] == 0
            assert stats["l2_misses"] == 1
        finally:
            l2.shutdown()

    def test_l2_lookup_miss_when_token_prefix_diverges(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3, 4, 5], key))
            cache = DFlashPrefixCache(max_entries=4, max_bytes=10**9, l2=l2)
            matched, hydrated = cache.lookup([1, 2, 99, 4, 5], key)
            assert matched == 0
            assert hydrated is None
        finally:
            l2.shutdown()

    def test_exact_hit_requires_prefill_kind(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            snap = _make_synthetic_snapshot([1, 2, 3], key)
            snap.kind = "generation"
            snap.last_logits = None
            l2._write_one(snap)
            res = l2.lookup((1, 2, 3), key)
            assert res is None
        finally:
            l2.shutdown()

    def test_prefix_hit_accepts_generation_kind(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            snap = _make_synthetic_snapshot([1, 2, 3], key)
            snap.kind = "generation"
            snap.last_logits = None
            l2._write_one(snap)

            res = l2.lookup((1, 2, 3, 4, 5), key)
            assert res is not None
            assert res.token_ids == (1, 2, 3)
        finally:
            l2.shutdown()

class TestFailureModes:
    def test_corrupt_file_rejected_and_deleted(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            bucket = _bucket_dir(tmp_path, key)
            bucket.mkdir(parents=True, exist_ok=True)

            name = _format_filename(
                token_len=3,
                token_hash=_token_hash([1, 2, 3]),
                kind="prefill",
                fp_short="0" * 16,
            )
            bad = bucket / name
            bad.write_bytes(b"not a safetensors file")
            cache = DFlashPrefixCache(max_entries=4, max_bytes=10**9, l2=l2)
            matched, hydrated = cache.lookup([1, 2, 3], key)
            assert matched == 0
            assert hydrated is None
            assert not bad.exists()
            stats = l2.stats()
            assert stats["load_errors"] >= 1
        finally:
            l2.shutdown()

    def test_schema_version_mismatch_rejected(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            snap = _make_synthetic_snapshot([1, 2], key)
            arrays, meta_dict = _serialize(snap)
            tampered = json.loads(meta_dict["dflash_meta"])
            tampered["schema_version"] = L2_SCHEMA_VERSION + 99
            meta_dict["dflash_meta"] = json.dumps(tampered)
            bucket = _bucket_dir(tmp_path, key)
            bucket.mkdir(parents=True, exist_ok=True)
            stale = bucket / _format_filename(
                token_len=2,
                token_hash=_token_hash([1, 2]),
                kind="prefill",
                fp_short="0" * 16,
            )
            mx.save_safetensors(str(stale), arrays, metadata=meta_dict)
            cache = DFlashPrefixCache(max_entries=4, max_bytes=10**9, l2=l2)
            matched, _ = cache.lookup([1, 2], key)
            assert matched == 0
            assert not stale.exists()
            stats = l2.stats()
            assert stats["schema_rejects"] >= 1
        finally:
            l2.shutdown()

    def test_runtime_version_mismatch_rejected(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            snap = _make_synthetic_snapshot([1, 2], key)
            arrays, meta_dict = _serialize(snap)
            tampered = json.loads(meta_dict["dflash_meta"])
            tampered["runtime_version"] = "0.0.0-not-a-real-version"
            meta_dict["dflash_meta"] = json.dumps(tampered)
            bucket = _bucket_dir(tmp_path, key)
            bucket.mkdir(parents=True, exist_ok=True)
            stale = bucket / _format_filename(
                token_len=2,
                token_hash=_token_hash([1, 2]),
                kind="prefill",
                fp_short="0" * 16,
            )
            mx.save_safetensors(str(stale), arrays, metadata=meta_dict)
            cache = DFlashPrefixCache(max_entries=4, max_bytes=10**9, l2=l2)
            matched, _ = cache.lookup([1, 2], key)
            assert matched == 0
            assert not stale.exists()
            stats = l2.stats()
            assert stats["schema_rejects"] >= 1
        finally:
            l2.shutdown()

    def test_disk_full_does_not_crash(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            import errno

            def _enospc(*_args, **_kwargs):
                raise OSError(errno.ENOSPC, "disk full (test)")

            monkeypatch.setattr("mlx.core.save_safetensors", _enospc)
            cache = DFlashPrefixCache(max_entries=1, max_bytes=10**9, l2=l2)
            key = _make_key()
            cache.insert(_make_synthetic_snapshot([1, 2, 3], key))
            cache.insert(_make_synthetic_snapshot([4, 5, 6], key))
            time.sleep(0.5)
            stats = l2.stats()
            assert stats["write_errors"] >= 1
            assert stats["writes"] == 0
        finally:
            l2.shutdown()

class TestEvictionAndStats:
    def test_byte_budget_evicts_oldest(self, tmp_path):
        probe_l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            probe_l2._write_one(_make_synthetic_snapshot([0, 0, 0], _make_key()))
            probe_files = _all_snapshot_files(tmp_path)
            on_disk = probe_files[0].stat().st_size
        finally:
            probe_l2.shutdown()

        for f in _all_snapshot_files(tmp_path):
            f.unlink()
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=int(on_disk * 1.5))
        try:
            for i in range(4):
                tokens = list(range(i * 10 + 1, i * 10 + 4))
                l2._write_one(_make_synthetic_snapshot(tokens, _make_key()))
            files = _all_snapshot_files(tmp_path)
            assert len(files) <= 2
            stats = l2.stats()
            assert stats["evictions"] >= 1
        finally:
            l2.shutdown()

    def test_stats_surfaced_through_l1(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            cache = DFlashPrefixCache(max_entries=1, max_bytes=10**9, l2=l2)
            key = _make_key()
            cache.insert(_make_synthetic_snapshot([1, 2, 3], key))
            cache.insert(_make_synthetic_snapshot([4, 5, 6], key))
            _wait_writes(l2, expected=1)
            cache.lookup([1, 2, 3], key)
            stats = cache.stats()
            assert "l2" in stats
            assert stats["l2"]["hits"] >= 1
        finally:
            l2.shutdown()

    def test_clear_removes_l2_files(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            cache = DFlashPrefixCache(max_entries=1, max_bytes=10**9, l2=l2)
            key = _make_key()
            cache.insert(_make_synthetic_snapshot([1, 2, 3], key))
            cache.insert(_make_synthetic_snapshot([4, 5, 6], key))
            _wait_writes(l2, expected=1)
            cache.clear()
            assert _all_snapshot_files(tmp_path) == []
        finally:
            l2.shutdown()

class TestAtomicity:
    def test_no_partial_files_left_after_normal_write(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            tmp_files = list((tmp_path / L2_LAYOUT_ROOT).rglob("*.tmp.safetensors"))
            assert tmp_files == []
            assert len(_all_snapshot_files(tmp_path)) == 1
        finally:
            l2.shutdown()

class TestConcurrency:
    def test_second_instance_falls_back_to_read_only(self, tmp_path):
        first = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            assert first.writable
            second = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
            try:
                assert not second.writable
                key = _make_key()
                first._write_one(_make_synthetic_snapshot([1, 2, 3], key))
                snap = second.lookup((1, 2, 3), key)
                assert snap is not None
                assert snap.token_ids == (1, 2, 3)
            finally:
                second.shutdown()
        finally:
            first.shutdown()

    def test_second_instance_silently_drops_writes(self, tmp_path):
        first = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            second = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
            try:
                key = _make_key()
                second.insert_async(_make_synthetic_snapshot([9, 9, 9], key))
                time.sleep(0.5)
                assert _all_snapshot_files(tmp_path) == []
            finally:
                second.shutdown()
        finally:
            first.shutdown()

class TestLookupHotPath:
    def test_many_bad_files_zero_loads(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            bucket = _bucket_dir(tmp_path, key)
            bucket.mkdir(parents=True, exist_ok=True)

            import secrets
            for i in range(100):
                name = _format_filename(
                    token_len=3,
                    token_hash=secrets.token_hex(8),
                    kind="prefill",
                    fp_short=secrets.token_hex(8),
                )
                (bucket / name).write_bytes(b"\x00")

            calls = {"n": 0}
            real_load = mx.load

            def _counting_load(*args, **kwargs):
                calls["n"] += 1
                return real_load(*args, **kwargs)

            monkeypatch.setattr("mlx.core.load", _counting_load)

            monkeypatch.setattr("dflash_mlx.cache.prefix_l2.mx.load", _counting_load)

            res = l2.lookup((1, 2, 3), key)
            assert res is None
            assert calls["n"] == 0, (
                f"lookup called mx.load {calls['n']} times; expected 0 — "
                "filename hash filter is broken"
            )
            stats = l2.stats()
            assert stats["lookup_hash_filtered"] >= 100
            assert stats["lookup_loads"] == 0
        finally:
            l2.shutdown()

    def test_single_good_file_one_load(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([5, 6, 7, 8], key))
            bucket = _bucket_dir(tmp_path, key)
            import secrets
            for _ in range(50):
                name = _format_filename(
                    token_len=4,
                    token_hash=secrets.token_hex(8),
                    kind="prefill",
                    fp_short=secrets.token_hex(8),
                )
                (bucket / name).write_bytes(b"\x00")

            calls = {"n": 0}
            real_load = mx.load

            def _counting_load(*args, **kwargs):
                calls["n"] += 1
                return real_load(*args, **kwargs)

            monkeypatch.setattr("dflash_mlx.cache.prefix_l2.mx.load", _counting_load)

            res = l2.lookup((5, 6, 7, 8), key)
            assert res is not None
            assert res.token_ids == (5, 6, 7, 8)
            assert calls["n"] == 1, f"expected exactly 1 mx.load, got {calls['n']}"
        finally:
            l2.shutdown()

def _stop_writer(l2):
    l2._stop.set()
    l2._write_queue.put(None)
    if l2._writer_thread is not None:
        l2._writer_thread.join(timeout=2.0)

class TestEpochInvalidation:
    def test_clear_during_pending_write_drops_payload(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            _stop_writer(l2)
            l2._stop.clear()

            key = _make_key()
            l2.insert_async(_make_synthetic_snapshot([1, 2, 3], key))
            assert l2._write_queue.qsize() == 1

            l2.clear()

            payload = l2._write_queue.get_nowait()
            l2._write_payload(payload)

            assert _all_snapshot_files(tmp_path) == []
            stats = l2.stats()
            assert stats["write_drops_epoch_invalidated"] >= 1
            assert stats["writes"] == 0
        finally:
            l2.shutdown()

    def test_writes_after_clear_use_new_epoch(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            assert len(_all_snapshot_files(tmp_path)) == 1
            l2.clear()
            assert _all_snapshot_files(tmp_path) == []
            l2._write_one(_make_synthetic_snapshot([7, 8, 9], key))
            assert len(_all_snapshot_files(tmp_path)) == 1
        finally:
            l2.shutdown()

class TestQueueBackpressure:
    def test_queue_full_drops_with_stat(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            _stop_writer(l2)

            key = _make_key()
            l2.insert_async(_make_synthetic_snapshot([1, 2, 3], key))
            l2.insert_async(_make_synthetic_snapshot([4, 5, 6], key))
            stats = l2.stats()
            assert stats["write_drops_queue_full"] >= 1
        finally:
            l2.shutdown()

    def test_default_max_in_flight_is_one(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            assert l2._max_in_flight == 1
        finally:
            l2.shutdown()

    def test_drop_does_not_call_mx_eval(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            _stop_writer(l2)
            key = _make_key()

            snap1 = _make_synthetic_snapshot([1, 2, 3], key)
            snap2 = _make_synthetic_snapshot([4, 5, 6], key)

            l2.insert_async(snap1)

            eval_calls = {"n": 0}
            from dflash_mlx.cache import prefix_l2 as _l2mod

            real_eval = _l2mod.mx.eval

            def _spy_eval(*args, **kwargs):
                eval_calls["n"] += 1
                return real_eval(*args, **kwargs)

            monkeypatch.setattr("dflash_mlx.cache.prefix_l2.mx.eval", _spy_eval)

            l2.insert_async(snap2)
            assert eval_calls["n"] == 0, (
                f"insert_async called mx.eval {eval_calls['n']} times "
                "after the slot was full — drop happens AFTER materialize"
            )
            stats = l2.stats()
            assert stats["write_drops_queue_full"] >= 1
        finally:
            l2.shutdown()

class TestReadOnlyInstance:
    def test_read_only_does_not_spawn_writer_thread(self, tmp_path):
        first = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            assert first._writer_thread is not None
            second = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
            try:
                assert not second.writable
                assert second._writer_thread is None
            finally:
                second.shutdown()
        finally:
            first.shutdown()

class TestStatsHotPath:
    def test_stats_does_not_walk_filesystem(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            l2._write_one(_make_synthetic_snapshot([4, 5, 6], key))

            calls = {"n": 0}
            real_walk = l2._walk_snapshots

            def _spy_walk():
                calls["n"] += 1
                return real_walk()

            l2._walk_snapshots = _spy_walk
            for _ in range(10):
                stats = l2.stats()
                assert stats["current_bytes"] > 0
            assert calls["n"] == 0, (
                f"stats() called _walk_snapshots {calls['n']} times — "
                "tracked counter is not used on the hot path"
            )
        finally:
            l2.shutdown()

    def test_tracked_bytes_matches_disk(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            l2._write_one(_make_synthetic_snapshot([4, 5, 6], key))
            l2._write_one(_make_synthetic_snapshot([7, 8, 9], key))
            on_disk = sum(p.stat().st_size for p in _all_snapshot_files(tmp_path))
            assert l2.stats()["current_bytes"] == on_disk
            l2.clear()
            assert l2.stats()["current_bytes"] == 0
        finally:
            l2.shutdown()

    def test_tracked_bytes_decremented_on_corrupt_unlink(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            initial = l2.stats()["current_bytes"]
            assert initial > 0

            bucket = _bucket_dir(tmp_path, key)
            corrupt_name = _format_filename(
                token_len=2,
                token_hash=_token_hash([1, 2]),
                kind="prefill",
                fp_short="0" * 16,
            )
            corrupt = bucket / corrupt_name
            corrupt.write_bytes(b"deadbeef" * 100)
            after_inject = l2.stats()["current_bytes"]

            l2.lookup((1, 2), key)
            assert not corrupt.exists()
            after = l2.stats()["current_bytes"]

            assert after >= 0

            assert after <= after_inject
        finally:
            l2.shutdown()

class TestZeroCopyWrite:
    def test_writer_uses_mx_save_safetensors(self, tmp_path, monkeypatch):
        import mlx.core as _mx
        import numpy as _np

        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            calls = {"n": 0}
            real_save = _mx.save_safetensors

            def _spy(path, arrays, metadata=None):
                calls["n"] += 1

                for v in arrays.values():
                    assert isinstance(v, _mx.array), (
                        f"writer must pass mx.array, got {type(v).__name__}"
                    )
                return real_save(path, arrays, metadata=metadata)

            monkeypatch.setattr("mlx.core.save_safetensors", _spy)

            real_np_array = _np.array

            def _forbid_copy(*args, **kwargs):
                if kwargs.get("copy") is True:
                    raise AssertionError(
                        "L2 writer regressed: np.array(copy=True) reintroduced"
                    )
                return real_np_array(*args, **kwargs)

            monkeypatch.setattr("numpy.array", _forbid_copy)

            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            assert calls["n"] >= 1
            assert len(_all_snapshot_files(tmp_path)) == 1
        finally:
            l2.shutdown()

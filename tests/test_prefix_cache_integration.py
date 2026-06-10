# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from __future__ import annotations

import threading
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from dflash_mlx.cache.codecs import (
    PrefixSnapshotBuilder,
    build_snapshot,
)
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.manager import (
    PrefixCacheLookupResult,
    RuntimeCacheManager,
    RuntimeCacheManagerClosed,
)
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.prefix_l2 import DFlashPrefixL2Cache
from dflash_mlx.cache.snapshot_service import SnapshotService
from dflash_mlx.cache.store import PrefixSnapshotStore
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
from dflash_mlx.runtime.config import runtime_config_from_defaults
from dflash_mlx.runtime.context import build_runtime_context

def _runtime_context(**overrides):
    values = dict(
        prefix_cache=True,
        prefix_cache_l2=False,
        prefix_cache_l2_dir="/tmp/dflash-prefix-l2-test",
    )
    values.update(overrides)
    return build_runtime_context(runtime_config_from_defaults(**values))

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
        template_hash="a" * 64,
        prompt_policy_hash="b" * 64,
    )
    base.update(overrides)
    return DFlashPrefixKey(**base)


class _FakeTokenizer:
    unk_token_id = -1
    chat_template = (
        "{% for message in messages %}"
        "{{ message.role }}:{{ message.content }}"
        "{% endfor %}"
    )
    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"
    has_thinking = True
    think_start = "<think>"
    think_end = "</think>"

    def __init__(
        self,
        *,
        chat_template=None,
        has_tool_calling=True,
        tool_call_start="<tool_call>",
        tool_call_end="</tool_call>",
    ):
        if chat_template is not None:
            self.chat_template = chat_template
        self.has_tool_calling = has_tool_calling
        self.tool_call_start = tool_call_start
        self.tool_call_end = tool_call_end

    def convert_tokens_to_ids(self, tokens):
        ids = {
            "<|im_start|>": 151644,
            "assistant": 77091,
        }
        return [ids.get(token, self.unk_token_id) for token in tokens]


class _FakeLoadedProvider:
    def __init__(self, *, tokenizer=None, chat_template_args=None):
        self.model_key = ("target/x", None, "draft/y")
        self.tokenizer = tokenizer if tokenizer is not None else _FakeTokenizer()
        self.cli_args = SimpleNamespace(
            chat_template_args=(
                chat_template_args if chat_template_args is not None else {"enable_thinking": False}
            )
        )


class _FakeDraft:
    target_layer_ids = [3, 7]


def _snapshot_service(
    cache: DFlashPrefixCache | PrefixSnapshotStore,
    key: DFlashPrefixKey,
    *,
    builder=None,
) -> SnapshotService:
    builder = builder if builder is not None else PrefixSnapshotBuilder(key=key)
    store = cache if isinstance(cache, PrefixSnapshotStore) else _store(cache)
    return SnapshotService(cache_manager=RuntimeCacheManager(store), builder=builder)

def _store(
    cache: DFlashPrefixCache | None = None,
    *,
    l2=None,
    max_entries: int = 4,
    max_bytes: int = 8 * 1024 * 1024 * 1024,
) -> PrefixSnapshotStore:
    l1 = cache if cache is not None else DFlashPrefixCache(
        max_entries=max_entries,
        max_bytes=max_bytes,
    )
    return PrefixSnapshotStore(
        l1=l1,
        l2=l2,
    )


def _manager(
    cache: DFlashPrefixCache | None = None,
    *,
    l2=None,
    max_entries: int = 4,
    max_bytes: int = 8 * 1024 * 1024 * 1024,
) -> RuntimeCacheManager:
    return RuntimeCacheManager(
        _store(cache, l2=l2, max_entries=max_entries, max_bytes=max_bytes)
    )

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

    snap = build_snapshot(
        token_ids=prompt_tokens,
        target_cache=live_cache,
        target_hidden=target_hidden,
        last_logits=last_logits,
        key=key,
    )
    cache.insert(snap)
    return snap, live_cache

class TestServeHelperShapes:

    def test_get_cache_disabled(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        assert cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=False)
        ) is None

    def test_get_cache_enabled_returns_singleton(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        context = _runtime_context(prefix_cache=True)
        first = cache_manager_mod.get_runtime_cache_manager(context)
        second = cache_manager_mod.get_runtime_cache_manager(context)
        assert first is not None
        assert first is second

    def test_build_prefix_key_from_provider(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        key = build_prefix_key(_FakeLoadedProvider(), _FakeDraft(), _runtime_context())
        assert key.target_model_id == "target/x"
        assert key.draft_model_id == "draft/y"
        assert key.capture_layer_ids == (3, 7)
        assert isinstance(key.draft_sink_size, int)
        assert isinstance(key.draft_window_size, int)
        assert key.format_version == 3
        assert len(key.template_hash) == 64
        assert len(key.prompt_policy_hash) == 64

    def test_build_prefix_key_uses_runtime_draft_window(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        key_default = build_prefix_key(
            _FakeLoadedProvider(),
            _FakeDraft(),
            _runtime_context(draft_sink_size=64, draft_window_size=1024),
        )
        key_windowed = build_prefix_key(
            _FakeLoadedProvider(),
            _FakeDraft(),
            _runtime_context(draft_sink_size=32, draft_window_size=512),
        )

        assert key_default.draft_sink_size == 64
        assert key_default.draft_window_size == 1024
        assert key_windowed.draft_sink_size == 32
        assert key_windowed.draft_window_size == 512
        assert key_default != key_windowed

    def test_build_prefix_key_records_template_identity(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        first = build_prefix_key(
            _FakeLoadedProvider(tokenizer=_FakeTokenizer(chat_template="template-a")),
            _FakeDraft(),
            _runtime_context(),
        )
        second = build_prefix_key(
            _FakeLoadedProvider(tokenizer=_FakeTokenizer(chat_template="template-b")),
            _FakeDraft(),
            _runtime_context(),
        )

        assert first.template_hash != second.template_hash
        assert first.prompt_policy_hash == second.prompt_policy_hash
        assert first != second

    def test_build_prefix_key_records_prompt_policy_identity(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        disabled = build_prefix_key(
            _FakeLoadedProvider(chat_template_args={"enable_thinking": False}),
            _FakeDraft(),
            _runtime_context(),
        )
        enabled = build_prefix_key(
            _FakeLoadedProvider(chat_template_args={"enable_thinking": True}),
            _FakeDraft(),
            _runtime_context(),
        )

        assert disabled.template_hash == enabled.template_hash
        assert disabled.prompt_policy_hash != enabled.prompt_policy_hash
        assert disabled != enabled

    def test_build_prefix_key_records_tool_marker_identity(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        plain = build_prefix_key(
            _FakeLoadedProvider(
                tokenizer=_FakeTokenizer(
                    tool_call_start="<function_call>",
                    tool_call_end="</function_call>",
                )
            ),
            _FakeDraft(),
            _runtime_context(),
        )
        tool_aware = build_prefix_key(
            _FakeLoadedProvider(tokenizer=_FakeTokenizer(has_tool_calling=True)),
            _FakeDraft(),
            _runtime_context(),
        )

        assert plain.template_hash == tool_aware.template_hash
        assert plain.prompt_policy_hash != tool_aware.prompt_policy_hash
        assert plain != tool_aware

    def test_build_prefix_key_requires_loaded_provider_and_draft(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        class FakeProvider:
            pass

        class FakeDraft:
            target_layer_ids = ()

        with pytest.raises(ValueError, match="model_key"):
            build_prefix_key(FakeProvider(), FakeDraft(), _runtime_context())

        with pytest.raises(ValueError, match="must not be empty"):
            build_prefix_key(_FakeLoadedProvider(), FakeDraft(), _runtime_context())

        provider = _FakeLoadedProvider()
        provider.tokenizer = None
        with pytest.raises(ValueError, match="tokenizer"):
            build_prefix_key(provider, _FakeDraft(), _runtime_context())

    def test_build_prefix_key_requires_runtime_identity_fields(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        with pytest.raises(ValueError, match="runtime context"):
            build_prefix_key(_FakeLoadedProvider(), _FakeDraft(), object())

class TestContextConfigExposedCorrectly:

    def test_prefix_cache_config_affects_singleton(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        assert cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=False)
        ) is None

        assert cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True)
        ) is not None

    def test_singleton_rebuilds_when_config_changes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        first = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=2)
        )
        second = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=7)
        )

        assert first is not None
        assert second is not None
        assert first is not second
        assert second.stats()["max_entries"] == 7

    def test_singleton_rebuild_shutdowns_previous_cache(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        shutdown_calls: list[DFlashPrefixCache] = []
        original_shutdown = DFlashPrefixCache.shutdown

        def tracked_shutdown(self):
            shutdown_calls.append(self)
            return original_shutdown(self)

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        monkeypatch.setattr(DFlashPrefixCache, "shutdown", tracked_shutdown)

        first = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=2)
        )
        second = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=7)
        )

        assert first is not None
        assert second is not None
        assert first is not second
        assert len(shutdown_calls) == 1

    def test_singleton_rebuild_fails_when_previous_cache_shutdown_fails(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        make_calls = []

        class BrokenShutdownCache(DFlashPrefixCache):
            def shutdown(self):
                raise RuntimeError("cannot close")

        def make_cache(_runtime_context):
            make_calls.append(_runtime_context)
            return _store(max_entries=2)

        old_manager = RuntimeCacheManager(_store(BrokenShutdownCache(max_entries=2)))
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", ("old",))
        monkeypatch.setattr(cache_manager_mod, "_make_prefix_store", make_cache)

        with pytest.raises(RuntimeError, match="cannot close"):
            cache_manager_mod.get_runtime_cache_manager(_runtime_context(prefix_cache=True))

        assert cache_manager_mod.current_runtime_cache_manager() is None
        assert cache_manager_mod._DFLASH_RUNTIME_CACHE_MANAGER is old_manager
        with pytest.raises(RuntimeError, match="shut down"):
            old_manager.stats()
        assert make_calls == []
        with pytest.raises(RuntimeError, match="cannot close"):
            cache_manager_mod.get_runtime_cache_manager(_runtime_context(prefix_cache=True))
        assert cache_manager_mod.current_runtime_cache_manager() is None
        assert cache_manager_mod._DFLASH_RUNTIME_CACHE_MANAGER is old_manager
        assert make_calls == []

    def test_retired_manager_handle_fails_after_clear(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        context = _runtime_context(prefix_cache=True)
        manager = cache_manager_mod.get_runtime_cache_manager(context)
        assert manager is not None
        ready = threading.Event()
        proceed = threading.Event()
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                handle = cache_manager_mod.get_runtime_cache_manager(context)
                assert handle is manager
                ready.set()
                assert proceed.wait(timeout=2.0)
                with pytest.raises(RuntimeError, match="shut down"):
                    handle.stats()
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        assert ready.wait(timeout=2.0)
        cache_manager_mod._clear_runtime_cache_manager()
        proceed.set()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert errors == []
        assert cache_manager_mod.current_runtime_cache_manager() is None

    def test_clear_blocks_rebuild_until_previous_shutdown_finishes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        shutdown_entered = threading.Event()
        release_shutdown = threading.Event()
        build_finished = threading.Event()
        make_calls = []
        build_results = []
        errors: list[BaseException] = []

        class BlockingShutdownCache(DFlashPrefixCache):
            def shutdown(self):
                shutdown_entered.set()
                assert release_shutdown.wait(timeout=2.0)
                return super().shutdown()

        def make_cache(_runtime_context):
            make_calls.append(_runtime_context)
            return _store(max_entries=2)

        old_manager = RuntimeCacheManager(_store(BlockingShutdownCache(max_entries=2)))
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", ("old",))
        monkeypatch.setattr(cache_manager_mod, "_make_prefix_store", make_cache)
        context = _runtime_context(prefix_cache=True)

        def clear_worker() -> None:
            try:
                cache_manager_mod._clear_runtime_cache_manager()
            except BaseException as exc:
                errors.append(exc)

        def build_worker() -> None:
            try:
                build_results.append(cache_manager_mod.get_runtime_cache_manager(context))
            except BaseException as exc:
                errors.append(exc)
            finally:
                build_finished.set()

        clear_thread = threading.Thread(target=clear_worker)
        clear_thread.start()
        assert shutdown_entered.wait(timeout=2.0)

        build_thread = threading.Thread(target=build_worker)
        build_thread.start()
        assert not build_finished.wait(timeout=0.05)
        assert make_calls == []

        release_shutdown.set()
        clear_thread.join(timeout=2.0)
        build_thread.join(timeout=2.0)

        assert not clear_thread.is_alive()
        assert not build_thread.is_alive()
        assert errors == []
        assert make_calls == [context]
        assert build_results == [cache_manager_mod.current_runtime_cache_manager()]

    def test_get_reuse_blocks_clear_until_trace_update_finishes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        trace_entered = threading.Event()
        release_trace = threading.Event()
        clear_started = threading.Event()
        errors: list[BaseException] = []
        get_results = []

        class BlockingTraceCache(DFlashPrefixCache):
            def set_trace_config(self, trace_config):
                trace_entered.set()
                assert release_trace.wait(timeout=2.0)
                return super().set_trace_config(trace_config)

        context = _runtime_context(prefix_cache=True)
        old_manager = _manager(BlockingTraceCache(max_entries=2))
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(
            cache_manager_mod,
            "_DFLASH_RUNTIME_CACHE_CONFIG_KEY",
            cache_manager_mod._prefix_cache_config_key(context),
        )

        def get_worker() -> None:
            try:
                get_results.append(cache_manager_mod.get_runtime_cache_manager(context))
            except BaseException as exc:
                errors.append(exc)

        def clear_worker() -> None:
            try:
                clear_started.set()
                cache_manager_mod._clear_runtime_cache_manager()
            except BaseException as exc:
                errors.append(exc)

        get_thread = threading.Thread(target=get_worker)
        get_thread.start()
        assert trace_entered.wait(timeout=2.0)

        clear_thread = threading.Thread(target=clear_worker)
        clear_thread.start()
        assert clear_started.wait(timeout=2.0)
        clear_thread.join(timeout=0.05)
        assert clear_thread.is_alive()

        release_trace.set()
        get_thread.join(timeout=2.0)
        clear_thread.join(timeout=2.0)

        assert not get_thread.is_alive()
        assert not clear_thread.is_alive()
        assert errors == []
        assert get_results == [old_manager]
        assert cache_manager_mod.current_runtime_cache_manager() is None

    def test_sync_reuse_blocks_clear_until_trace_update_finishes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        trace_entered = threading.Event()
        release_trace = threading.Event()
        clear_started = threading.Event()
        errors: list[BaseException] = []
        sync_results = []

        class BlockingTraceCache(DFlashPrefixCache):
            def set_trace_config(self, trace_config):
                trace_entered.set()
                assert release_trace.wait(timeout=2.0)
                return super().set_trace_config(trace_config)

        context = _runtime_context(prefix_cache=True)
        old_manager = _manager(BlockingTraceCache(max_entries=2))
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(
            cache_manager_mod,
            "_DFLASH_RUNTIME_CACHE_CONFIG_KEY",
            cache_manager_mod._prefix_cache_config_key(context),
        )

        def sync_worker() -> None:
            try:
                sync_results.append(cache_manager_mod.sync_runtime_cache_manager(context))
            except BaseException as exc:
                errors.append(exc)

        def clear_worker() -> None:
            try:
                clear_started.set()
                cache_manager_mod._clear_runtime_cache_manager()
            except BaseException as exc:
                errors.append(exc)

        sync_thread = threading.Thread(target=sync_worker)
        sync_thread.start()
        assert trace_entered.wait(timeout=2.0)

        clear_thread = threading.Thread(target=clear_worker)
        clear_thread.start()
        assert clear_started.wait(timeout=2.0)
        clear_thread.join(timeout=0.05)
        assert clear_thread.is_alive()

        release_trace.set()
        sync_thread.join(timeout=2.0)
        clear_thread.join(timeout=2.0)

        assert not sync_thread.is_alive()
        assert not clear_thread.is_alive()
        assert errors == []
        assert sync_results == [old_manager]
        assert cache_manager_mod.current_runtime_cache_manager() is None

    def test_clear_blocks_until_inflight_lookup_finishes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        lookup_entered = threading.Event()
        release_lookup = threading.Event()
        clear_started = threading.Event()
        errors: list[BaseException] = []
        lookup_results = []
        key = _make_key()

        class BlockingLookupCache(DFlashPrefixCache):
            def lookup(self, req_tokens, lookup_key):
                lookup_entered.set()
                assert release_lookup.wait(timeout=2.0)
                return super().lookup(req_tokens, lookup_key)

        context = _runtime_context(prefix_cache=True)
        old_manager = _manager(BlockingLookupCache(max_entries=2))
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(
            cache_manager_mod,
            "_DFLASH_RUNTIME_CACHE_CONFIG_KEY",
            cache_manager_mod._prefix_cache_config_key(context),
        )

        def lookup_worker() -> None:
            try:
                lookup_results.append(old_manager.lookup([1, 2, 3], key))
            except BaseException as exc:
                errors.append(exc)

        def clear_worker() -> None:
            try:
                clear_started.set()
                cache_manager_mod._clear_runtime_cache_manager()
            except BaseException as exc:
                errors.append(exc)

        lookup_thread = threading.Thread(target=lookup_worker)
        lookup_thread.start()
        assert lookup_entered.wait(timeout=2.0)

        clear_thread = threading.Thread(target=clear_worker)
        clear_thread.start()
        assert clear_started.wait(timeout=2.0)
        clear_thread.join(timeout=0.05)
        assert clear_thread.is_alive()

        release_lookup.set()
        lookup_thread.join(timeout=2.0)
        clear_thread.join(timeout=2.0)

        assert not lookup_thread.is_alive()
        assert not clear_thread.is_alive()
        assert errors == []
        assert len(lookup_results) == 1
        assert lookup_results[0].matched_tokens == 0
        assert cache_manager_mod.current_runtime_cache_manager() is None

    def test_clear_blocks_rebuild_until_inflight_insert_finishes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        insert_entered = threading.Event()
        release_insert = threading.Event()
        build_finished = threading.Event()
        errors: list[BaseException] = []
        insert_results = []
        build_results = []
        make_calls = []
        key = _make_key()
        prompt = [1, 2, 3]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        snap = build_snapshot(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            key=key,
        )

        class BlockingInsertCache(DFlashPrefixCache):
            def insert_with_evictions(self, snapshot, *, skip_too_long=True):
                insert_entered.set()
                assert release_insert.wait(timeout=2.0)
                return super().insert_with_evictions(
                    snapshot,
                    skip_too_long=skip_too_long,
                )

        def make_cache(_runtime_context):
            make_calls.append(_runtime_context)
            return _store(max_entries=2)

        context = _runtime_context(prefix_cache=True)
        old_manager = _manager(BlockingInsertCache(max_entries=2))
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", ("old",))
        monkeypatch.setattr(cache_manager_mod, "_make_prefix_store", make_cache)

        def insert_worker() -> None:
            try:
                insert_results.append(
                    old_manager.maybe_insert_snapshot(
                        snap,
                        key=key,
                        kind="prefill",
                        require_logits=True,
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        def clear_worker() -> None:
            try:
                cache_manager_mod._clear_runtime_cache_manager()
            except BaseException as exc:
                errors.append(exc)

        def build_worker() -> None:
            try:
                build_results.append(cache_manager_mod.get_runtime_cache_manager(context))
            except BaseException as exc:
                errors.append(exc)
            finally:
                build_finished.set()

        insert_thread = threading.Thread(target=insert_worker)
        insert_thread.start()
        assert insert_entered.wait(timeout=2.0)

        clear_thread = threading.Thread(target=clear_worker)
        clear_thread.start()
        build_thread = threading.Thread(target=build_worker)
        build_thread.start()
        assert not build_finished.wait(timeout=0.05)
        assert make_calls == []

        release_insert.set()
        insert_thread.join(timeout=2.0)
        clear_thread.join(timeout=2.0)
        build_thread.join(timeout=2.0)

        assert not insert_thread.is_alive()
        assert not clear_thread.is_alive()
        assert not build_thread.is_alive()
        assert errors == []
        assert len(insert_results) == 1
        assert make_calls == [context]
        assert build_results == [cache_manager_mod.current_runtime_cache_manager()]

    def test_current_manager_blocks_until_clear_finishes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        shutdown_entered = threading.Event()
        release_shutdown = threading.Event()
        current_finished = threading.Event()
        current_results = []
        errors: list[BaseException] = []

        class BlockingShutdownCache(DFlashPrefixCache):
            def shutdown(self):
                shutdown_entered.set()
                assert release_shutdown.wait(timeout=2.0)
                return super().shutdown()

        old_manager = _manager(BlockingShutdownCache(max_entries=2))
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", ("old",))

        def clear_worker() -> None:
            try:
                cache_manager_mod._clear_runtime_cache_manager()
            except BaseException as exc:
                errors.append(exc)

        def current_worker() -> None:
            try:
                current_results.append(cache_manager_mod.current_runtime_cache_manager())
            except BaseException as exc:
                errors.append(exc)
            finally:
                current_finished.set()

        clear_thread = threading.Thread(target=clear_worker)
        clear_thread.start()
        assert shutdown_entered.wait(timeout=2.0)

        current_thread = threading.Thread(target=current_worker)
        current_thread.start()
        assert not current_finished.wait(timeout=0.05)

        release_shutdown.set()
        clear_thread.join(timeout=2.0)
        current_thread.join(timeout=2.0)

        assert not clear_thread.is_alive()
        assert not current_thread.is_alive()
        assert errors == []
        assert current_results == [None]
        assert cache_manager_mod.current_runtime_cache_manager() is None

    def test_singleton_rebuild_construction_failure_clears_shutdown_previous_cache(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        shutdown_calls = []

        class TrackedShutdownCache(DFlashPrefixCache):
            def shutdown(self):
                shutdown_calls.append(self)
                return super().shutdown()

        old_cache = TrackedShutdownCache(max_entries=2)
        old_manager = _manager(old_cache)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", ("old",))
        monkeypatch.setattr(
            cache_manager_mod,
            "_make_prefix_store",
            lambda _runtime_context: (_ for _ in ()).throw(RuntimeError("build failed")),
        )

        with pytest.raises(RuntimeError, match="build failed"):
            cache_manager_mod.get_runtime_cache_manager(_runtime_context(prefix_cache=True))

        assert cache_manager_mod.current_runtime_cache_manager() is None
        assert shutdown_calls == [old_cache]

    def test_l2_rebuild_releases_old_lock_before_new_manager(self, monkeypatch, tmp_path):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        context = _runtime_context(
            prefix_cache=True,
            prefix_cache_l2=True,
            prefix_cache_l2_dir=str(tmp_path / "l2"),
        )

        try:
            first = cache_manager_mod.get_runtime_cache_manager(context, cache_identity="model-a")
            assert first is not None
            assert first.stats()["l2"]["writable"] is True

            second = cache_manager_mod.get_runtime_cache_manager(context, cache_identity="model-b")
            assert second is not None
            assert second is not first
            assert second.stats()["l2"]["writable"] is True
        finally:
            cache_manager_mod.shutdown_runtime_cache_manager()

    def test_shutdown_runtime_cache_manager_shutdowns_current_cache(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        shutdown_calls: list[DFlashPrefixCache] = []
        original_shutdown = DFlashPrefixCache.shutdown

        def tracked_shutdown(self):
            shutdown_calls.append(self)
            return original_shutdown(self)

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        monkeypatch.setattr(DFlashPrefixCache, "shutdown", tracked_shutdown)

        manager = cache_manager_mod.get_runtime_cache_manager(_runtime_context(prefix_cache=True))
        assert manager is not None

        cache_manager_mod.shutdown_runtime_cache_manager()

        assert len(shutdown_calls) == 1
        assert cache_manager_mod.current_runtime_cache_manager() is None

    def test_shutdown_runtime_cache_manager_logs_failure(self, monkeypatch, capsys):
        import dflash_mlx.cache.manager as cache_manager_mod

        class BrokenShutdownCache(DFlashPrefixCache):
            def shutdown(self):
                raise RuntimeError("broken close")

        manager = _manager(BrokenShutdownCache(max_entries=2))
        monkeypatch.setattr(
            cache_manager_mod,
            "_DFLASH_RUNTIME_CACHE_MANAGER",
            manager,
        )
        monkeypatch.setattr(
            cache_manager_mod,
            "_DFLASH_RUNTIME_CACHE_CONFIG_KEY",
            ("old",),
        )

        cache_manager_mod.shutdown_runtime_cache_manager()

        assert "runtime cache manager shutdown failed: broken close" in capsys.readouterr().err
        assert cache_manager_mod.current_runtime_cache_manager() is None
        assert cache_manager_mod._DFLASH_RUNTIME_CACHE_MANAGER is manager
        with pytest.raises(RuntimeError, match="shut down"):
            manager.stats()

    def test_disabled_runtime_cleanup_failure_leaves_singleton_poisoned(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        class BrokenShutdownCache(DFlashPrefixCache):
            def shutdown(self):
                raise RuntimeError("cannot close")

        old_manager = _manager(BrokenShutdownCache(max_entries=2))
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", ("old",))

        with pytest.raises(RuntimeError, match="cannot close"):
            cache_manager_mod.get_runtime_cache_manager(_runtime_context(prefix_cache=False))

        assert cache_manager_mod.current_runtime_cache_manager() is None
        assert cache_manager_mod._DFLASH_RUNTIME_CACHE_MANAGER is old_manager
        with pytest.raises(RuntimeError, match="shut down"):
            old_manager.stats()

    def test_sync_retries_poisoned_shutdown_without_replacement(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        make_calls = []

        class BrokenShutdownCache(DFlashPrefixCache):
            def shutdown(self):
                raise RuntimeError("cannot close")

        def make_cache(_runtime_context):
            make_calls.append(_runtime_context)
            return _store(max_entries=2)

        context = _runtime_context(prefix_cache=True)
        old_manager = _manager(BrokenShutdownCache(max_entries=2))
        with pytest.raises(RuntimeError, match="cannot close"):
            old_manager.shutdown()
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(
            cache_manager_mod,
            "_DFLASH_RUNTIME_CACHE_CONFIG_KEY",
            cache_manager_mod._prefix_cache_config_key(context),
        )
        monkeypatch.setattr(cache_manager_mod, "_make_prefix_store", make_cache)

        with pytest.raises(RuntimeError, match="cannot close"):
            cache_manager_mod.sync_runtime_cache_manager(context)

        assert cache_manager_mod.current_runtime_cache_manager() is None
        assert cache_manager_mod._DFLASH_RUNTIME_CACHE_MANAGER is old_manager
        assert make_calls == []

    def test_sync_clears_poisoned_manager_after_successful_retry(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        shutdown_calls = []

        class FlakyShutdownCache(DFlashPrefixCache):
            def shutdown(self):
                shutdown_calls.append(self)
                if len(shutdown_calls) == 1:
                    raise RuntimeError("first close failed")
                return super().shutdown()

        context = _runtime_context(prefix_cache=True)
        old_manager = _manager(FlakyShutdownCache(max_entries=2))
        with pytest.raises(RuntimeError, match="first close failed"):
            old_manager.shutdown()
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(
            cache_manager_mod,
            "_DFLASH_RUNTIME_CACHE_CONFIG_KEY",
            cache_manager_mod._prefix_cache_config_key(context),
        )

        assert cache_manager_mod.sync_runtime_cache_manager(context) is None

        assert len(shutdown_calls) == 2
        assert cache_manager_mod.current_runtime_cache_manager() is None
        assert cache_manager_mod._DFLASH_RUNTIME_CACHE_MANAGER is None

    def test_chat_template_stable_marker_returns_none_without_converter(self):
        from dflash_mlx.server.prefix_cache_manager import chat_template_stable_marker

        assert chat_template_stable_marker(object()) == (None, None, 0)

    def test_chat_template_stable_marker_raises_on_tokenizer_failure(self):
        from dflash_mlx.server.prefix_cache_manager import chat_template_stable_marker

        class BrokenTokenizer:
            unk_token_id = -1

            def convert_tokens_to_ids(self, _tokens):
                raise RuntimeError("tokenizer closed")

        with pytest.raises(RuntimeError, match="chat template marker ids"):
            chat_template_stable_marker(BrokenTokenizer())

    def test_chat_template_stable_marker_falls_back_to_gemma_turn_model(self):
        from dflash_mlx.server.prefix_cache_manager import chat_template_stable_marker

        class GemmaTokenizer:
            unk_token_id = 3

            def convert_tokens_to_ids(self, tokens):
                if tokens == ["<|im_start|>", "assistant"]:
                    return [3, 111457]
                if tokens == ["<|turn>", "model"]:
                    return [105, 4368]
                return [3 for _ in tokens]

            def encode(self, text):
                if text == "<|turn>model\n":
                    return [105, 4368, 107]
                return [3]

        assert chat_template_stable_marker(GemmaTokenizer()) == (105, 4368, 3)

    def test_chat_template_stable_marker_keeps_chatml_boundary(self):
        from dflash_mlx.server.prefix_cache_manager import chat_template_stable_marker

        class QwenTokenizer:
            unk_token_id = -1

            def convert_tokens_to_ids(self, tokens):
                if tokens == ["<|im_start|>", "assistant"]:
                    return [900, 901]
                return [-1 for _ in tokens]

        assert chat_template_stable_marker(QwenTokenizer()) == (900, 901, 0)

    def test_budgets_propagate_to_cache(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        manager = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(
                prefix_cache=True,
                prefix_cache_max_entries=7,
                prefix_cache_max_bytes=1234567,
            )
        )
        assert manager is not None
        stats = manager.stats()
        assert stats["max_entries"] == 7
        assert stats["max_bytes"] == 1234567

    def test_l2_frontier_stride_has_disk_pressure_floor(self, monkeypatch, tmp_path):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        manager = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(
                prefix_cache=True,
                prefix_cache_l2=True,
                prefix_cache_l2_dir=str(tmp_path),
                prefill_step_size=4096,
            )
        )

        assert manager is not None
        assert manager.frontier_stride() == 8192

    def test_l2_frontier_stride_never_splits_prefill_chunks(self, monkeypatch, tmp_path):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        manager = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(
                prefix_cache=True,
                prefix_cache_l2=True,
                prefix_cache_l2_dir=str(tmp_path),
                prefill_step_size=16384,
            )
        )

        assert manager is not None
        assert manager.frontier_stride() == 16384

    def test_l2_frontier_stride_rounds_floor_to_chunk_multiple(self, monkeypatch, tmp_path):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        manager = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(
                prefix_cache=True,
                prefix_cache_l2=True,
                prefix_cache_l2_dir=str(tmp_path),
                prefill_step_size=5000,
            )
        )

        assert manager is not None
        assert manager.frontier_stride() == 10000

    def test_l2_frontier_stride_honors_larger_configured_value(
        self, monkeypatch, tmp_path
    ):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        context = _runtime_context(
            prefix_cache=True,
            prefix_cache_l2=True,
            prefix_cache_l2_dir=str(tmp_path),
            prefill_step_size=2048,
            prefix_cache_l2_frontier_stride=32000,
        )
        manager = cache_manager_mod.get_runtime_cache_manager(
            context,
            cache_identity=("configured-frontier",),
        )

        assert manager is not None
        assert manager.frontier_stride() == 32768

    def test_frontier_stride_disabled_without_l2(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        manager = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(
                prefix_cache=True,
                prefix_cache_l2=False,
                prefill_step_size=4096,
            )
        )

        assert manager is not None
        assert manager.frontier_stride() == 0

    def test_l2_filesystem_failure_falls_back_to_l1(self, monkeypatch, capsys):
        import dflash_mlx.cache.manager as cache_manager_mod

        def raise_os_error(**_kwargs):
            raise OSError("l2 unavailable")

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        monkeypatch.setattr(cache_manager_mod, "DFlashPrefixL2Cache", raise_os_error)

        manager = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True, prefix_cache_l2=True)
        )

        assert manager is not None
        assert "l2" not in manager.stats()
        assert "prefix L2 cache disabled: l2 unavailable" in capsys.readouterr().err

    def test_l2_constructor_bug_is_not_silently_downgraded(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        def raise_type_error(**_kwargs):
            raise TypeError("bad l2 constructor")

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        monkeypatch.setattr(cache_manager_mod, "DFlashPrefixL2Cache", raise_type_error)

        with pytest.raises(TypeError, match="bad l2 constructor"):
            cache_manager_mod.get_runtime_cache_manager(
                _runtime_context(prefix_cache=True, prefix_cache_l2=True)
            )

        assert cache_manager_mod.current_runtime_cache_manager() is None

    def test_manager_exposes_memory_waterfall_bytes_without_raw_cache(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [51, 52, 53]
        snap, _ = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=3)
        manager = _manager(cache)

        memory = manager.memory_waterfall_bytes()

        assert not hasattr(manager, "prefix_cache_for_memory")
        assert memory["l1_snapshot_bytes"] == snap.nbytes
        assert memory["l1_snapshot_target_hidden_bytes"] > 0
        assert memory["l1_snapshot_last_logits_bytes"] > 0

    def test_prefix_flow_memory_ignores_retired_cache_manager(self):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        manager = _manager(max_entries=4)
        flow = flow_mod.PrefixCacheFlow(cache_manager=manager)
        manager.shutdown()

        assert flow.prefix_cache_memory_bytes() is None

    def test_retired_manager_handle_rejects_lookup_and_insert(self):
        cache = DFlashPrefixCache(max_entries=4)
        manager = _manager(cache)
        key = _make_key()
        prompt = [71, 72, 73]
        snap = build_snapshot(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=mx.zeros((1, len(prompt), 6), dtype=mx.float32),
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            key=key,
        )
        manager.shutdown()

        with pytest.raises(RuntimeCacheManagerClosed):
            manager.lookup(prompt, key)
        with pytest.raises(RuntimeCacheManagerClosed):
            manager.maybe_insert_snapshot(
                snap,
                key=key,
                kind="prefill",
                require_logits=True,
            )

    def test_singleton_rebuilds_when_identity_changes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        context = _runtime_context(prefix_cache=True)

        first = cache_manager_mod.get_runtime_cache_manager(context, cache_identity="model-a")
        second = cache_manager_mod.get_runtime_cache_manager(context, cache_identity="model-b")

        assert first is not None
        assert second is not None
        assert first is not second

    def test_prefix_flow_lookup_records_hit(self, monkeypatch):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key(
            target_model_id="target/x",
            draft_model_id="draft/y",
            capture_layer_ids=(3, 7),
        )
        prompt = [11, 12, 13, 14]
        snap, _ = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=4)

        monkeypatch.setattr(
            flow_mod,
            "get_runtime_cache_manager",
            lambda _runtime_context, *, cache_identity=None: _manager(cache),
        )
        monkeypatch.setattr(flow_mod, "build_prefix_key", lambda *_args: key)

        flow = flow_mod.PrefixCacheFlow.for_request(
            model_provider=_FakeLoadedProvider(),
            draft_model=_FakeDraft(),
            tokenizer=_FakeTokenizer(),
            prompt=prompt,
            runtime_context=_runtime_context(prefix_cache=True),
        )

        assert flow.cache_active
        assert flow.hit_tokens == len(prompt)
        assert flow.snapshot is snap
        assert flow.stable_prefix_len == len(prompt)

    def test_prefix_flow_requires_full_coverage_above_projected_ctx(self, monkeypatch):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        key = _make_key()
        seen: dict = {}

        class RecordingManager:
            def lookup(self, tokens, lookup_key, **kwargs):
                seen.update(kwargs)
                return PrefixCacheLookupResult(
                    matched_tokens=0,
                    snapshot=None,
                    elapsed_ms=0.0,
                )

            def log_stats(self, label=""):
                return None

        monkeypatch.setattr(
            flow_mod,
            "get_runtime_cache_manager",
            lambda _runtime_context, *, cache_identity=None: RecordingManager(),
        )
        monkeypatch.setattr(flow_mod, "build_prefix_key", lambda *_args: key)

        provider = _FakeLoadedProvider()
        provider.model = object()
        provider.target_ops = SimpleNamespace(
            capabilities_for=lambda _model: SimpleNamespace(
                supports_full_context_draft_layers=True
            )
        )
        full_attention_draft = SimpleNamespace(
            target_layer_ids=[3, 7],
            args=SimpleNamespace(layer_types=("full_attention",), sliding_window=0),
        )

        flow_mod.PrefixCacheFlow.for_request(
            model_provider=provider,
            draft_model=full_attention_draft,
            tokenizer=_FakeTokenizer(),
            prompt=[11, 12, 13, 14],
            max_new_tokens=32768,
            runtime_context=_runtime_context(prefix_cache=True),
        )

        assert seen.get("require_full_coverage") is True

    def test_prefix_flow_skips_coverage_filter_below_projected_ctx(self, monkeypatch):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        key = _make_key()
        seen: dict = {}

        class RecordingManager:
            def lookup(self, tokens, lookup_key, **kwargs):
                seen.update(kwargs)
                return PrefixCacheLookupResult(
                    matched_tokens=0,
                    snapshot=None,
                    elapsed_ms=0.0,
                )

            def log_stats(self, label=""):
                return None

        monkeypatch.setattr(
            flow_mod,
            "get_runtime_cache_manager",
            lambda _runtime_context, *, cache_identity=None: RecordingManager(),
        )
        monkeypatch.setattr(flow_mod, "build_prefix_key", lambda *_args: key)

        provider = _FakeLoadedProvider()
        provider.model = object()
        provider.target_ops = SimpleNamespace(
            capabilities_for=lambda _model: SimpleNamespace(
                supports_full_context_draft_layers=True
            )
        )
        full_attention_draft = SimpleNamespace(
            target_layer_ids=[3, 7],
            args=SimpleNamespace(layer_types=("full_attention",), sliding_window=0),
        )

        flow_mod.PrefixCacheFlow.for_request(
            model_provider=provider,
            draft_model=full_attention_draft,
            tokenizer=_FakeTokenizer(),
            prompt=[11, 12, 13, 14],
            max_new_tokens=256,
            runtime_context=_runtime_context(prefix_cache=True),
        )

        assert "require_full_coverage" not in seen

    def test_prefix_flow_keeps_generation_snapshots_for_tool_chat(self, monkeypatch):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key(
            target_model_id="target/x",
            draft_model_id="draft/y",
            capture_layer_ids=(3, 7),
        )
        prompt = [11, 12, 13, 14]

        monkeypatch.setattr(
            flow_mod,
            "get_runtime_cache_manager",
            lambda _runtime_context, *, cache_identity=None: _manager(cache),
        )
        monkeypatch.setattr(flow_mod, "build_prefix_key", lambda *_args: key)

        flow = flow_mod.PrefixCacheFlow.for_request(
            model_provider=_FakeLoadedProvider(),
            draft_model=_FakeDraft(),
            tokenizer=_FakeTokenizer(),
            prompt=prompt,
            request=SimpleNamespace(
                request_type="chat",
                messages=[{"role": "tool", "content": "result"}],
                tools=[{"type": "function"}],
            ),
            runtime_context=_runtime_context(prefix_cache=True),
        )

        assert flow.cache_active
        assert flow.publish_generation_snapshot is True

    def test_prefix_flow_keeps_lookup_hit_when_log_stats_manager_retires(self, monkeypatch):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        prompt = [11, 12, 13, 14]

        class LogStatsRetiredManager:
            def lookup(self, tokens, key):
                snap, _ = _simulate_serve_insert(
                    DFlashPrefixCache(max_entries=4),
                    key,
                    list(tokens),
                    n_cached_tokens=len(tokens),
                )
                return PrefixCacheLookupResult(
                    matched_tokens=len(tokens),
                    snapshot=snap,
                    elapsed_ms=0.25,
                )

            def log_stats(self, label=""):
                raise RuntimeCacheManagerClosed("runtime cache manager is shut down")

        monkeypatch.setattr(
            flow_mod,
            "get_runtime_cache_manager",
            lambda _runtime_context, *, cache_identity=None: LogStatsRetiredManager(),
        )

        flow = flow_mod.PrefixCacheFlow.for_request(
            model_provider=_FakeLoadedProvider(),
            draft_model=_FakeDraft(),
            tokenizer=_FakeTokenizer(),
            prompt=prompt,
            runtime_context=_runtime_context(prefix_cache=True),
        )

        assert not flow.cache_active
        assert flow.hit_tokens == len(prompt)
        assert flow.snapshot is not None
        assert flow.lookup_ms == 0.25
        assert flow.snapshot_service is None

    def test_prefix_flow_treats_retired_lookup_manager_as_inactive(self, monkeypatch):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        manager = _manager(max_entries=4)
        manager.shutdown()
        monkeypatch.setattr(
            flow_mod,
            "get_runtime_cache_manager",
            lambda _runtime_context, *, cache_identity=None: manager,
        )

        flow = flow_mod.PrefixCacheFlow.for_request(
            model_provider=_FakeLoadedProvider(),
            draft_model=_FakeDraft(),
            tokenizer=_FakeTokenizer(),
            prompt=[1, 2, 3],
            runtime_context=_runtime_context(prefix_cache=True),
        )

        assert not flow.cache_active
        assert flow.key is None
        assert flow.snapshot_service is None

    def test_snapshot_service_inserts_built_prefill_snapshot(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [21, 22, 23]
        target_hidden = mx.arange(1 * len(prompt) * 6, dtype=mx.float32).reshape(
            1, len(prompt), 6
        )
        service = _snapshot_service(cache, key)
        publication = service.publish(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            kind="prefill",
            require_logits=True,
            snapshot_boundary=len(prompt),
            allow_full_attention_context=False,
        )

        matched, found = cache.lookup(prompt, key)
        assert publication is not None
        assert publication.admitted is True
        assert publication.prefix_len == len(prompt)
        assert matched == len(prompt)
        assert found is not None
        assert found.token_ids == tuple(prompt)
        assert service.insert_ms >= 0.0

    def test_snapshot_service_reports_skipped_snapshot_as_not_admitted(self):
        cache = DFlashPrefixCache(max_entries=4, max_snapshot_tokens=2)
        key = _make_key()
        prompt = [21, 22, 23]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        service = _snapshot_service(cache, key)
        publication = service.publish(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            kind="prefill",
            require_logits=True,
            snapshot_boundary=len(prompt),
            allow_full_attention_context=False,
        )

        matched, found = cache.lookup(prompt, key)
        assert publication is not None
        assert publication.admitted is False
        assert publication.prefix_len == len(prompt)
        assert publication.insert_ms >= 0.0
        assert service.insert_ms >= 0.0
        assert cache.stats()["skipped_too_long"] == 1
        assert matched == 0
        assert found is None

    def test_snapshot_service_reports_l2_only_insert_as_admitted(self):
        class ImmediateL2:
            def __init__(self):
                self.snapshots = []

            def insert_async(self, snapshot):
                self.snapshots.append(snapshot)
                return True

            def lookup(self, tokens, key, *, min_token_len=0, require_full_coverage=False):
                req = tuple(tokens)
                for snapshot in reversed(self.snapshots):
                    if snapshot.key == key and req[: len(snapshot.token_ids)] == snapshot.token_ids:
                        return snapshot
                return None

            def stats(self):
                return {"writes": len(self.snapshots)}

            def clear(self):
                self.snapshots.clear()

        l2 = ImmediateL2()
        cache = _store(DFlashPrefixCache(max_entries=0), l2=l2)
        key = _make_key()
        prompt = [21, 22, 23]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        service = _snapshot_service(cache, key)
        publication = service.publish(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            kind="prefill",
            require_logits=True,
            snapshot_boundary=len(prompt),
            allow_full_attention_context=False,
        )

        matched, found = cache.lookup(prompt, key)
        assert publication is not None
        assert publication.admitted is True
        assert cache.stats()["current_entries"] == 0
        assert l2.snapshots
        assert matched == len(prompt)
        assert found is not None
        assert found.token_ids == tuple(prompt)

    def test_snapshot_service_reports_l2_rejected_insert_as_not_admitted(self, tmp_path):
        writable_l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        read_only_l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            assert writable_l2.writable
            assert not read_only_l2.writable
            cache = _store(DFlashPrefixCache(max_entries=0), l2=read_only_l2)
            key = _make_key()
            prompt = [21, 22, 23]
            target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
            service = _snapshot_service(cache, key)
            publication = service.publish(
                token_ids=prompt,
                target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
                target_hidden=target_hidden,
                last_logits=mx.zeros((1, 32), dtype=mx.float32),
                kind="prefill",
                require_logits=True,
                snapshot_boundary=len(prompt),
                allow_full_attention_context=False,
            )

            matched, found = cache.lookup(prompt, key)
            assert publication is not None
            assert publication.admitted is False
            assert matched == 0
            assert found is None
        finally:
            read_only_l2.shutdown()
            writable_l2.shutdown()

    def test_snapshot_service_inserts_generation_snapshot_as_prefix_only(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [41, 42, 43]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        service = _snapshot_service(cache, key)
        publication = service.publish(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=None,
            kind="generation",
            require_logits=False,
            snapshot_boundary=len(prompt),
            allow_full_attention_context=False,
        )

        exact_len, exact = cache.lookup(prompt, key)
        assert exact_len == 0
        assert exact is None

        prefix_len, prefix = cache.lookup(prompt + [99], key)
        assert publication is not None
        assert publication.admitted is True
        assert prefix_len == len(prompt)
        assert prefix is not None
        assert prefix.kind == "generation"
        assert prefix.token_ids == tuple(prompt)

    def test_snapshot_service_raises_on_invalid_snapshot_contract(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        wrong_key = _make_key(target_model_id="other/target")
        prompt = [31, 32, 33]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        wrong_snap = build_snapshot(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            key=wrong_key,
        )

        class WrongTypeBuilder:
            def build(self, **_kwargs):
                return {"snapshot": "not-a-snapshot"}

        class WrongKeyBuilder:
            def build(self, **_kwargs):
                return wrong_snap

        wrong_type_builder = WrongTypeBuilder()
        wrong_type_builder.key = key
        wrong_key_builder = WrongKeyBuilder()
        wrong_key_builder.key = key

        with pytest.raises(TypeError, match="expected DFlashPrefixSnapshot"):
            _snapshot_service(cache, key, builder=wrong_type_builder).publish(
                token_ids=prompt,
                target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
                target_hidden=target_hidden,
                last_logits=mx.zeros((1, 32), dtype=mx.float32),
                kind="prefill",
                require_logits=True,
                snapshot_boundary=len(prompt),
                allow_full_attention_context=False,
            )
        with pytest.raises(ValueError, match="snapshot key"):
            _snapshot_service(cache, key, builder=wrong_key_builder).publish(
                token_ids=prompt,
                target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
                target_hidden=target_hidden,
                last_logits=mx.zeros((1, 32), dtype=mx.float32),
                kind="prefill",
                require_logits=True,
                snapshot_boundary=len(prompt),
                allow_full_attention_context=False,
            )
        with pytest.raises(ValueError, match="requires last_logits"):
            _snapshot_service(cache, key).publish(
                token_ids=prompt,
                target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
                target_hidden=target_hidden,
                last_logits=None,
                kind="prefill",
                require_logits=True,
                snapshot_boundary=len(prompt),
                allow_full_attention_context=False,
            )

        matched, found = cache.lookup(prompt, key)
        assert matched == 0
        assert found is None

    def test_snapshot_service_ignores_retired_manager_on_insert(self, capsys):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [61, 62, 63]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        manager = _manager(cache)
        manager.shutdown()
        service = SnapshotService(
            cache_manager=manager,
            builder=PrefixSnapshotBuilder(key=key),
        )

        publication = service.publish(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            kind="prefill",
            require_logits=True,
            snapshot_boundary=len(prompt),
            allow_full_attention_context=False,
        )

        assert publication is not None
        assert publication.admitted is False
        assert service.active is False
        assert service.insert_ms == 0.0
        matched, found = cache.lookup(prompt, key)
        assert matched == 0
        assert found is None
        assert "prefix cache insert failed" not in capsys.readouterr().err

    def test_snapshot_service_raises_and_logs_cache_insert_failure(self, capsys):
        class BrokenInsertCache(DFlashPrefixCache):
            def insert_with_evictions(self, snapshot, *, skip_too_long=True):
                raise RuntimeError("insert failed")

        cache = BrokenInsertCache(max_entries=4)
        key = _make_key()
        prompt = [51, 52, 53]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        service = _snapshot_service(cache, key)

        with pytest.raises(RuntimeError, match="insert failed"):
            service.publish(
                token_ids=prompt,
                target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
                target_hidden=target_hidden,
                last_logits=mx.zeros((1, 32), dtype=mx.float32),
                kind="prefill",
                require_logits=True,
                snapshot_boundary=len(prompt),
                allow_full_attention_context=False,
            )

        assert "prefix cache insert failed: insert failed" in capsys.readouterr().err

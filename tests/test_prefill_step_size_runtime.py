# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import time
from types import SimpleNamespace

import mlx.core as mx
import pytest

from dflash_mlx.cache.codecs import PrefixSnapshotBuilder
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.manager import RuntimeCacheManager
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.cache.snapshot_service import SnapshotService
from dflash_mlx.cache.store import PrefixSnapshotStore
from dflash_mlx.engine.events import (
    CycleCompleteEvent,
    MemoryWaterfallEvent,
    PrefillCompleteEvent,
    PrefillProgressEvent,
    SnapshotPublishedEvent,
    SummaryEvent,
    TokenEvent,
)
from dflash_mlx.engine import spec_epoch
from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
from dflash_mlx.runtime.config import runtime_config_from_defaults
from dflash_mlx.runtime.context import build_runtime_context


class _FakeTargetOps:
    def __init__(self, *, restore_delay_s: float = 0.0, restore_ns: int = 0) -> None:
        self.forward_lengths: list[int] = []
        self.logits_last_only_flags: list[bool] = []
        self.cleanup_calls = 0
        self.restore_delay_s = float(restore_delay_s)
        self.restore_ns = int(restore_ns)

    def capabilities_for(self, _target_model):
        return SimpleNamespace(supports_prefix_snapshot=True, supports_tree_verify=False)

    def supports_tree_cache(self, _target_cache):
        return True

    def make_cache(self, *_args, **_kwargs):
        return []

    def forward_with_hidden_capture(
        self,
        _target_model,
        *,
        input_ids,
        cache,
        capture_layer_ids,
        logits_last_only=False,
    ):
        del cache, capture_layer_ids
        batch, seq_len = input_ids.shape
        self.forward_lengths.append(int(seq_len))
        self.logits_last_only_flags.append(bool(logits_last_only))
        logits_len = 1 if logits_last_only else seq_len
        logits = mx.zeros((batch, logits_len, 8), dtype=mx.float32)
        hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
        return logits, hidden

    def extract_context_feature(self, hidden_states, target_layer_id_list):
        layer_id = int(target_layer_id_list[0]) + 1
        return hidden_states[layer_id]

    def cleanup_generation_caches(self, *_args) -> None:
        self.cleanup_calls += 1

    def arm_rollback(self, *_args, **_kwargs) -> None:
        return None

    def verify_block(
        self,
        *,
        target_model,
        verify_ids,
        target_cache,
        capture_layer_ids,
    ):
        del target_model, target_cache, capture_layer_ids
        batch, seq_len = verify_ids.shape
        logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
        hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
        return logits, hidden

    def restore_after_acceptance(self, *_args, **_kwargs) -> int:
        if self.restore_delay_s > 0:
            time.sleep(self.restore_delay_s)
        return self.restore_ns


class _FakeDraftBackend:
    def make_cache(self, **_kwargs):
        return []

    def draft_greedy(self, **_kwargs):
        block_len = int(_kwargs["block_len"])
        return mx.zeros((max(0, block_len - 1),), dtype=mx.uint32)

    def advance_context(self, **_kwargs) -> None:
        return None


class _RecordingDraftBackend(_FakeDraftBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []
        self.advance_context_lengths: list[int] = []
        self.ddtree_topk_calls: list[tuple[int, int]] = []
        self.ddtree_branch_batch_calls: list[tuple[int, int]] = []

    def draft_greedy(self, **kwargs):
        block_len = int(kwargs["block_len"])
        self.calls.append((block_len, bool(kwargs["async_launch"])))
        return mx.zeros((max(0, block_len - 1),), dtype=mx.uint32)

    def advance_context(self, **kwargs) -> None:
        self.advance_context_lengths.append(int(kwargs["draft_context"].shape[1]))

    def draft_with_topk(self, **kwargs):
        from dflash_mlx.engine.ddtree import draft_block_with_topk

        self.ddtree_topk_calls.append(
            (int(kwargs["block_len"]), int(kwargs["top_width"]))
        )
        drafted, top_ids, top_values, _draft_us = draft_block_with_topk(**kwargs)
        return drafted, top_ids, top_values

    def draft_branch_blocks_batch(self, **kwargs):
        from dflash_mlx.engine.ddtree import draft_branch_blocks_batch

        self.ddtree_branch_batch_calls.append(
            (len(kwargs["branch_prefixes"]), int(kwargs["block_len"]))
        )
        candidate_ids, _draft_us = draft_branch_blocks_batch(**kwargs)
        return candidate_ids


class _RecordingL2:
    def __init__(self) -> None:
        self.snapshots: list[DFlashPrefixSnapshot] = []

    def lookup(self, req_tokens, key, *, min_token_len: int = 0):
        req_tuple = tuple(int(token) for token in req_tokens)
        matches = [
            snapshot
            for snapshot in self.snapshots
            if snapshot.key == key
            and len(snapshot.token_ids) > int(min_token_len)
            and len(snapshot.token_ids) <= len(req_tuple)
            and req_tuple[: len(snapshot.token_ids)] == snapshot.token_ids
        ]
        if not matches:
            return None
        return max(matches, key=lambda snapshot: len(snapshot.token_ids))

    def insert_async(self, snapshot):
        self.snapshots.append(snapshot)
        return True

    def stats(self):
        return {}

    def clear(self) -> None:
        self.snapshots.clear()

    def shutdown(self, *, wait: bool = True) -> None:
        del wait


def _runtime_context(
    diagnostics_config: DiagnosticsConfig | None = None,
    verify_mode: str | None = None,
    verify_len_cap: int | None = None,
):
    return build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
            verify_mode=verify_mode,
            verify_len_cap=0 if verify_len_cap is None else verify_len_cap,
        ),
        diagnostics_config=diagnostics_config,
    )


def _draft_model(*, block_size: int = 4):
    return SimpleNamespace(
        target_layer_ids=[0],
        block_size=block_size,
        mask_token_id=0,
        project_target_hidden=lambda value: value,
    )


class _TokenizingTokenizer:
    def __init__(self):
        self.encode_calls: list[str] = []
        self.template_calls: list[tuple[list[dict[str, str]], bool, bool]] = []

    def encode(self, prompt):
        self.encode_calls.append(prompt)
        return [11, 12]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.template_calls.append((messages, tokenize, add_generation_prompt))
        return [21, 22, 23]


def _prefix_key() -> DFlashPrefixKey:
    return DFlashPrefixKey(
        target_model_id="target",
        draft_model_id="draft",
        capture_layer_ids=(0,),
        draft_sink_size=64,
        draft_window_size=1024,
        template_hash="a" * 64,
        prompt_policy_hash="b" * 64,
        target_fa_window=0,
    )


def _snapshot_builder(draft_model):
    return PrefixSnapshotBuilder(
        key=_prefix_key(),
        draft_model=draft_model,
        draft_sink_size=64,
        draft_window_size=1024,
    )


def _snapshot_service(draft_model, *, cache=None, builder=None):
    cache = cache if cache is not None else DFlashPrefixCache(max_entries=4)
    builder = builder if builder is not None else _snapshot_builder(draft_model)
    return SnapshotService(
        cache_manager=RuntimeCacheManager(PrefixSnapshotStore(l1=cache)),
        builder=builder,
    )


def _frontier_snapshot_setup(
    *,
    with_l2: bool,
    frontier_stride: int = 4,
    prefill_step_size: int = 4,
):
    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=prefill_step_size,
            prefix_cache=True,
            prefix_cache_l2=True,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
        project_target_hidden=lambda value: value,
    )
    key = DFlashPrefixKey(
        target_model_id="target",
        draft_model_id="draft",
        capture_layer_ids=(0,),
        draft_sink_size=64,
        draft_window_size=1024,
        template_hash="a" * 64,
        prompt_policy_hash="b" * 64,
        target_fa_window=0,
    )
    cache = DFlashPrefixCache(max_entries=8, frontier_stride=frontier_stride)
    l2 = _RecordingL2() if with_l2 else None
    manager = RuntimeCacheManager(PrefixSnapshotStore(l1=cache, l2=l2))
    return (
        context,
        draft_model,
        key,
        cache,
        l2,
        manager,
        SnapshotService(
            cache_manager=manager,
            builder=PrefixSnapshotBuilder(
                key=key,
                draft_model=draft_model,
                draft_sink_size=64,
                draft_window_size=1024,
            ),
        ),
    )


def test_session_request_derives_token_views():
    token_ids = [1, 2]

    request = spec_epoch._SessionRequest.from_tokens(
        prompt_tokens=token_ids,
        max_new_tokens=3,
        block_tokens=None,
        stop_token_ids=[7],
        suppress_token_ids=None,
        prefix_snapshot=None,
        snapshot_service=None,
        stable_prefix_len=None,
        prefix_cache_active=False,
    )
    token_ids.append(99)

    assert request.prompt_tokens == (1, 2)
    assert request.prompt_len == 2
    assert tuple(int(x) for x in request.prompt_array.reshape(-1).tolist()) == (1, 2)
    assert request.stop_token_array is not None
    assert tuple(int(x) for x in request.stop_token_array.tolist()) == (7,)


def test_dflash_stream_tokenizes_plain_prompt_without_override():
    target_ops = _FakeTargetOps()
    tokenizer = _TokenizingTokenizer()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=tokenizer,
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="plain",
            max_new_tokens=0,
            use_chat_template=False,
            runtime_context=_runtime_context(),
        )
    )

    prefill_event = next(
        event for event in events if isinstance(event, PrefillCompleteEvent)
    )
    assert tokenizer.encode_calls == ["plain"]
    assert tokenizer.template_calls == []
    assert prefill_event.prompt_token_count == 2


def test_dflash_stream_tokenizes_chat_template_without_override():
    target_ops = _FakeTargetOps()
    tokenizer = _TokenizingTokenizer()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=tokenizer,
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="chat",
            max_new_tokens=0,
            use_chat_template=True,
            runtime_context=_runtime_context(),
        )
    )

    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    assert tokenizer.encode_calls == []
    assert tokenizer.template_calls == [
        ([{"role": "user", "content": "chat"}], True, True)
    ]
    assert prefill_event.prompt_token_count == 3


def test_dflash_max_ctx_fallback_skips_session_request_materialization(monkeypatch):
    target_ops = _FakeTargetOps()

    class FakeTarget:
        def __call__(self, input_ids, cache):
            del cache
            batch, seq_len = input_ids.shape
            return mx.zeros((batch, seq_len, 8), dtype=mx.float32)

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
            dflash_max_ctx=2,
        )
    )
    calls = []
    original_from_tokens = spec_epoch._SessionRequest.from_tokens

    def tracked_from_tokens(*args, **kwargs):
        calls.append((args, kwargs))
        return original_from_tokens(*args, **kwargs)

    monkeypatch.setattr(spec_epoch._SessionRequest, "from_tokens", tracked_from_tokens)

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=FakeTarget(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=1,
            prompt_tokens_override=[1, 2],
            runtime_context=context,
        )
    )

    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    summary_event = next(event for event in events if isinstance(event, SummaryEvent))
    assert calls == []
    assert prefill_event.fallback_ar is True
    assert summary_event.fallback_ar is True
    assert "projected_ctx=3" in summary_event.fallback_reason


def test_dflash_max_ctx_fallback_accounts_for_requested_generation(monkeypatch):
    target_ops = _FakeTargetOps()

    class FakeTarget:
        def __call__(self, input_ids, cache):
            del cache
            batch, seq_len = input_ids.shape
            return mx.zeros((batch, seq_len, 8), dtype=mx.float32)

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
            dflash_max_ctx=4,
        )
    )
    calls = []
    original_from_tokens = spec_epoch._SessionRequest.from_tokens

    def tracked_from_tokens(*args, **kwargs):
        calls.append((args, kwargs))
        return original_from_tokens(*args, **kwargs)

    monkeypatch.setattr(spec_epoch._SessionRequest, "from_tokens", tracked_from_tokens)

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=FakeTarget(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=2,
            prompt_tokens_override=[1, 2],
            runtime_context=context,
        )
    )

    summary_event = next(event for event in events if isinstance(event, SummaryEvent))
    assert calls == []
    assert summary_event.fallback_ar is True
    assert "projected_ctx=4" in summary_event.fallback_reason


def _prefix_snapshot(
    *,
    token_ids=(1, 2),
    fa_states=(),
    gdn_states=(),
    last_logits=None,
    kind="prefill",
):
    return DFlashPrefixSnapshot(
        token_ids=tuple(token_ids),
        fa_states=fa_states,
        gdn_states=gdn_states,
        target_hidden_chunks=(mx.zeros((1, len(token_ids), 2), dtype=mx.float32),),
        target_hidden_chunk_spans=((0, len(token_ids)),),
        target_hidden_total_len=len(token_ids),
        last_logits=last_logits,
        key=_prefix_key(),
        kind=kind,
    )


def test_yield_pause_tracker_records_enabled_pause(monkeypatch):
    ticks = iter([10, 17, 100, 140])
    monkeypatch.setattr(spec_epoch.time, "perf_counter_ns", lambda: next(ticks))

    tracker = spec_epoch._YieldPauseTracker(enabled=True)

    mark = tracker.mark()
    tracker.done(mark)
    assert tracker.pause_ns == 7

    mark = tracker.mark()
    tracker.done(mark)
    assert tracker.pause_ns == 47


def test_yield_pause_tracker_disabled_noops(monkeypatch):
    calls = []
    monkeypatch.setattr(
        spec_epoch.time,
        "perf_counter_ns",
        lambda: calls.append("called") or 1,
    )

    tracker = spec_epoch._YieldPauseTracker(enabled=False)

    assert tracker.mark() == 0
    tracker.done(10)
    assert tracker.pause_ns == 0
    assert calls == []


def test_runtime_prefill_chunks_use_configured_step_size():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
        project_target_hidden=lambda value: value,
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(10)),
            runtime_context=context,
        )
    )

    assert target_ops.forward_lengths == [4, 4, 1, 1]
    assert target_ops.logits_last_only_flags == [True, True, True, True]
    assert [
        event.tokens_processed
        for event in events
        if isinstance(event, PrefillProgressEvent)
    ] == [4, 8, 9, 10]
    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    assert prefill_event.logical_ctx_tokens == 10
    assert prefill_event.physical_prefill_tokens == 10
    assert prefill_event.prefill_tokens_restored == 0
    assert prefill_event.prefill_tokens_computed == 10
    assert isinstance(events[-1], SummaryEvent)


def test_clear_cache_boundaries_false_skips_mlx_clear_cache(monkeypatch):
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    calls = []
    monkeypatch.setattr(
        spec_epoch.mx,
        "synchronize",
        lambda: calls.append("sync"),
        raising=False,
    )
    monkeypatch.setattr(
        spec_epoch.mx,
        "clear_cache",
        lambda: calls.append("clear"),
        raising=False,
    )

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            clear_cache_boundaries=False,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(6)),
            runtime_context=context,
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))
    assert summary.clear_cache_boundaries is False
    assert calls == []


def test_clear_cache_boundaries_true_clears_safe_boundaries(monkeypatch):
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    calls = []
    monkeypatch.setattr(
        spec_epoch.mx,
        "synchronize",
        lambda: calls.append("sync"),
        raising=False,
    )
    monkeypatch.setattr(
        spec_epoch.mx,
        "clear_cache",
        lambda: calls.append("clear"),
        raising=False,
    )

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            clear_cache_boundaries=True,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(6)),
            runtime_context=context,
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))
    assert summary.clear_cache_boundaries is True
    assert calls == [
        "sync",
        "clear",
        "sync",
        "clear",
        "sync",
        "clear",
        "sync",
        "clear",
    ]


def test_clear_cache_boundaries_true_clears_during_long_decode(monkeypatch):
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    calls = []
    monkeypatch.setattr(
        spec_epoch.mx,
        "synchronize",
        lambda: calls.append("sync"),
        raising=False,
    )
    monkeypatch.setattr(
        spec_epoch.mx,
        "clear_cache",
        lambda: calls.append("clear"),
        raising=False,
    )

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            clear_cache_boundaries=True,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=1030,
            prompt_tokens_override=list(range(6)),
            runtime_context=context,
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))
    assert summary.generation_tokens == 1030
    assert calls == ["sync", "clear"] * 5


def test_runtime_prefill_accounting_reports_warm_restore():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
        project_target_hidden=lambda value: value,
    )
    prefix_snapshot = DFlashPrefixSnapshot(
        token_ids=tuple(range(6)),
        fa_states=(),
        gdn_states=(),
        target_hidden_chunks=(mx.zeros((1, 6, 2), dtype=mx.float32),),
        target_hidden_chunk_spans=((0, 6),),
        target_hidden_total_len=6,
        last_logits=mx.zeros((1, 8), dtype=mx.float32),
        key=DFlashPrefixKey(
            target_model_id="target",
            draft_model_id="draft",
            capture_layer_ids=(0,),
            draft_sink_size=64,
            draft_window_size=1024,
            template_hash="a" * 64,
            prompt_policy_hash="b" * 64,
            target_fa_window=0,
        ),
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(10)),
            prefix_snapshot=prefix_snapshot,
            stable_prefix_len=6,
            runtime_context=context,
        )
    )

    assert target_ops.forward_lengths == [4]
    assert target_ops.logits_last_only_flags == [True]
    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    assert prefill_event.logical_ctx_tokens == 10
    assert prefill_event.physical_prefill_tokens == 4
    assert prefill_event.prefill_tokens_restored == 6
    assert prefill_event.prefill_tokens_computed == 4


def test_generation_snapshot_restore_computes_only_prompt_tail():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    prefix_snapshot = _prefix_snapshot(
        token_ids=(1, 2, 0, 0, 0),
        kind="generation",
    )
    snapshot_service = _snapshot_service(draft_model)
    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=True,
            prefix_cache_l2=False,
        )
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=[1, 2, 0, 0, 0, 88, 89],
            prefix_snapshot=prefix_snapshot,
            snapshot_service=snapshot_service,
            stable_prefix_len=7,
            prefix_cache_active=True,
            runtime_context=context,
        )
    )

    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    assert target_ops.forward_lengths == [1, 1]
    assert prefill_event.prefill_tokens_restored == 5
    assert prefill_event.prefill_tokens_computed == 2


def test_prefix_snapshot_released_after_prefill_seam():
    import gc
    import weakref

    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    snapshot = _prefix_snapshot(
        token_ids=(1, 2, 0, 0, 0),
        kind="generation",
    )
    snapshot_ref = weakref.ref(snapshot)
    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=True,
            prefix_cache_l2=False,
        )
    )

    events = spec_epoch.stream_dflash_generate_impl(
        target_model=object(),
        target_ops=target_ops,
        tokenizer=object(),
        draft_model=draft_model,
        draft_backend=draft_backend,
        prompt="unused",
        max_new_tokens=8,
        prompt_tokens_override=[1, 2, 0, 0, 0, 88, 89],
        prefix_snapshot=snapshot,
        snapshot_service=_snapshot_service(draft_model),
        stable_prefix_len=7,
        prefix_cache_active=True,
        runtime_context=context,
    )
    del snapshot
    try:
        saw_prefill_complete = False
        for event in events:
            if isinstance(event, PrefillCompleteEvent):
                saw_prefill_complete = True
                break
        assert saw_prefill_complete
        gc.collect()
        assert snapshot_ref() is None, (
            "prefix snapshot must be released once prefill consumed it, "
            "not stay pinned by request/generator refs for the whole "
            "generation"
        )
    finally:
        events.close()


def test_warm_exact_hit_skips_prefill_republish():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    prefix_snapshot = _prefix_snapshot(
        token_ids=(1, 2),
        last_logits=mx.zeros((1, 8), dtype=mx.float32),
    )
    cache = DFlashPrefixCache(max_entries=4)
    snapshot_service = _snapshot_service(draft_model, cache=cache)
    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=True,
            prefix_cache_l2=False,
        )
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=[1, 2],
            prefix_snapshot=prefix_snapshot,
            snapshot_service=snapshot_service,
            stable_prefix_len=2,
            prefix_cache_active=True,
            runtime_context=context,
        )
    )

    assert not any(
        isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
        for event in events
    )


def test_prefix_snapshot_hydration_failure_raises():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    prefix_snapshot = _prefix_snapshot(
        fa_states=(
            (
                mx.zeros((1, 1, 2, 2), dtype=mx.float32),
                mx.zeros((1, 1, 2, 2), dtype=mx.float32),
                2,
            ),
        ),
        gdn_states=(None,),
        last_logits=mx.zeros((1, 8), dtype=mx.float32),
    )

    with pytest.raises(RuntimeError, match="prefix snapshot hydrate failed for 2 tokens"):
        list(
            spec_epoch.stream_dflash_generate_impl(
                target_model=object(),
                target_ops=target_ops,
                tokenizer=object(),
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="unused",
                max_new_tokens=0,
                prompt_tokens_override=[1, 2, 3],
                prefix_snapshot=prefix_snapshot,
                stable_prefix_len=2,
                runtime_context=_runtime_context(),
            )
        )


def test_prefill_snapshot_event_carries_snapshot_not_live_arrays():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
        project_target_hidden=lambda value: value,
    )
    key = DFlashPrefixKey(
        target_model_id="target",
        draft_model_id="draft",
        capture_layer_ids=(0,),
        draft_sink_size=64,
        draft_window_size=1024,
        template_hash="a" * 64,
        prompt_policy_hash="b" * 64,
        target_fa_window=0,
    )
    cache = DFlashPrefixCache(max_entries=4)
    snapshot_service = _snapshot_service(
        draft_model,
        cache=cache,
        builder=PrefixSnapshotBuilder(
            key=key,
            draft_model=draft_model,
            draft_sink_size=64,
            draft_window_size=1024,
        ),
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=[1, 2],
            snapshot_service=snapshot_service,
            runtime_context=context,
        )
    )

    snapshot_event = next(
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
    )
    matched, snapshot = cache.lookup([1, 2], key)
    assert snapshot_event.admitted is True
    assert snapshot_event.prefix_len == 2
    assert matched == 2
    assert snapshot is not None
    assert snapshot.key == key
    assert snapshot.token_ids == (1, 2)


def test_prefill_publishes_aligned_frontiers_when_l2_cache_is_enabled():
    context, draft_model, key, cache, l2, manager, snapshot_service = _frontier_snapshot_setup(
        with_l2=True
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_FakeTargetOps(),
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(1, 11)),
            snapshot_service=snapshot_service,
            runtime_context=context,
        )
    )

    prefill_snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
    ]
    assert [event.prefix_len for event in prefill_snapshots] == [4, 8, 10]

    matched, snapshot = cache.lookup([1, 2, 3, 4, 99], key)
    assert matched == 0
    assert snapshot is None

    result = manager.lookup([1, 2, 3, 4, 99], key)
    matched = result.matched_tokens
    snapshot = result.snapshot
    assert matched == 4
    assert snapshot is not None
    assert snapshot.token_ids == (1, 2, 3, 4)
    assert l2 is not None
    assert [snapshot.prefix_len for snapshot in l2.snapshots] == [4, 8, 10]


def test_prefill_frontiers_use_cache_stride_not_prefill_step_size():
    context, draft_model, key, cache, l2, manager, snapshot_service = _frontier_snapshot_setup(
        with_l2=True,
        frontier_stride=8,
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_FakeTargetOps(),
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(1, 11)),
            snapshot_service=snapshot_service,
            runtime_context=context,
        )
    )

    prefill_snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
    ]
    assert [event.prefix_len for event in prefill_snapshots] == [8, 10]

    result = manager.lookup([1, 2, 3, 4, 99], key)
    assert result.matched_tokens == 0
    assert result.snapshot is None

    result = manager.lookup([1, 2, 3, 4, 5, 6, 7, 8, 99], key)
    assert result.matched_tokens == 8
    assert result.snapshot is not None
    assert result.snapshot.token_ids == tuple(range(1, 9))
    assert l2 is not None
    assert [snapshot.prefix_len for snapshot in l2.snapshots] == [8, 10]


def test_prefill_frontier_stride_can_follow_non_divisor_chunk_size():
    context, draft_model, key, cache, l2, manager, snapshot_service = _frontier_snapshot_setup(
        with_l2=True,
        frontier_stride=10000,
        prefill_step_size=5000,
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_FakeTargetOps(),
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(1, 10003)),
            snapshot_service=snapshot_service,
            runtime_context=context,
        )
    )

    prefill_snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
    ]
    assert [event.prefix_len for event in prefill_snapshots] == [10000, 10002]

    result = manager.lookup([*range(1, 5001), 99], key)
    assert result.matched_tokens == 0
    assert result.snapshot is None

    result = manager.lookup([*range(1, 10001), 99], key)
    assert result.matched_tokens == 10000
    assert result.snapshot is not None
    assert result.snapshot.token_ids == tuple(range(1, 10001))
    assert l2 is not None
    assert [snapshot.prefix_len for snapshot in l2.snapshots] == [10000, 10002]


def test_prefill_skips_aligned_frontiers_without_l2_store():
    context, draft_model, _key, _cache, _l2, _manager, snapshot_service = (
        _frontier_snapshot_setup(with_l2=False)
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_FakeTargetOps(),
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(1, 11)),
            snapshot_service=snapshot_service,
            runtime_context=context,
        )
    )

    prefill_snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
    ]
    assert [event.prefix_len for event in prefill_snapshots] == [10]


def test_prefill_snapshot_service_build_failure_propagates():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()

    class BrokenBuilder:
        def build(self, **_kwargs):
            raise RuntimeError("snapshot builder broken")

    with pytest.raises(RuntimeError, match="snapshot builder broken"):
        list(
            spec_epoch.stream_dflash_generate_impl(
                target_model=object(),
                target_ops=target_ops,
                tokenizer=object(),
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="unused",
                max_new_tokens=0,
                prompt_tokens_override=[1, 2],
                snapshot_service=_snapshot_service(
                    draft_model,
                    builder=BrokenBuilder(),
                ),
                runtime_context=_runtime_context(),
            )
        )


def test_prefill_snapshot_service_requires_logits_for_warm_exact_hit():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    prefix_snapshot = _prefix_snapshot()

    with pytest.raises(ValueError, match="prefill snapshot requires last_logits"):
        list(
            spec_epoch.stream_dflash_generate_impl(
                target_model=object(),
                target_ops=target_ops,
                tokenizer=object(),
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="unused",
                max_new_tokens=0,
                prompt_tokens_override=[1, 2],
                prefix_snapshot=prefix_snapshot,
                stable_prefix_len=2,
                snapshot_service=_snapshot_service(draft_model),
                runtime_context=_runtime_context(),
            )
        )


def test_warm_exact_hit_without_snapshot_service_requires_logits_before_decode():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    prefix_snapshot = _prefix_snapshot()

    with pytest.raises(
        RuntimeError,
        match="prefill logits unavailable after prefix snapshot restore",
    ):
        list(
            spec_epoch.stream_dflash_generate_impl(
                target_model=object(),
                target_ops=target_ops,
                tokenizer=object(),
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="unused",
                max_new_tokens=0,
                prompt_tokens_override=[1, 2],
                prefix_snapshot=prefix_snapshot,
                stable_prefix_len=2,
                runtime_context=_runtime_context(),
            )
        )


def test_generation_snapshot_service_build_failure_propagates():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    delegate = _snapshot_builder(draft_model)

    class GenerationFailingBuilder:
        def build(self, **kwargs):
            if kwargs["kind"] == "generation":
                raise RuntimeError("generation snapshot broken")
            return delegate.build(**kwargs)

    failing_builder = GenerationFailingBuilder()
    failing_builder.key = delegate.key

    with pytest.raises(RuntimeError, match="generation snapshot broken"):
        list(
            spec_epoch.stream_dflash_generate_impl(
                target_model=object(),
                target_ops=target_ops,
                tokenizer=object(),
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="unused",
                max_new_tokens=1,
                prompt_tokens_override=[1, 2],
                snapshot_service=_snapshot_service(
                    draft_model,
                    builder=failing_builder,
                ),
                runtime_context=_runtime_context(),
            )
        )


def test_request_state_preserves_decode_summary_and_generation_snapshot():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    cache = DFlashPrefixCache(max_entries=4)
    key = _prefix_key()
    snapshot_service = _snapshot_service(
        draft_model,
        cache=cache,
        builder=PrefixSnapshotBuilder(
            key=key,
            draft_model=draft_model,
            draft_sink_size=64,
            draft_window_size=1024,
        ),
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=3,
            prompt_tokens_override=[1, 2],
            snapshot_service=snapshot_service,
            runtime_context=_runtime_context(),
        )
    )

    token_events = [event for event in events if isinstance(event, TokenEvent)]
    summary = next(event for event in events if isinstance(event, SummaryEvent))
    generation_snapshot = next(
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "generation"
    )

    assert [event.token_id for event in token_events] == [0, 0, 0]
    assert summary.generated_token_ids == (0, 0, 0)
    assert summary.generation_tokens == 3
    assert summary.cycles_completed == 1
    assert summary.accepted_from_draft == 2
    assert summary.acceptance_history == (2,)
    assert generation_snapshot.kind == "generation"
    assert generation_snapshot.prefix_len == 5
    matched, snapshot = cache.lookup([1, 2, 0, 0, 0, 99], key)
    assert matched == 5
    assert snapshot is not None
    assert snapshot.kind == "generation"
    assert snapshot.token_ids == (1, 2, 0, 0, 0)


def test_generation_snapshot_published_for_truncated_stable_prefix():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    cache = DFlashPrefixCache(max_entries=4)
    key = _prefix_key()
    snapshot_service = _snapshot_service(
        draft_model,
        cache=cache,
        builder=PrefixSnapshotBuilder(
            key=key,
            draft_model=draft_model,
            draft_sink_size=64,
            draft_window_size=1024,
        ),
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2, 3],
            snapshot_service=snapshot_service,
            stable_prefix_len=2,
            runtime_context=_runtime_context(
                diagnostics_config=DiagnosticsConfig(memory_waterfall=True),
            ),
        )
    )

    prefill_snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
    ]
    generation_snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "generation"
    ]
    matched, snapshot = cache.lookup([1, 2, 3, 0, 0, 0, 0, 0, 0, 99], key)

    # The generation snapshot subsumes every prefill prefix, so no
    # prefill-kind snapshot is published when generation will run.
    assert prefill_snapshots == []
    assert len(generation_snapshots) == 1
    assert generation_snapshots[0].kind == "generation"
    assert generation_snapshots[0].prefix_len == 9
    assert matched == 9
    assert snapshot is not None
    assert snapshot.kind == "generation"


def test_generation_snapshot_skipped_when_request_policy_disallows_it():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    cache = DFlashPrefixCache(max_entries=4)
    key = _prefix_key()
    snapshot_service = _snapshot_service(
        draft_model,
        cache=cache,
        builder=PrefixSnapshotBuilder(
            key=key,
            draft_model=draft_model,
            draft_sink_size=64,
            draft_window_size=1024,
        ),
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            snapshot_service=snapshot_service,
            publish_generation_snapshot=False,
            runtime_context=_runtime_context(
                diagnostics_config=DiagnosticsConfig(memory_waterfall=True),
            ),
        )
    )

    generation_snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "generation"
    ]
    matched, snapshot = cache.lookup([1, 2, 0, 0, 0, 99], key)

    assert generation_snapshots == []
    assert matched == 2
    assert snapshot is not None
    assert snapshot.kind == "prefill"
    _assert_no_generation_chunks_retained(events)


def test_generation_snapshot_hidden_not_retained_when_snapshot_service_retires():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    cache = DFlashPrefixCache(max_entries=4)
    manager = RuntimeCacheManager(PrefixSnapshotStore(l1=cache))
    manager.shutdown()
    snapshot_service = SnapshotService(
        cache_manager=manager,
        builder=PrefixSnapshotBuilder(
            key=_prefix_key(),
            draft_model=draft_model,
            draft_sink_size=64,
            draft_window_size=1024,
        ),
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            snapshot_service=snapshot_service,
            runtime_context=_runtime_context(
                diagnostics_config=DiagnosticsConfig(memory_waterfall=True),
            ),
        )
    )

    snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent)
    ]

    assert len(snapshots) == 1
    assert snapshots[0].kind == "prefill"
    assert snapshots[0].admitted is False
    assert snapshot_service.active is False
    _assert_no_generation_chunks_retained(events)


def _assert_no_generation_chunks_retained(events):
    after_rollback = [
        event.fields
        for event in events
        if isinstance(event, MemoryWaterfallEvent)
        and event.fields.get("memory_phase") == "after_rollback"
    ]
    assert after_rollback
    assert all(int(fields["gen_hidden_chunks_bytes"]) == 0 for fields in after_rollback)


def test_request_state_preserves_async_prefetched_draft_reuse():
    target_ops = _FakeTargetOps()
    draft_model = _draft_model()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls == [(4, True), (2, True)]
    assert summary.generated_token_ids == (0, 0, 0, 0, 0, 0)
    assert summary.generation_tokens == 6
    assert summary.cycles_completed == 2
    assert summary.accepted_from_draft == 4
    assert summary.acceptance_history == (3, 1)


def test_copyspec_full_block_copy_skips_draft_backend():
    target_ops = _FakeTargetOps()
    draft_model = _draft_model()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=4,
            prompt_tokens_override=[1, 2, 3, 4, 5, 0, 0, 0, 0, 1, 2, 3, 4, 5],
            runtime_context=_runtime_context(),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls == []
    assert draft_backend.advance_context_lengths == [14]
    assert summary.generated_token_ids == (0, 0, 0, 0)
    assert summary.accepted_from_draft == 3
    assert summary.acceptance_history == (3,)
    assert summary.copyspec_hits == 1
    assert summary.copyspec_tokens == 3


def test_ddtree_copyspec_full_block_copy_skips_ddtree_draft_backend():
    target_ops = _FakeTargetOps()
    draft_model = _draft_model()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=4,
            prompt_tokens_override=[1, 2, 3, 4, 5, 0, 0, 0, 0, 1, 2, 3, 4, 5],
            runtime_context=_runtime_context(verify_mode="ddtree"),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls == []
    assert draft_backend.ddtree_topk_calls == []
    assert draft_backend.ddtree_branch_batch_calls == []
    assert draft_backend.advance_context_lengths == [14]
    assert summary.generated_token_ids == (0, 0, 0, 0)
    assert summary.accepted_from_draft == 3
    assert summary.acceptance_history == (3,)
    assert summary.copyspec_hits == 1
    assert summary.copyspec_tokens == 3


def test_ddtree_copyspec_partial_acceptance_preserves_rollback_snapshot():
    class _RollbackEntry:
        def __init__(self) -> None:
            self.value = mx.zeros((1, 1), dtype=mx.float32)
            self._snapshot = None

    class _RollbackTargetOps(_FakeTargetOps):
        def __init__(self) -> None:
            super().__init__()
            self.restore_snapshot_shapes: list[tuple[int, ...] | None] = []
            self.restore_acceptance_lengths: list[int] = []

        def capabilities_for(self, _target_model):
            return SimpleNamespace(
                supports_prefix_snapshot=True,
                supports_tree_verify=True,
            )

        def make_cache(self, *_args, **_kwargs):
            return [_RollbackEntry()]

        def arm_rollback(self, cache_entries, *, prefix_len: int) -> None:
            del prefix_len
            for cache_entry in cache_entries:
                cache_entry._snapshot = cache_entry.value

        def verify_block(
            self,
            *,
            target_model,
            verify_ids,
            target_cache,
            capture_layer_ids,
        ):
            del target_model, capture_layer_ids
            batch, seq_len = verify_ids.shape
            target_cache[0].value = mx.ones((batch, seq_len), dtype=mx.float32)
            logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
            logits[:, 0, 0] = 1.0
            if seq_len > 1:
                logits[:, 1:, 7] = 1.0
            hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
            return logits, hidden

        def restore_after_acceptance(
            self,
            cache_entries,
            *,
            target_len: int,
            acceptance_length: int,
            drafted_tokens: int = 0,
        ) -> int:
            del target_len, drafted_tokens
            snapshot = cache_entries[0]._snapshot
            self.restore_snapshot_shapes.append(
                None if snapshot is None else tuple(int(dim) for dim in snapshot.shape)
            )
            self.restore_acceptance_lengths.append(int(acceptance_length))
            return 0

    target_ops = _RollbackTargetOps()
    draft_model = _draft_model()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=4,
            stop_token_ids=[0],
            prompt_tokens_override=[1, 2, 3, 4, 5, 0, 0, 0, 0, 1, 2, 3, 4, 5],
            runtime_context=_runtime_context(verify_mode="ddtree"),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.ddtree_topk_calls == []
    assert target_ops.restore_snapshot_shapes == [(1, 1)]
    assert target_ops.restore_acceptance_lengths == [1]
    assert summary.acceptance_history == (1,)
    assert summary.copyspec_hits == 1
    assert summary.copyspec_tokens == 3


def test_copyspec_consecutive_hits_advance_draft_contexts():
    target_ops = _FakeTargetOps()
    draft_model = _draft_model()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=8,
            prompt_tokens_override=[
                1,
                2,
                3,
                4,
                5,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            runtime_context=_runtime_context(),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls == []
    assert draft_backend.advance_context_lengths == [18, 4]
    assert summary.generated_token_ids == (0, 0, 0, 0, 0, 0, 0, 0)
    assert summary.accepted_from_draft == 6
    assert summary.acceptance_history == (3, 3)
    assert summary.copyspec_hits == 2
    assert summary.copyspec_tokens == 6


def test_copyspec_disables_after_full_rejection():
    target_ops = _FakeTargetOps()
    draft_model = _draft_model()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=4,
            prompt_tokens_override=[1, 2, 3, 4, 5, 0, 7, 7, 7, 1, 2, 3, 4, 5],
            runtime_context=_runtime_context(),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls == [(3, True)]
    assert draft_backend.advance_context_lengths == [14]
    assert summary.generated_token_ids == (0, 0, 0, 0)
    assert summary.accepted_from_draft == 2
    assert summary.acceptance_history == (0, 2)
    assert summary.copyspec_hits == 1
    assert summary.copyspec_tokens == 3


def test_profile_cycle_events_disable_async_prefetch_and_match_summary():
    target_ops = _FakeTargetOps(restore_delay_s=0.002, restore_ns=2_000_000)
    draft_model = _draft_model()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(
                diagnostics_config=DiagnosticsConfig(
                    trace=TraceConfig(cycle_events=True),
                ),
            ),
        )
    )

    cycle_events = [event for event in events if isinstance(event, CycleCompleteEvent)]
    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls == [(4, False), (2, False)]
    assert [(event.cycle, event.block_len, event.commit_count, event.acceptance_len) for event in cycle_events] == [
        (1, 4, 4, 3),
        (2, 2, 2, 1),
    ]
    first_payload = cycle_events[0].to_payload()
    assert first_payload["prefix_len"] == 2
    assert first_payload["verify_token_count"] == 4
    assert first_payload["draft_source"] == "dflash"
    assert first_payload["candidate_count"] == 1
    assert first_payload["cycle_wall_us"] >= first_payload["cycle_engine_us"]
    assert first_payload["cycle_engine_us"] == first_payload["cycle_total_us"]
    assert first_payload["commit_us"] >= 0.0
    assert first_payload["rollback_us"] == pytest.approx(2_000.0)
    assert first_payload["commit_us"] < first_payload["rollback_us"]
    assert first_payload["yield_pause_us"] >= 0.0
    for event in cycle_events:
        payload = event.to_payload()
        named_without_other = sum(
            float(payload[key])
            for key in (
                "draft_us",
                "verify_us",
                "acceptance_us",
                "hidden_extraction_us",
                "commit_us",
                "rollback_us",
            )
        )
        assert named_without_other <= payload["cycle_total_us"] + 250.0
        assert named_without_other + payload["other_us"] == pytest.approx(
            payload["cycle_total_us"],
            abs=250.0,
        )
    assert summary.cycle_profile_us == tuple(cycle_events)
    assert summary.cycle_profile_totals_us is not None
    assert set(summary.cycle_profile_totals_us) == {
        "draft",
        "verify",
        "acceptance",
        "hidden_extraction",
        "commit",
        "rollback",
        "other",
        "cycle_total",
        "cycle_wall",
        "yield_pause",
    }


def test_adaptive_verify_mode_drops_and_recovers_block_len():
    draft_model = _draft_model(block_size=8)

    class _PatternTargetOps(_FakeTargetOps):
        def __init__(self) -> None:
            super().__init__()
            self.verify_lengths: list[int] = []
            self._cycle = 0

        def verify_block(
            self,
            *,
            target_model,
            verify_ids,
            target_cache,
            capture_layer_ids,
        ):
            del target_model, target_cache, capture_layer_ids
            batch, seq_len = verify_ids.shape
            self.verify_lengths.append(int(seq_len))
            self._cycle += 1
            if self._cycle <= 4:
                pattern = [1] + [0] * (seq_len - 1)
            else:
                pattern = [0] * seq_len
            logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
            for pos, token_id in enumerate(pattern[:seq_len]):
                logits[:, pos, int(token_id)] = 1.0
            hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
            return logits, hidden

    target_ops = _PatternTargetOps()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=340,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(verify_mode="adaptive"),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls[:24] == [(8, True)] * 4 + [(4, True)] * 16 + [(8, True)] * 4
    assert target_ops.verify_lengths[:24] == [8] * 4 + [4] * 16 + [8] * 4
    assert summary.block_tokens == 8
    assert summary.verify_len_cap == 8
    assert summary.acceptance_history[:24] == (0,) * 4 + (3,) * 16 + (7,) * 4
    assert summary.adaptive_block_reductions == 1
    assert summary.adaptive_block_cycles == 16
    assert summary.adaptive_block_min == 4
    assert summary.adaptive_metrics is not None
    assert summary.adaptive_metrics["tokens_per_second_by_block"]["4"] > 1.0
    assert summary.adaptive_metrics["tokens_per_second_by_block"]["8"] > 1.0


def test_adaptive_verify_mode_starts_reduced_for_long_context():
    short_policy = spec_epoch._AdaptiveBlockPolicy.from_runtime(
        runtime_config=_runtime_context(verify_mode="adaptive").runtime,
        effective_block_tokens=8,
        verify_len_cap=8,
        prompt_len=32767,
    )
    policy = spec_epoch._AdaptiveBlockPolicy.from_runtime(
        runtime_config=_runtime_context(verify_mode="adaptive").runtime,
        effective_block_tokens=8,
        verify_len_cap=8,
        prompt_len=32768,
    )

    assert short_policy is not None
    assert short_policy.mode == "full"
    assert short_policy.block_limit() == 8
    assert short_policy.reductions == 0
    assert short_policy.reduced_burst_cycles == 64
    assert policy is not None
    assert policy.mode == "reduced"
    assert policy.block_limit() == 4
    assert policy.reductions == 1
    assert policy.reduced_burst_cycles == 64


def test_adaptive_verify_mode_long_context_burst_is_initial_only():
    policy = spec_epoch._AdaptiveBlockPolicy.from_runtime(
        runtime_config=_runtime_context(verify_mode="adaptive").runtime,
        effective_block_tokens=8,
        verify_len_cap=8,
        prompt_len=32768,
    )
    assert policy is not None

    for _ in range(15):
        policy.record(block_len=4, acceptance_len=3)

    assert policy.mode == "reduced"

    policy.record(block_len=4, acceptance_len=3)
    assert policy.mode == "probe"
    assert policy.reduced_burst_cycles == 64

    for _ in range(3):
        policy.record(block_len=8, acceptance_len=4)
        assert policy.mode == "probe"

    policy.record(block_len=8, acceptance_len=4)
    assert policy.mode == "full"

    for _ in range(4):
        policy.record(block_len=8, acceptance_len=3)

    assert policy.mode == "full"
    assert policy.reductions == 1
    assert policy.reduced_burst_cycles == 64


def test_adaptive_verify_mode_failed_early_probe_uses_full_burst_before_retry():
    policy = spec_epoch._AdaptiveBlockPolicy.from_runtime(
        runtime_config=_runtime_context(verify_mode="adaptive").runtime,
        effective_block_tokens=8,
        verify_len_cap=8,
        prompt_len=2,
    )
    assert policy is not None

    policy._enter_reduced()
    for _ in range(16):
        policy.record(block_len=4, acceptance_len=3)

    assert policy.mode == "probe"

    for _ in range(4):
        policy.record(block_len=8, acceptance_len=0)

    assert policy.mode == "reduced"
    assert policy.early_probe_lockout is True

    for _ in range(63):
        policy.record(block_len=4, acceptance_len=3)
        assert policy.mode == "reduced"

    policy.record(block_len=4, acceptance_len=3)
    assert policy.mode == "probe"
    assert policy.probe_cycles_remaining == 8


def test_adaptive_verify_mode_probe_falls_back_to_reduced_after_eight_low_full_cycles():
    policy = spec_epoch._AdaptiveBlockPolicy.from_runtime(
        runtime_config=_runtime_context(verify_mode="adaptive").runtime,
        effective_block_tokens=8,
        verify_len_cap=8,
        prompt_len=32768,
    )
    assert policy is not None

    for _ in range(16):
        policy.record(block_len=4, acceptance_len=3)

    assert policy.mode == "probe"

    for _ in range(3):
        policy.record(block_len=8, acceptance_len=0)
        assert policy.mode == "probe"

    policy.record(block_len=8, acceptance_len=0)
    assert policy.mode == "reduced"
    assert policy.reduced_cycles_since_probe == 0
    assert policy.reduced_burst_cycles == 64


def test_adaptive_verify_mode_long_context_handoff_starts_block4():
    draft_model = _draft_model(block_size=8)

    class _RecordingTargetOps(_FakeTargetOps):
        def __init__(self) -> None:
            super().__init__()
            self.verify_lengths: list[int] = []

        def verify_block(
            self,
            *,
            target_model,
            verify_ids,
            target_cache,
            capture_layer_ids,
        ):
            self.verify_lengths.append(int(verify_ids.shape[1]))
            return super().verify_block(
                target_model=target_model,
                verify_ids=verify_ids,
                target_cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )

    target_ops = _RecordingTargetOps()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=4,
            prompt_tokens_override=[1] * 32768,
            runtime_context=_runtime_context(verify_mode="adaptive"),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert target_ops.verify_lengths[:1] == [4]
    assert summary.adaptive_block_reductions == 1
    assert summary.adaptive_block_cycles == 1
    assert summary.adaptive_block_min == 4


def test_adaptive_verify_mode_keeps_full_at_acceptance_boundary():
    draft_model = _draft_model(block_size=8)

    class _MixedTargetOps(_FakeTargetOps):
        def __init__(self) -> None:
            super().__init__()
            self.verify_lengths: list[int] = []
            self._cycle = 0

        def verify_block(
            self,
            *,
            target_model,
            verify_ids,
            target_cache,
            capture_layer_ids,
        ):
            del target_model, target_cache, capture_layer_ids
            batch, seq_len = verify_ids.shape
            self.verify_lengths.append(int(seq_len))
            self._cycle += 1
            accepted = 3
            pattern = [0] + [0] * min(accepted, seq_len - 1)
            pattern += [1] * max(0, seq_len - len(pattern))
            logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
            for pos, token_id in enumerate(pattern[:seq_len]):
                logits[:, pos, int(token_id)] = 1.0
            hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
            return logits, hidden

    target_ops = _MixedTargetOps()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=64,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(verify_mode="adaptive"),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert target_ops.verify_lengths[:8] == [8] * 8
    assert summary.adaptive_block_reductions == 0
    assert summary.adaptive_block_cycles == 0
    assert summary.adaptive_block_min is None


def test_adaptive_verify_mode_drops_after_isolated_high_commit():
    policy = spec_epoch._AdaptiveBlockPolicy(full_block_tokens=8)

    policy.record(block_len=8, acceptance_len=7)
    for _ in range(3):
        policy.record(block_len=8, acceptance_len=0)

    assert policy.mode == "reduced"
    assert policy.reductions == 1


def test_adaptive_verify_mode_probe_resumes_at_acceptance_boundary():
    policy = spec_epoch._AdaptiveBlockPolicy.from_runtime(
        runtime_config=_runtime_context(verify_mode="adaptive").runtime,
        effective_block_tokens=8,
        verify_len_cap=8,
        prompt_len=32768,
    )
    assert policy is not None

    for _ in range(16):
        policy.record(block_len=4, acceptance_len=3)

    assert policy.mode == "probe"

    for _ in range(3):
        policy.record(block_len=8, acceptance_len=4)
        assert policy.mode == "probe"

    policy.record(block_len=8, acceptance_len=4)

    assert policy.mode == "full"


def test_ddtree_verify_mode_selects_branch_candidate():
    class _Embedding:
        def __call__(self, input_ids):
            return input_ids.astype(mx.float32)[..., None]

    class _DDTreeTargetOps(_FakeTargetOps):
        def __init__(self) -> None:
            super().__init__()
            self.verify_shapes: list[tuple[int, int]] = []

        def embed_tokens(self, _target_model):
            return _Embedding()

        def logits_from_hidden(self, _target_model, hidden_states):
            return hidden_states

        def verify_block(
            self,
            *,
            target_model,
            verify_ids,
            target_cache,
            capture_layer_ids,
        ):
            del target_model, target_cache, capture_layer_ids
            batch, seq_len = verify_ids.shape
            self.verify_shapes.append((int(batch), int(seq_len)))
            logits = mx.zeros((batch, seq_len, 4), dtype=mx.float32)
            for pos in range(seq_len):
                logits[:, pos, 2] = 1.0
            hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
            return logits, hidden

    class _DDTreeDraftModel:
        target_layer_ids = [0]
        block_size = 4
        mask_token_id = 0

        def project_target_hidden(self, value):
            return value

        def forward_projected_context(
            self,
            *,
            noise_embedding,
            draft_context,
            cache,
        ):
            del draft_context, cache
            rows: list[list[list[float]]] = []
            for row in noise_embedding[..., 0].tolist():
                branch_mode = len(row) > 1 and int(row[1]) == 2
                token = 2 if branch_mode else 1
                row_logits: list[list[float]] = []
                for _pos in row:
                    logits = [0.0, 0.0, 0.0, 0.0]
                    logits[token] = 2.0
                    if not branch_mode:
                        logits[2] = 1.0
                    row_logits.append(logits)
                rows.append(row_logits)
            return mx.array(rows, dtype=mx.float32)

    target_ops = _DDTreeTargetOps()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_DDTreeDraftModel(),
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=4,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(verify_mode="ddtree"),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls == []
    assert draft_backend.ddtree_topk_calls == [(4, 2)]
    assert draft_backend.ddtree_branch_batch_calls == [(2, 4)]
    assert target_ops.verify_shapes == [(3, 4)]
    assert summary.generated_token_ids == (0, 2, 2, 2)
    assert summary.acceptance_history == (3,)
    assert summary.accepted_from_draft == 3
    assert summary.cycles_completed == 1
    assert summary.tokens_per_cycle == 4.0
    assert summary.copyspec_hits == 0


def test_ddtree_target_tree_topk_uses_verify_cap_not_full_block():
    class _Embedding:
        def __call__(self, input_ids):
            return input_ids.astype(mx.float32)[..., None]

    class _TargetTreeOps(_FakeTargetOps):
        def __init__(self) -> None:
            super().__init__()
            self.tree_sizes: list[int] = []
            self.accepted_slots: list[tuple[int, ...]] = []

        def capabilities_for(self, _target_model):
            return SimpleNamespace(
                supports_prefix_snapshot=True,
                supports_tree_verify=True,
            )

        def embed_tokens(self, _target_model):
            return _Embedding()

        def logits_from_hidden(self, _target_model, hidden_states):
            return hidden_states

        def verify_tree_block(
            self,
            *,
            target_model,
            tree_inputs,
            target_cache,
            capture_layer_ids,
        ):
            del target_model, target_cache, capture_layer_ids
            tree_size = int(tree_inputs.size)
            self.tree_sizes.append(tree_size)
            logits = mx.zeros((1, tree_size, 4), dtype=mx.float32)
            hidden = {1: mx.zeros((1, tree_size, 2), dtype=mx.float32)}
            return logits, hidden

        def restore_after_tree_acceptance(
            self,
            _target_cache,
            *,
            accepted_tree_indices,
        ) -> int:
            self.accepted_slots.append(tuple(int(idx) for idx in accepted_tree_indices))
            return 0

    class _DDTreeDraftModel:
        target_layer_ids = [0]
        block_size = 8
        mask_token_id = 0

        def project_target_hidden(self, value):
            return value

        def forward_projected_context(
            self,
            *,
            noise_embedding,
            draft_context,
            cache,
        ):
            del draft_context, cache
            rows: list[list[list[float]]] = []
            for row in noise_embedding[..., 0].tolist():
                row_logits: list[list[float]] = []
                for _pos in row:
                    logits = [0.0, 0.0, 0.0, 0.0]
                    logits[1] = 2.0
                    logits[2] = 1.0
                    row_logits.append(logits)
                rows.append(row_logits)
            return mx.array(rows, dtype=mx.float32)

    target_ops = _TargetTreeOps()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_DDTreeDraftModel(),
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=8,
            prompt_tokens_override=[9, 8],
            runtime_context=_runtime_context(verify_mode="ddtree", verify_len_cap=4),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.ddtree_topk_calls[0] == (4, 2)
    assert target_ops.tree_sizes[0] == 4
    assert summary.verify_len_cap == 4
    assert summary.block_tokens == 8


def test_memory_waterfall_decode_cycle_order_is_stable():
    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_FakeTargetOps(),
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(
                diagnostics_config=DiagnosticsConfig(memory_waterfall=True),
            ),
        )
    )

    phases = [
        event.fields["memory_phase"]
        for event in events
        if isinstance(event, MemoryWaterfallEvent)
    ]
    decode_phases = [
        phase
        for phase in phases
        if phase in {"before_verify_cycle", "after_verify_cycle", "after_rollback"}
    ]

    assert decode_phases == [
        "before_verify_cycle",
        "after_verify_cycle",
        "after_rollback",
        "before_verify_cycle",
        "after_verify_cycle",
        "after_rollback",
    ]


def test_stop_token_breaks_after_current_committed_segment():
    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_FakeTargetOps(),
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            stop_token_ids=[0],
            runtime_context=_runtime_context(),
        )
    )

    token_events = [event for event in events if isinstance(event, TokenEvent)]
    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert [event.generated_tokens for event in token_events] == [1, 2, 3, 4]
    assert summary.generated_token_ids == (0, 0, 0, 0)
    assert summary.generation_tokens == 4
    assert summary.cycles_completed == 1
    assert summary.acceptance_history == (3,)


def test_stream_close_cleans_session_caches():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    stream = spec_epoch.stream_dflash_generate_impl(
        target_model=object(),
        target_ops=target_ops,
        tokenizer=object(),
        draft_model=_draft_model(),
        draft_backend=draft_backend,
        prompt="unused",
        max_new_tokens=6,
        prompt_tokens_override=[1, 2],
        runtime_context=_runtime_context(),
    )

    assert target_ops.cleanup_calls == 0
    first_event = next(stream)
    assert isinstance(first_event, PrefillProgressEvent)
    assert target_ops.cleanup_calls == 0

    stream.close()
    assert target_ops.cleanup_calls == 1


def test_prefill_snapshot_publication_skips_without_snapshot_service():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
        project_target_hidden=lambda value: value,
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=[1, 2],
            runtime_context=context,
        )
    )

    assert not any(isinstance(event, SnapshotPublishedEvent) for event in events)


def test_active_prefix_cache_requires_snapshot_service():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_defaults(
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
        project_target_hidden=lambda value: value,
    )

    try:
        list(
            spec_epoch.stream_dflash_generate_impl(
                target_model=object(),
                target_ops=target_ops,
                tokenizer=object(),
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="unused",
                max_new_tokens=0,
                prompt_tokens_override=[1, 2],
                prefix_cache_active=True,
                runtime_context=context,
            )
        )
    except ValueError as exc:
        assert "snapshot_service is required" in str(exc)
        assert target_ops.cleanup_calls == 1
    else:
        raise AssertionError("active prefix cache without snapshot service must fail")

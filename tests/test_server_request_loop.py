# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from queue import SimpleQueue
from types import SimpleNamespace

import pytest

from dflash_mlx.engine.events import (
    MemoryWaterfallEvent,
    PrefillCompleteEvent,
    SnapshotPublishedEvent,
    SummaryEvent,
    TokenEvent,
)
from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
from dflash_mlx.server import metrics as metrics_mod
from dflash_mlx.server import request_loop as request_loop_mod
from dflash_mlx.server import runtime as server_runtime_mod
from dflash_mlx.server.metrics import (
    get_live_metrics_payload,
    _reset_live_metrics_state,
    start_live_request,
)
from dflash_mlx.server.request_loop import consume_dflash_events
from dflash_mlx.server.runtime import PreparedDFlashRequest, ServerRuntime

class FakeDetokenizer:
    def __init__(self):
        self.last_segment = ""
        self.reset_calls = 0
        self.tokens = []

    def reset(self):
        self.reset_calls += 1
        self.last_segment = ""
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(int(token))
        self.last_segment = f"T{int(token)}"

    def finalize(self):
        self.last_segment = ""

class FakeTokenizer:
    eos_token_ids = {0}

    def __init__(self):
        self.detokenizer = FakeDetokenizer()

class FakeContext:
    _should_stop = False

class ClosableEvents:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True

def test_consume_dflash_events_streams_pending_token_and_summary():
    rqueue = SimpleQueue()
    events = ClosableEvents(
        [
            PrefillCompleteEvent(
                prefill_us=1.0,
                prompt_token_count=3,
                logical_ctx_tokens=3,
                physical_prefill_tokens=3,
                prefill_tokens_restored=0,
                prefill_tokens_computed=3,
            ),
            TokenEvent(token_id=10, generated_tokens=1, acceptance_ratio=0.5, cycles_completed=1),
            TokenEvent(token_id=11, generated_tokens=2, acceptance_ratio=0.5, cycles_completed=1),
            SummaryEvent(
                elapsed_us=10.0,
                prompt_token_count=3,
                generated_token_ids=(10, 11),
                generation_tokens=2,
                accepted_from_draft=1,
                acceptance_ratio=0.5,
                cycles_completed=1,
                phase_timings_us={},
            ),
        ]
    )

    result = consume_dflash_events(
        event_iter=events,
        rqueue=rqueue,
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
    )

    responses = []
    while not rqueue.empty():
        responses.append(rqueue.get())

    assert events.closed is True
    assert result.summary_event is not None
    assert result.live_token_count == 2
    assert result.finish_reason == "stop"
    assert len(responses) == 2
    assert responses[0].token == 10
    assert responses[0].text == "T10"
    assert responses[1].token == 11
    assert responses[1].text == "T11"


def test_consume_dflash_events_updates_live_metrics_from_engine_events(monkeypatch):
    _reset_live_metrics_state()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)
    start_live_request(
        request_id=15,
        mode_used="dflash",
        prompt_tokens=3,
        max_tokens=16,
        cache_hit_tokens=1,
        cache_lookup_ms=0.25,
    )
    rqueue = SimpleQueue()
    events = ClosableEvents(
        [
            PrefillCompleteEvent(
                prefill_us=1.0,
                prompt_token_count=3,
                logical_ctx_tokens=3,
                physical_prefill_tokens=2,
                prefill_tokens_restored=1,
                prefill_tokens_computed=2,
                phase_cold_us=100.0,
                phase_seam_us=20.0,
            ),
            TokenEvent(
                token_id=10,
                generated_tokens=1,
                acceptance_ratio=0.5,
                cycles_completed=1,
                adaptive_block_reductions=1,
                adaptive_block_cycles=4,
                adaptive_block_min=4,
                copyspec_hits=1,
                copyspec_tokens=3,
            ),
            SummaryEvent(
                elapsed_us=10.0,
                prompt_token_count=3,
                generated_token_ids=(10,),
                generation_tokens=1,
                accepted_from_draft=1,
                acceptance_ratio=0.5,
                cycles_completed=1,
                phase_timings_us={
                    "prefill": 1.0,
                    "draft": 2.0,
                    "verify": 3.0,
                    "replay": 4.0,
                },
                tokens_per_cycle=1.0,
                adaptive_block_reductions=1,
                adaptive_block_cycles=4,
                adaptive_block_min=4,
                copyspec_hits=1,
                copyspec_tokens=3,
            ),
        ]
    )

    result = consume_dflash_events(
        event_iter=events,
        rqueue=rqueue,
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
        request_id=15,
    )

    current = get_live_metrics_payload()["current_request"]

    assert result.summary_event is not None
    assert current["request_id"] == 15
    assert current["state"] == "finishing"
    assert current["ttft_s"] is not None
    assert current["ttft_s"] >= 0.0
    assert current["tokens_per_cycle"] == 1.0
    assert current["cycles"] == 1
    assert current["adaptive_block_reductions"] == 1
    assert current["adaptive_block_cycles"] == 4
    assert current["adaptive_block_min"] == 4
    assert current["copyspec_hits"] == 1
    assert current["copyspec_tokens"] == 3
    assert current["prefill_phase_timings_us"] == {
        "phase_cold_us": 100.0,
        "phase_seam_us": 20.0,
    }
    assert current["phase_timings_us"] == {
        "prefill": 1.0,
        "draft": 2.0,
        "verify": 3.0,
        "replay": 4.0,
    }


def test_consume_dflash_events_updates_live_cycle_metrics_before_summary(monkeypatch):
    _reset_live_metrics_state()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)
    start_live_request(
        request_id=16,
        mode_used="dflash",
        prompt_tokens=3,
        max_tokens=16,
    )
    rqueue = SimpleQueue()
    events = ClosableEvents(
        [
            PrefillCompleteEvent(
                prefill_us=1.0,
                prompt_token_count=3,
                logical_ctx_tokens=3,
                physical_prefill_tokens=3,
                prefill_tokens_restored=0,
                prefill_tokens_computed=3,
            ),
            TokenEvent(
                token_id=10,
                generated_tokens=1,
                acceptance_ratio=0.5,
                cycles_completed=1,
            ),
            TokenEvent(
                token_id=11,
                generated_tokens=2,
                acceptance_ratio=1.0,
                cycles_completed=1,
                adaptive_block_reductions=1,
                adaptive_block_cycles=4,
                adaptive_block_min=4,
                copyspec_hits=1,
                copyspec_tokens=3,
            ),
        ]
    )

    result = consume_dflash_events(
        event_iter=events,
        rqueue=rqueue,
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
        request_id=16,
    )

    current = get_live_metrics_payload()["current_request"]

    assert result.summary_event is None
    assert current["state"] == "decode"
    assert current["generated_tokens"] == 2
    assert current["cycles"] == 1
    assert current["tokens_per_cycle"] == 2.0
    assert current["acceptance_rate"] == 1.0
    assert current["adaptive_block_reductions"] == 1
    assert current["adaptive_block_cycles"] == 4
    assert current["adaptive_block_min"] == 4
    assert current["copyspec_hits"] == 1
    assert current["copyspec_tokens"] == 3


def test_consume_dflash_events_ignores_snapshot_publication_metadata():
    prefix_flow = SimpleNamespace(lookup_ms=0.1, hit_tokens=2, insert_ms=0.5)
    rqueue = SimpleQueue()
    events = ClosableEvents(
        [
            SnapshotPublishedEvent(
                kind="prefill",
                snapshot_boundary=3,
                prefix_len=3,
                insert_ms=0.2,
                admitted=True,
            ),
            SnapshotPublishedEvent(
                kind="generation",
                snapshot_boundary=5,
                prefix_len=5,
                insert_ms=0.3,
                admitted=True,
            ),
            SummaryEvent(
                elapsed_us=10.0,
                prompt_token_count=3,
                generated_token_ids=(),
                generation_tokens=0,
                accepted_from_draft=0,
                acceptance_ratio=0.0,
                cycles_completed=0,
                phase_timings_us={},
            ),
        ]
    )

    result = consume_dflash_events(
        event_iter=events,
        rqueue=rqueue,
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
        prefix_flow=prefix_flow,
    )

    assert events.closed is True
    assert rqueue.empty()
    assert result.summary_event is not None
    assert result.cache_lookup_ms == 0.1
    assert result.cache_hit_tokens == 2
    assert result.cache_insert_ms == 0.5


def test_consume_dflash_events_rejects_stale_dict_events():
    rqueue = SimpleQueue()
    events = ClosableEvents([{"event": "token", "token_id": 10}])

    with pytest.raises(TypeError, match="Unsupported DFlash engine event: dict"):
        consume_dflash_events(
            event_iter=events,
            rqueue=rqueue,
            ctx=FakeContext(),
            tokenizer=FakeTokenizer(),
            prompt=[1, 2, 3],
            max_tokens=16,
            eos_token_ids=set(),
            request_start_ns=0,
        )

    assert events.closed is True
    assert rqueue.empty()


def test_server_runtime_routes_tool_chat_generation_snapshot_policy(monkeypatch):
    _reset_live_metrics_state()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)
    captured = {}
    runtime_context = SimpleNamespace(
        runtime=SimpleNamespace(quantize_kv_cache=False),
        diagnostics=SimpleNamespace(
            trace=SimpleNamespace(log_dir=None),
            memory_waterfall=False,
        ),
    )
    provider = SimpleNamespace(
        model=object(),
        tokenizer=FakeTokenizer(),
        draft_model=object(),
        draft_backend=object(),
        target_ops=object(),
        cli_args=SimpleNamespace(runtime_context=runtime_context),
    )
    prefix_flow = SimpleNamespace(
        hit_tokens=4,
        hit_kind="l1_exact",
        lookup_ms=0.25,
        snapshot=object(),
        snapshot_service=object(),
        stable_prefix_len=3,
        cache_active=True,
        publish_generation_snapshot=True,
        insert_ms=0.0,
        prefix_cache_memory_bytes=lambda: None,
    )

    def make_prefix_flow(**kwargs):
        assert kwargs["request"].tools == [{"type": "function"}]
        return prefix_flow

    def stream(**kwargs):
        captured.update(kwargs)
        return ClosableEvents(
            [
                PrefillCompleteEvent(
                    prefill_us=1.0,
                    prompt_token_count=3,
                    logical_ctx_tokens=3,
                    physical_prefill_tokens=3,
                    prefill_tokens_restored=0,
                    prefill_tokens_computed=3,
                ),
                TokenEvent(
                    token_id=10,
                    generated_tokens=1,
                    acceptance_ratio=1.0,
                    cycles_completed=1,
                ),
                SummaryEvent(
                    elapsed_us=10.0,
                    prompt_token_count=3,
                    generated_token_ids=(10,),
                    generation_tokens=1,
                    accepted_from_draft=1,
                    acceptance_ratio=1.0,
                    cycles_completed=1,
                    phase_timings_us={},
                ),
            ]
        )

    monkeypatch.setattr(
        server_runtime_mod.PrefixCacheFlow,
        "for_request",
        staticmethod(make_prefix_flow),
    )
    monkeypatch.setattr(server_runtime_mod, "stream_dflash_generate", stream)
    monkeypatch.setattr(
        server_runtime_mod,
        "_build_generation_context",
        lambda *_args, **_kwargs: SimpleNamespace(prompt_cache_count=0, _should_stop=False),
    )
    monkeypatch.setattr(
        server_runtime_mod,
        "_finalize_request_observability",
        lambda **_kwargs: None,
    )

    runtime = ServerRuntime(
        host="127.0.0.1",
        port=8000,
        model_provider=provider,
        version="test",
    )
    request = SimpleNamespace(
        request_type="chat",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}],
    )

    runtime.serve_dflash_request(
        request_id=42,
        rqueue=SimpleQueue(),
        request=request,
        args=SimpleNamespace(
            max_tokens=16,
            stop_words=[],
            seed=None,
            chat_template_args={},
            use_default_chat_template=False,
            chat_template=None,
        ),
        prepared=PreparedDFlashRequest(
            prompt=[1, 2, 3],
            sequences={},
            state_machine=None,
            state_machine_state=None,
            has_thinking=False,
        ),
    )

    assert captured["publish_generation_snapshot"] is True
    assert captured["prefix_snapshot"] is prefix_flow.snapshot
    assert captured["snapshot_service"] is prefix_flow.snapshot_service
    assert captured["stable_prefix_len"] == 3
    assert captured["prefix_cache_active"] is True


def test_memory_waterfall_events_are_enriched_with_prefix_cache_memory():
    class FakePrefixFlow:
        lookup_ms = 0.0
        hit_tokens = 0
        insert_ms = 0.0

        def prefix_cache_memory_bytes(self):
            return {
                "l1_snapshot_bytes": 123,
                "l1_snapshot_target_hidden_bytes": 45,
            }

    runtime_context = SimpleNamespace(
        diagnostics=SimpleNamespace(memory_waterfall=True, trace=None)
    )
    rqueue = SimpleQueue()
    events = ClosableEvents(
        [
            MemoryWaterfallEvent(fields={"memory_phase": "during_decode"}),
            SummaryEvent(
                elapsed_us=10.0,
                prompt_token_count=3,
                generated_token_ids=(),
                generation_tokens=0,
                accepted_from_draft=0,
                acceptance_ratio=0.0,
                cycles_completed=0,
                phase_timings_us={},
            ),
        ]
    )

    result = consume_dflash_events(
        event_iter=events,
        rqueue=rqueue,
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
        prefix_flow=FakePrefixFlow(),
        runtime_context=runtime_context,
    )

    assert result.memory_waterfall_peak is not None
    assert result.memory_waterfall_peak["l1_snapshot_bytes"] == 123
    assert result.memory_waterfall_peak["l1_snapshot_target_hidden_bytes"] == 45
    assert result.memory_waterfall_peak["l1_snapshot_gb"] > 0.0


def test_basic_diagnostics_records_boundary_memory_snapshots(monkeypatch):
    snapshots = iter(
        [
            {"phys_footprint_bytes": 10_000_000_000, "mlx_cache_bytes": 1_000_000},
            {"phys_footprint_bytes": 12_500_000_000, "mlx_cache_bytes": 2_000_000},
        ]
    )
    monkeypatch.setattr(
        request_loop_mod,
        "process_memory_snapshot",
        lambda include_system_wired=False: next(snapshots),
    )
    runtime_context = SimpleNamespace(
        diagnostics=DiagnosticsConfig(
            mode="basic",
            memory_waterfall=False,
            trace=TraceConfig(),
        )
    )
    events = ClosableEvents(
        [
            SummaryEvent(
                elapsed_us=10.0,
                prompt_token_count=3,
                generated_token_ids=(),
                generation_tokens=0,
                accepted_from_draft=0,
                acceptance_ratio=0.0,
                cycles_completed=0,
                phase_timings_us={},
            ),
        ]
    )

    result = consume_dflash_events(
        event_iter=events,
        rqueue=SimpleQueue(),
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
        runtime_context=runtime_context,
    )

    assert result.memory_waterfall_start["memory_phase"] == "request_start"
    assert result.memory_waterfall_start["phys_footprint_gb"] == 10.0
    assert result.memory_waterfall_end["memory_phase"] == "request_end"
    assert result.memory_waterfall_end["phys_footprint_gb"] == 12.5

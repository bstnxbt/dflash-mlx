# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json

import mlx.core as mx

from tools.benchmarks.analyze_trace import (
    classify_verdict,
    compute_phase_deltas,
    load_memory_events,
    render_summary,
    summarize_events,
)
from dflash_mlx.engine.memory_waterfall import (
    collect_memory_waterfall,
    draft_cache_bytes,
    prefix_cache_bytes,
    target_cache_bytes,
)

class FakeKVCache:
    def __init__(self):
        self.keys = mx.zeros((1, 2, 3, 4), dtype=mx.float16)
        self.values = mx.zeros((1, 2, 3, 4), dtype=mx.float16)

class FakeStateKVCache:
    def __init__(self):
        self._state = (
            mx.zeros((1, 2, 3, 4), dtype=mx.float16),
            mx.zeros((1, 2, 3, 4), dtype=mx.float16),
        )

    @property
    def state(self):
        return self._state

class FakeRecurrentRollbackCache:
    def __init__(self):
        self.cache = [mx.zeros((1, 2, 3), dtype=mx.float16)]
        self._tape = [mx.zeros((1, 2, 3), dtype=mx.float16)]
        self._tape_k = None
        self._tape_g = None
        self._tape_qkv = mx.zeros((1, 2), dtype=mx.float16)
        self._snapshot = None

class FakeSnapshot:
    nbytes = 100

    def nbytes_breakdown(self):
        return {
            "fa_kv": 10,
            "gdn_state": 20,
            "target_hidden": 30,
            "last_logits": 40,
        }

class FakePrefixCache:
    def __init__(self):
        self._entries = {1: FakeSnapshot()}
        self._lock = None

    def stats(self):
        return {
            "current_bytes": 100,
            "prefix_prunes": 2,
            "cross_kind_prunes": 3,
            "byte_budget_evictions": 4,
            "l2_hits": 5,
            "l2_misses": 6,
            "l2": {
                "current_bytes": 700,
                "writes": 8,
            },
        }

def test_target_cache_bytes_splits_fa_gdn_and_tape():
    out = target_cache_bytes([FakeKVCache(), FakeRecurrentRollbackCache()])
    assert out["target_fa_kv_bytes"] == 96
    assert out["target_gdn_state_bytes"] == 12
    assert out["rollback_tape_bytes"] == 16

def test_draft_cache_bytes_counts_keys_values_and_state():
    out = draft_cache_bytes([FakeKVCache(), FakeStateKVCache()])
    assert out["draft_kv_bytes"] == 192

def test_prefix_cache_bytes_breakdown_and_stats():
    out = prefix_cache_bytes(FakePrefixCache())
    assert out["l1_snapshot_bytes"] == 100
    assert out["l1_snapshot_fa_kv_bytes"] == 10
    assert out["l1_snapshot_gdn_state_bytes"] == 20
    assert out["l1_snapshot_target_hidden_bytes"] == 30
    assert out["l1_snapshot_last_logits_bytes"] == 40
    assert out["l2_disk_bytes"] == 700
    assert out["prefix_prunes"] == 2
    assert out["cross_kind_prunes"] == 3
    assert out["byte_budget_evictions"] == 4
    assert out["l2_hits"] == 5
    assert out["l2_writes"] == 8

def test_collect_memory_waterfall_handles_missing_fields():
    out = collect_memory_waterfall(
        phase="test",
        target_cache=None,
        draft_cache=None,
        target_hidden=None,
        gen_hidden_chunks=None,
        prefix_cache=None,
    )
    assert out["memory_phase"] == "test"
    assert out["target_fa_kv_gb"] == 0.0
    assert out["target_gdn_state_gb"] == 0.0
    assert out["rollback_tape_gb"] == 0.0
    assert out["draft_kv_gb"] == 0.0
    assert out["l1_snapshot_gb"] == 0.0
    assert out["l2_disk_gb"] == 0.0

def test_collect_memory_waterfall_deduplicates_hidden_buckets():
    hidden = mx.zeros((1, 2, 3), dtype=mx.float16)
    out = collect_memory_waterfall(
        phase="test",
        target_hidden=hidden,
        gen_hidden_chunks=[hidden],
    )
    assert out["target_hidden_active_bytes"] == hidden.nbytes
    assert out["gen_hidden_chunks_bytes"] == 0

def test_analyzer_loads_jsonl_and_classifies_target_hidden(tmp_path):
    path = tmp_path / "cycle_events.jsonl"
    rows = [
        {"memory_phase": "before_verify_cycle", "target_hidden_active_gb": 2.0, "target_fa_kv_gb": 1.0},
        {"memory_phase": "after_cleanup", "mlx_cache_gb": 0.5},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    events = load_memory_events(path)
    summary = summarize_events(events)
    assert len(events) == 2
    assert summary["peaks_gb"]["target_hidden_active"] == 2.0
    assert summary["verdict"] == "optimize target_hidden first"

def test_analyzer_classifies_l1_fa_and_allocator():
    assert classify_verdict({"l1_snapshot": 3.0, "target_fa_kv": 1.0}) == "optimize L1/L2 pressure"
    assert classify_verdict({"target_fa_kv": 3.0, "target_hidden_active": 1.0}) == "paged/quantized KV worth investigating"
    assert classify_verdict({"mlx_cache": 3.0, "target_fa_kv": 1.0}) == "allocator/scratch/cache policy first"

def test_analyzer_delta_reports_positive_phase_growth():
    events = [
        {"memory_phase": "after_target_cache_create", "mlx_active_gb": 2.0},
        {"memory_phase": "after_prefill_chunk", "mlx_active_gb": 14.0, "target_fa_kv_gb": 1.0},
        {"memory_phase": "after_cleanup", "mlx_active_gb": 10.0, "mlx_cache_gb": 3.0},
    ]
    deltas = compute_phase_deltas(events)
    assert deltas["by_phase"]["after_prefill_chunk"]["mlx_active"] == 12.0
    assert deltas["by_phase"]["after_cleanup"]["mlx_cache"] == 3.0
    rendered = render_summary(summarize_events(events), include_delta=True)
    assert "after_target_cache_create -> after_prefill_chunk: mlx_active +12.000GB" in rendered

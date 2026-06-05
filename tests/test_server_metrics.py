# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest

from dflash_mlx.cache.manager import RuntimeCacheManager
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.store import PrefixSnapshotStore
from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
from dflash_mlx.engine.events import PrefillCompleteEvent, SummaryEvent
from dflash_mlx.observability import memory as memory_obs
from dflash_mlx.server import metrics as metrics_mod
from dflash_mlx.server.metrics import (
    configure_live_metrics,
    finalize_request_observability,
    get_live_metrics_payload,
    record_cycle_diagnostic,
    record_target_only_request,
    _reset_live_metrics_state,
    start_live_request,
    update_live_request,
    write_post_request_memory_line,
)
from dflash_mlx.serve import DFlashAPIHandler
from dflash_mlx.server.runtime import build_prompt_regime


def _runtime_config(**overrides):
    values = {
        "prefill_step_size": 2048,
        "draft_sink_size": 64,
        "draft_window_size": 1024,
        "verify_len_cap": 0,
        "prefix_cache": True,
        "prefix_cache_l2": False,
        "prefix_cache_l2_dir": "",
        "prefix_cache_l2_max_bytes": 0,
        "target_fa_window": 0,
        "dflash_max_ctx": 0,
        "max_snapshot_tokens": 32000,
        "clear_cache_boundaries": False,
        "verify_mode": "dflash",
        "prefix_cache_max_entries": 4,
        "prefix_cache_max_bytes": 8 * 1024**3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_tokenizer():
    return SimpleNamespace(
        chat_template="template",
        unk_token_id=-1,
        has_tool_calling=False,
        has_thinking=False,
        convert_tokens_to_ids=lambda tokens: [-1 for _ in tokens],
    )


def _configure_metrics(runtime_config=None):
    runtime_config = runtime_config or _runtime_config()
    runtime_context = SimpleNamespace(
        runtime=runtime_config,
        diagnostics=SimpleNamespace(trace=None),
    )
    configure_live_metrics(
        version="test-version",
        model_provider=SimpleNamespace(
            model_key=("target-model", None, "draft-model"),
            draft_model=SimpleNamespace(target_layer_ids=(3, 7)),
            effective_draft_quant="w4",
            draft_meta={"draft_quant_source": "model_default"},
            target_meta={},
            tokenizer=_fake_tokenizer(),
            cli_args=SimpleNamespace(
                model="target-model",
                draft_model=None,
                runtime_config=runtime_config,
                runtime_context=runtime_context,
                chat_template_args={"enable_thinking": False},
                diagnostics="off",
                metal_limits=SimpleNamespace(wired_bytes=64 * 1024**3),
            ),
        ),
    )
    return runtime_config


def _call_metrics_endpoint():
    statuses = []
    headers = []
    handler = object.__new__(DFlashAPIHandler)
    handler.path = "/metrics"
    handler.wfile = BytesIO()
    handler._set_completion_headers = lambda status=200: statuses.append(status)
    handler.send_header = lambda *args: headers.append(args)
    handler.end_headers = lambda: None

    DFlashAPIHandler.do_GET(handler)

    assert statuses == [200]
    assert any(header[0] == "Content-Length" for header in headers)
    return json.loads(handler.wfile.getvalue().decode())


class FakePrefixCache:
    def stats(self):
        return {
            "current_entries": 2,
            "max_entries": 4,
            "current_bytes": 1024,
            "max_bytes": 4096,
            "exact_hits": 1,
            "prefix_hits": 2,
            "misses": 4,
            "insertions": 6,
            "evictions": 1,
            "prefill_tokens_saved": 750,
        }


class FakePrefixCacheManager:
    def __init__(self, cache=None):
        self.cache = cache or FakePrefixCache()

    def stats(self):
        return self.cache.stats()


class BrokenStderr:
    def write(self, _line):
        raise RuntimeError("stderr closed")

    def flush(self):
        raise RuntimeError("stderr closed")


def _memory_snapshot(**overrides):
    values = {
        "mlx_active_gb": 2.0,
        "mlx_cache_gb": 0.5,
        "mlx_peak_gb": 3.0,
        "rss_gb": 4.0,
        "rss_peak_gb": 5.0,
    }
    values.update(overrides)
    return values


def _patch_memory_snapshot(monkeypatch, **overrides):
    monkeypatch.setattr(
        metrics_mod,
        "get_memory_snapshot",
        lambda: _memory_snapshot(**overrides),
    )


def _summary_event(
    *,
    generation_tokens: int,
    acceptance_ratio: float = 0.0,
    cycles_completed: int = 0,
    tokens_per_cycle: float = 0.0,
    adaptive_block_reductions: int = 0,
    adaptive_block_cycles: int = 0,
    adaptive_block_min: int | None = None,
    copyspec_hits: int = 0,
    copyspec_tokens: int = 0,
    adaptive_metrics: dict[str, object] | None = None,
    phase_timings_us: dict[str, float] | None = None,
    cycle_profile_totals_us: dict[str, float] | None = None,
    elapsed_us: float = 0.0,
    hit_kind: str = "miss",
) -> SummaryEvent:
    return SummaryEvent(
        elapsed_us=elapsed_us,
        prompt_token_count=0,
        generated_token_ids=tuple(0 for _ in range(generation_tokens)),
        generation_tokens=generation_tokens,
        accepted_from_draft=0,
        acceptance_ratio=acceptance_ratio,
        cycles_completed=cycles_completed,
        phase_timings_us=phase_timings_us or {},
        tokens_per_cycle=tokens_per_cycle,
        adaptive_block_reductions=adaptive_block_reductions,
        adaptive_block_cycles=adaptive_block_cycles,
        adaptive_block_min=adaptive_block_min,
        copyspec_hits=copyspec_hits,
        copyspec_tokens=copyspec_tokens,
        adaptive_metrics=adaptive_metrics,
        cycle_profile_totals_us=cycle_profile_totals_us,
        hit_kind=hit_kind,
    )


def _prefill_event(
    *,
    prompt_token_count: int,
    logical_ctx_tokens: int,
    physical_prefill_tokens: int,
    prefill_tokens_restored: int,
    prefill_tokens_computed: int,
    phase_cold_us: float | None = None,
    phase_seam_us: float | None = None,
) -> PrefillCompleteEvent:
    return PrefillCompleteEvent(
        prefill_us=0.0,
        prompt_token_count=prompt_token_count,
        logical_ctx_tokens=logical_ctx_tokens,
        physical_prefill_tokens=physical_prefill_tokens,
        prefill_tokens_restored=prefill_tokens_restored,
        prefill_tokens_computed=prefill_tokens_computed,
        phase_cold_us=phase_cold_us,
        phase_seam_us=phase_seam_us,
    )


def test_diagnostics_post_event_records_prefill_details(tmp_path):
    diagnostics = DiagnosticsConfig(
        mode="full",
        run_dir=tmp_path,
        trace=TraceConfig(log_dir=tmp_path, cycle_events=True),
    )

    finalize_request_observability(
        request_id=7,
        summary_event=_summary_event(
            generation_tokens=1,
            acceptance_ratio=0.0,
            cycles_completed=0,
            tokens_per_cycle=0.0,
            phase_timings_us={"prefill": 2_000_000.0},
            hit_kind="l2_prefix",
        ),
        request_start_ns=1_000_000_000,
        request_done_ns=3_500_000_000,
        first_token_ns=3_000_000_000,
        prefill_done_ns=3_000_000_000,
        prompt_token_count=4096,
        live_token_count=1,
        cache_lookup_ms=0.0,
        cache_hit_tokens=0,
        cache_insert_ms=0.0,
        finish_reason="length",
        max_tokens=1,
        memory_waterfall_start={
            "phys_footprint_bytes": 10_000_000_000,
            "phys_footprint_gb": 10.0,
        },
        memory_waterfall_end={
            "phys_footprint_bytes": 12_500_000_000,
            "phys_footprint_gb": 12.5,
        },
        prefill_event=_prefill_event(
            prompt_token_count=4096,
            logical_ctx_tokens=4096,
            physical_prefill_tokens=1024,
            prefill_tokens_restored=3072,
            prefill_tokens_computed=1024,
            phase_cold_us=1_900_000.0,
            phase_seam_us=100_000.0,
        ),
        runtime_config=SimpleNamespace(
            prefill_step_size=8192,
            draft_sink_size=64,
            draft_window_size=1024,
            verify_len_cap=0,
            prefix_cache=False,
            prefix_cache_l2=False,
            target_fa_window=0,
            dflash_max_ctx=0,
            max_snapshot_tokens=32000,
            clear_cache_boundaries=False,
            verify_mode="dflash",
        ),
        diagnostics=diagnostics,
    )

    row = json.loads((tmp_path / "post_events.jsonl").read_text().splitlines()[-1])

    assert row["prefill_tok_s"] == 2048.0
    assert row["prefill_tok_s_apparent"] == 2048.0
    assert row["prefill_tok_s_physical"] == 512.0
    assert row["prefill_tok_s_restored"] == 1536.0
    assert row["logical_ctx_tokens"] == 4096
    assert row["physical_prefill_tokens"] == 1024
    assert row["prefill_tokens_restored"] == 3072
    assert row["prefill_tokens_computed"] == 1024
    assert row["hit_kind"] == "l2_prefix"
    assert row["runtime_config"]["prefill_step_size"] == 8192
    assert "memory_waterfall" not in row["runtime_config"]
    assert row["memory_boundary_start"]["phys_footprint_bytes"] == 10_000_000_000
    assert row["memory_boundary_end"]["phys_footprint_gb"] == 12.5
    assert row["prefill_phase_timings_us"] == {
        "phase_cold_us": 1_900_000.0,
        "phase_seam_us": 100_000.0,
    }
    assert "prefill_logical_tok_s" in (tmp_path / "summary.md").read_text()


def test_target_only_request_records_live_metrics_and_post_event(tmp_path, monkeypatch):
    _reset_live_metrics_state()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)
    diagnostics = DiagnosticsConfig(
        mode="basic",
        run_dir=tmp_path,
        trace=TraceConfig(log_dir=tmp_path),
    )

    record_target_only_request(
        request_id=8,
        mode_used="ar_fastpath",
        wall_ms=125.0,
        max_tokens=32,
        diagnostics=diagnostics,
    )

    row = json.loads((tmp_path / "post_events.jsonl").read_text().splitlines()[-1])
    assert row["request_id"] == 8
    assert row["mode_used"] == "ar_fastpath"
    assert row["wall_ms"] == 125.0
    assert row["max_tokens"] == 32
    assert row["cache_hit_tokens"] == 0
    assert row["cache_status"] == "COLD"
    assert row["hit_kind"] == "miss"
    payload = get_live_metrics_payload()
    assert payload["last_request"]["request_id"] == 8
    assert payload["last_request"]["mode_used"] == "ar_fastpath"
    assert payload["last_request"]["max_tokens"] == 32
    assert payload["last_request"]["cache_status"] == "COLD"
    assert payload["last_request"]["hit_kind"] == "miss"
    assert payload["totals"]["requests"] == 1


def test_cycle_diagnostic_records_cycle_event(tmp_path):
    diagnostics = DiagnosticsConfig(
        mode="full",
        run_dir=tmp_path,
        trace=TraceConfig(log_dir=tmp_path, cycle_events=True),
    )

    record_cycle_diagnostic(
        diagnostics=diagnostics,
        request_id=9,
        fields={"cycle": 1, "commit_count": 4},
    )

    row = json.loads((tmp_path / "cycle_events.jsonl").read_text().splitlines()[-1])
    assert row["request_id"] == 9
    assert row["cycle"] == 1
    assert row["commit_count"] == 4


def test_finalize_request_observability_records_all_outputs(tmp_path, monkeypatch, capsys):
    _reset_live_metrics_state()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)
    monkeypatch.setattr(
        metrics_mod,
        "get_memory_snapshot",
        lambda: _memory_snapshot(),
    )
    diagnostics = DiagnosticsConfig(
        mode="full",
        run_dir=tmp_path,
        trace=TraceConfig(log_dir=tmp_path, cycle_events=True),
    )

    finalize_request_observability(
        request_id=10,
        summary_event=_summary_event(
            generation_tokens=4,
            acceptance_ratio=0.5,
            cycles_completed=2,
            tokens_per_cycle=2.0,
            phase_timings_us={"prefill": 1_000_000.0},
            elapsed_us=3_000_000.0,
        ),
        request_start_ns=0,
        request_done_ns=3_000_000_000,
        first_token_ns=1_500_000_000,
        prefill_done_ns=1_000_000_000,
        prompt_token_count=8,
        live_token_count=4,
        cache_lookup_ms=0.25,
        cache_hit_tokens=0,
        cache_insert_ms=0.5,
        finish_reason="stop",
        max_tokens=16,
        prompt_regime={"request_type": "chat"},
        memory_waterfall_peak={"mlx_active_gb": 6.0, "mlx_cache_gb": 1.0},
        diagnostics=diagnostics,
        prefill_event=_prefill_event(
            prompt_token_count=8,
            logical_ctx_tokens=8,
            physical_prefill_tokens=8,
            prefill_tokens_restored=0,
            prefill_tokens_computed=8,
        ),
        runtime_config=_runtime_config(prefix_cache=False),
    )

    post = json.loads((tmp_path / "post_events.jsonl").read_text().splitlines()[-1])
    assert post["request_id"] == 10
    assert post["mode_used"] == "dflash"
    assert post["prompt_tokens"] == 8
    assert post["generated_tokens"] == 4
    assert post["prefill_tok_s"] == 8.0
    assert post["prefill_tok_s_apparent"] == 8.0
    assert post["prefill_tok_s_physical"] == 8.0
    assert post["prefill_tok_s_restored"] == 0.0
    assert post["decode_tok_s"] == 2.0
    assert post["cache_status"] == "COLD"
    assert post["prefill_event"]["physical_prefill_tokens"] == 8
    assert post["memory_waterfall_peak"]["mlx_active_gb"] == 6.0
    summary = (tmp_path / "summary.md").read_text()
    assert "| 10 | 8 | 4 | 3000.0 | 1500.0 | 1000.0 | 8.00 | 8.00 | 0.00 |" in summary
    payload = get_live_metrics_payload()
    assert payload["last_request"]["request_id"] == 10
    assert payload["last_request"]["generated_tokens"] == 4
    err = capsys.readouterr().err
    assert (
        "decode 2.0 tok/s | prefill logical 8.0 tok/s | "
        "prefill real 8.0 tok/s | prefill restored 0.0 tok/s"
    ) in err
    assert "req#10 mlx_active=2.00" in err
    assert "req#10 mlx_active=6.00" not in err
    assert "memory snapshot partial" not in err


def test_diagnostics_summary_append_failure_is_signaled(
    tmp_path,
    monkeypatch,
    capsys,
):
    _reset_live_metrics_state()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)

    def fail_open(self, *_args, **_kwargs):
        if self.name == "summary.md":
            raise OSError("summary full")
        raise AssertionError(f"unexpected Path.open call for {self}")

    monkeypatch.setattr(metrics_mod.Path, "open", fail_open)
    diagnostics = DiagnosticsConfig(
        mode="full",
        run_dir=tmp_path,
        trace=TraceConfig(log_dir=tmp_path, cycle_events=True),
    )

    finalize_request_observability(
        request_id=11,
        summary_event=_summary_event(
            generation_tokens=2,
            acceptance_ratio=0.5,
            cycles_completed=1,
            tokens_per_cycle=2.0,
            phase_timings_us={},
        ),
        request_start_ns=0,
        request_done_ns=2_000_000_000,
        first_token_ns=1_000_000_000,
        prefill_done_ns=1_000_000_000,
        prompt_token_count=8,
        live_token_count=2,
        cache_lookup_ms=0.0,
        cache_hit_tokens=0,
        cache_insert_ms=0.0,
        finish_reason="stop",
        max_tokens=4,
        prefill_event=_prefill_event(
            prompt_token_count=8,
            logical_ctx_tokens=8,
            physical_prefill_tokens=8,
            prefill_tokens_restored=0,
            prefill_tokens_computed=8,
        ),
        runtime_config=_runtime_config(prefix_cache=False),
        diagnostics=diagnostics,
    )

    err = capsys.readouterr().err
    assert "diagnostics summary append failed: summary full" in err
    assert get_live_metrics_payload()["last_request"]["request_id"] == 11
    assert (tmp_path / "post_events.jsonl").exists()


def test_metrics_endpoint_returns_json_before_request(monkeypatch):
    _reset_live_metrics_state()
    _configure_metrics()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)

    payload = _call_metrics_endpoint()

    assert payload["server"]["version"] == "test-version"
    assert payload["server"]["model"] == "target-model"
    assert payload["server"]["draft"] == "draft-model"
    assert payload["server"]["draft_quant"] == "w4"
    assert payload["server"]["draft_quant_source"] == "model_default"
    assert payload["runtime"]["prefill_step_size"] == 2048
    assert payload["memory"].keys() >= {
        "rss_gb",
        "rss_peak_gb",
        "mlx_active_gb",
        "mlx_cache_gb",
        "mlx_peak_gb",
        "wired_gb",
        "wired_limit_gb",
    }
    assert payload["current_request"] is None
    assert payload["last_request"] is None
    assert payload["recent_requests"] == []
    assert payload["rates"].keys() >= {
        "requests_per_s",
        "requests_per_s_60s",
        "generated_tokens_per_s",
        "average_decode_tok_s",
        "prefill_tokens_physical_per_s",
        "prefill_tokens_restored_per_s",
        "active_decode_tok_s",
    }
    assert payload["rates"]["requests_per_s"] == 0.0
    assert payload["rates"]["requests_per_s_60s"] == 0.0
    assert payload["rates"]["average_decode_tok_s"] is None
    assert payload["prefix_cache"]["entries"] == 0
    assert payload["totals"]["requests"] == 0
    assert payload["totals"]["decode_s"] == 0.0


def test_live_memory_payload_does_not_spawn_system_wired_probe(monkeypatch):
    monkeypatch.setattr(
        memory_obs,
        "system_wired_bytes",
        lambda: (_ for _ in ()).throw(AssertionError("vm_stat should not run")),
    )

    payload = memory_obs.live_memory_payload(wired_limit_bytes=64 * 1024**3)

    assert payload["wired_gb"] is None
    assert payload["wired_limit_gb"] is not None


def test_live_memory_payload_keeps_current_and_peak_rss_separate(monkeypatch):
    monkeypatch.setattr(memory_obs, "current_rss_bytes", lambda: None)
    monkeypatch.setattr(memory_obs, "rss_peak_bytes", lambda: 5_000_000_000)

    payload = memory_obs.live_memory_payload()

    assert payload["rss_gb"] is None
    assert payload["rss_peak_gb"] == 5.0


def test_metrics_startup_clears_stale_cache_when_runtime_disables_prefix_cache(monkeypatch):
    import dflash_mlx.cache.manager as cache_manager_mod

    shutdown_calls: list[DFlashPrefixCache] = []
    original_shutdown = DFlashPrefixCache.shutdown

    def tracked_shutdown(self):
        shutdown_calls.append(self)
        return original_shutdown(self)

    stale_cache = DFlashPrefixCache()
    monkeypatch.setattr(
        cache_manager_mod,
        "_DFLASH_RUNTIME_CACHE_MANAGER",
        RuntimeCacheManager(PrefixSnapshotStore(l1=stale_cache)),
    )
    monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", ("old",))
    monkeypatch.setattr(DFlashPrefixCache, "shutdown", tracked_shutdown)
    monkeypatch.setattr(
        metrics_mod,
        "build_prefix_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("disabled prefix cache should not build a cache key")
        ),
    )
    _reset_live_metrics_state()

    _configure_metrics(_runtime_config(prefix_cache=False, target_fa_window=2048))
    payload = _call_metrics_endpoint()

    assert shutdown_calls == [stale_cache]
    assert cache_manager_mod.current_runtime_cache_manager() is None
    assert payload["runtime"]["prefix_cache_enabled"] is False
    assert payload["prefix_cache"]["entries"] is None


def test_metrics_startup_preserves_request_cache_identity(monkeypatch):
    import dflash_mlx.cache.manager as cache_manager_mod
    from dflash_mlx.server.prefix_cache_manager import build_prefix_key

    runtime_config = _runtime_config(prefix_cache=True)
    runtime_context = SimpleNamespace(
        runtime=runtime_config,
        diagnostics=SimpleNamespace(trace=None),
    )
    draft_model = SimpleNamespace(target_layer_ids=(3, 7))
    model_provider = SimpleNamespace(
        model_key=("target-model", None, "draft-model"),
        draft_model=draft_model,
        tokenizer=_fake_tokenizer(),
        cli_args=SimpleNamespace(
            model="target-model",
            draft_model=None,
            runtime_config=runtime_config,
            runtime_context=runtime_context,
            chat_template_args={"enable_thinking": False},
            diagnostics="off",
            metal_limits=SimpleNamespace(wired_bytes=64 * 1024**3),
        ),
    )
    cache_identity = build_prefix_key(model_provider, draft_model, runtime_context)
    monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
    monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
    existing = cache_manager_mod.get_runtime_cache_manager(
        runtime_context,
        cache_identity=cache_identity,
    )
    assert existing is not None

    configure_live_metrics(version="test-version", model_provider=model_provider)

    assert cache_manager_mod.current_runtime_cache_manager() is existing


def test_metrics_endpoint_treats_retired_prefix_cache_manager_as_absent(monkeypatch):
    _reset_live_metrics_state()
    _configure_metrics()
    manager = RuntimeCacheManager(PrefixSnapshotStore(l1=DFlashPrefixCache()))
    manager.shutdown()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: manager)

    payload = _call_metrics_endpoint()

    assert payload["prefix_cache"]["entries"] == 0
    assert payload["totals"]["cache_hits"] == 0
    assert payload["totals"]["cache_misses"] == 0


def test_post_request_memory_line_logs_snapshot(monkeypatch, capsys):
    _patch_memory_snapshot(monkeypatch)

    write_post_request_memory_line(request_id=7)

    err = capsys.readouterr().err
    assert "[dflash] req#7 mlx_active=2.00" in err
    assert "rss_now=4.00 rss_peak=5.00 untracked=1.50 GB" in err


def test_post_request_memory_line_logs_snapshot_failure(monkeypatch, capsys):
    def boom():
        raise RuntimeError("no memory today")

    monkeypatch.setattr(metrics_mod, "get_memory_snapshot", boom)

    write_post_request_memory_line(request_id=8)

    err = capsys.readouterr().err
    assert "[dflash] req#8 memory snapshot failed: no memory today" in err


def test_post_request_memory_line_logs_partial_snapshot(monkeypatch, capsys):
    _patch_memory_snapshot(monkeypatch, mlx_active_gb=None, rss_gb=None)

    write_post_request_memory_line(request_id=9)

    err = capsys.readouterr().err
    assert "[dflash] req#9 memory snapshot partial unavailable=mlx_active_gb,rss_gb" in err
    assert "[dflash] req#9 mlx_active=n/a" in err
    assert "rss_now=n/a" in err
    assert "untracked=n/a GB" in err


def test_post_request_memory_line_stderr_failure_does_not_raise(monkeypatch):
    writes: list[tuple[int, bytes]] = []
    monkeypatch.setattr(metrics_mod.sys, "stderr", BrokenStderr())
    monkeypatch.setattr(
        metrics_mod.os,
        "write",
        lambda fd, data: writes.append((fd, data)) or len(data),
    )
    _patch_memory_snapshot(monkeypatch)

    write_post_request_memory_line(request_id=10)

    assert writes
    assert b"observability stderr write failed: stderr closed" in writes[0][1]


def test_post_request_memory_line_stderr_and_fallback_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(metrics_mod.sys, "stderr", BrokenStderr())
    monkeypatch.setattr(
        metrics_mod.os,
        "write",
        lambda _fd, _data: (_ for _ in ()).throw(OSError("fd closed")),
    )
    _patch_memory_snapshot(monkeypatch)

    write_post_request_memory_line(request_id=11)


def test_current_rss_bytes_propagates_programmer_errors(monkeypatch):
    monkeypatch.setattr(memory_obs.sys, "platform", "linux")

    def broken_open(*_args, **_kwargs):
        raise TypeError("broken open contract")

    monkeypatch.setattr(memory_obs.builtins, "open", broken_open)

    with pytest.raises(TypeError, match="broken open contract"):
        memory_obs.current_rss_bytes()


def test_darwin_rss_helpers_propagate_programmer_errors(monkeypatch):
    def broken_cdll(*_args, **_kwargs):
        raise TypeError("broken ctypes contract")

    monkeypatch.setattr(memory_obs.ctypes, "CDLL", broken_cdll)

    with pytest.raises(TypeError, match="broken ctypes contract"):
        memory_obs.darwin_proc_resident_size_bytes()
    with pytest.raises(TypeError, match="broken ctypes contract"):
        memory_obs.darwin_task_resident_size_bytes()


def test_rss_peak_propagates_programmer_errors(monkeypatch):
    monkeypatch.setattr(
        memory_obs.resource,
        "getrusage",
        lambda _kind: SimpleNamespace(ru_maxrss=object()),
    )

    with pytest.raises(TypeError):
        memory_obs.rss_peak_bytes()


def test_mlx_memory_probe_propagates_programmer_errors(monkeypatch):
    def broken_getter():
        raise TypeError("broken mlx memory contract")

    monkeypatch.setattr(memory_obs.mx, "get_active_memory", broken_getter)

    with pytest.raises(TypeError, match="broken mlx memory contract"):
        memory_obs.mlx_memory_bytes("get_active_memory")


def test_metrics_endpoint_reports_current_request(monkeypatch):
    _reset_live_metrics_state()
    _configure_metrics()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)

    start_live_request(
        request_id=9,
        mode_used="dflash",
        prompt_tokens=4096,
        max_tokens=64,
        cache_hit_tokens=3072,
        cache_lookup_ms=1.5,
        hit_kind="l1_exact",
    )
    update_live_request(
        request_id=9,
        state="prefill",
        prefill_tokens_processed=1024,
        prefill_tokens_total=4096,
    )

    payload = _call_metrics_endpoint()

    current = payload["current_request"]
    assert current["request_id"] == 9
    assert current["state"] == "prefill"
    assert current["prompt_tokens"] == 4096
    assert current["max_tokens"] == 64
    assert current["cache_hit_tokens"] == 3072
    assert current["cache_status"] == "WARM"
    assert current["hit_kind"] == "l1_exact"
    assert current["cache_lookup_ms"] == 1.5
    assert current["prefill_tokens_processed"] == 1024
    assert current["prefill_tokens_total"] == 4096
    assert current["ttft_s"] is None
    assert current["tokens_per_cycle"] is None
    assert current["cycles"] is None
    assert current["adaptive_block_reductions"] == 0
    assert current["adaptive_block_cycles"] == 0
    assert current["adaptive_block_min"] is None
    assert current["copyspec_hits"] == 0
    assert current["copyspec_tokens"] == 0
    assert current["prefill_phase_timings_us"] == {}
    assert current["phase_timings_us"] == {}
    assert current["elapsed_s"] >= 0.0
    assert payload["last_request"] is None

    update_live_request(
        request_id=9,
        state="decode",
        generated_tokens=3,
        ttft_s=2.5,
        decode_tok_s=24.0,
        acceptance_rate=0.75,
        tokens_per_cycle=1.5,
        cycles=2,
        adaptive_block_reductions=1,
        adaptive_block_cycles=8,
        adaptive_block_min=4,
        copyspec_hits=1,
        copyspec_tokens=3,
        phase_timings_us={"draft": 1000.0, "verify": 2000.0, "replay": 300.0},
    )

    payload = _call_metrics_endpoint()

    assert payload["current_request"]["state"] == "decode"
    assert payload["current_request"]["generated_tokens"] == 3
    assert payload["current_request"]["ttft_s"] == 2.5
    assert payload["current_request"]["decode_tok_s"] == 24.0
    assert payload["current_request"]["acceptance_rate"] == 0.75
    assert payload["current_request"]["tokens_per_cycle"] == 1.5
    assert payload["current_request"]["cycles"] == 2
    assert payload["current_request"]["adaptive_block_reductions"] == 1
    assert payload["current_request"]["adaptive_block_cycles"] == 8
    assert payload["current_request"]["adaptive_block_min"] == 4
    assert payload["current_request"]["copyspec_hits"] == 1
    assert payload["current_request"]["copyspec_tokens"] == 3
    assert payload["current_request"]["phase_timings_us"] == {
        "draft": 1000.0,
        "verify": 2000.0,
        "replay": 300.0,
    }
    assert payload["rates"]["active_decode_tok_s"] == 24.0


def test_metrics_endpoint_reports_last_request_and_prefix_cache(monkeypatch):
    _reset_live_metrics_state()
    runtime_config = _configure_metrics()
    monkeypatch.setattr(
        metrics_mod,
        "current_runtime_cache_manager",
        lambda: FakePrefixCacheManager(),
    )

    finalize_request_observability(
        request_id=12,
        summary_event=_summary_event(
            generation_tokens=100,
            acceptance_ratio=0.81,
            cycles_completed=44,
            tokens_per_cycle=2.27,
            adaptive_block_reductions=2,
            adaptive_block_cycles=19,
            adaptive_block_min=4,
            adaptive_metrics={"cycles_by_block": {"4": 19, "16": 25}},
            copyspec_hits=3,
            copyspec_tokens=12,
            phase_timings_us={
                "prefill": 1_000_000.0,
                "draft": 300_000.0,
                "verify": 400_000.0,
                "replay": 50_000.0,
            },
            cycle_profile_totals_us={
                "cycle_total": 900_000.0,
                "cycle_wall": 910_000.0,
                "yield_pause": 10_000.0,
            },
            hit_kind="l1_prefix",
        ),
        request_start_ns=0,
        request_done_ns=4_000_000_000,
        first_token_ns=1_000_000_000,
        prefill_done_ns=1_000_000_000,
        prompt_token_count=1000,
        live_token_count=100,
        cache_lookup_ms=0.0,
        cache_hit_tokens=750,
        cache_insert_ms=0.0,
        finish_reason="stop",
        max_tokens=100,
        prefill_event=_prefill_event(
            prompt_token_count=1000,
            logical_ctx_tokens=1000,
            physical_prefill_tokens=250,
            prefill_tokens_restored=750,
            prefill_tokens_computed=250,
        ),
        runtime_config=runtime_config,
        diagnostics=None,
    )

    payload = _call_metrics_endpoint()

    assert payload["current_request"] is None
    last = payload["last_request"]
    assert last["request_id"] == 12
    assert last["prompt_tokens"] == 1000
    assert last["generated_tokens"] == 100
    assert last["cache_hit_tokens"] == 750
    assert last["cache_status"] == "WARM"
    assert last["hit_kind"] == "l1_prefix"
    assert last["ttft_s"] == 1.0
    assert last["prefill_tok_s_physical"] == 250.0
    assert last["prefill_tok_s_apparent"] == 1000.0
    assert last["prefill_tok_s_restored"] == 750.0
    assert last["prefill_tokens_physical"] == 250
    assert last["prefill_tokens_restored"] == 750
    assert last["prefill_tokens_computed"] == 250
    assert round(last["decode_tok_s"], 2) == 33.33
    assert last["acceptance_rate"] == 0.81
    assert last["tokens_per_cycle"] == 2.27
    assert last["cycles"] == 44
    assert last["adaptive_block_reductions"] == 2
    assert last["adaptive_block_cycles"] == 19
    assert last["adaptive_block_min"] == 4
    assert last["adaptive_metrics"] == {"cycles_by_block": {"4": 19, "16": 25}}
    assert last["copyspec_hits"] == 3
    assert last["copyspec_tokens"] == 12
    assert last["finish_reason"] == "stop"
    assert last["phase_timings_us"] == {
        "prefill": 1_000_000.0,
        "draft": 300_000.0,
        "verify": 400_000.0,
        "replay": 50_000.0,
    }
    assert last["cycle_profile_totals_us"] == {
        "cycle_total": 900_000.0,
        "cycle_wall": 910_000.0,
        "yield_pause": 10_000.0,
    }
    assert last["prefill_phase_timings_us"] == {}
    assert payload["prefix_cache"]["hits"] == 3
    assert payload["prefix_cache"]["misses"] == 4
    assert payload["prefix_cache"]["last_restored_tokens"] == 750
    assert payload["prefix_cache"]["last_computed_tokens"] == 250
    assert payload["totals"]["requests"] == 1
    assert payload["totals"]["generated_tokens"] == 100
    assert payload["totals"]["decode_s"] == 3.0
    assert payload["totals"]["prefill_tokens_physical"] == 250
    assert payload["totals"]["prefill_tokens_restored"] == 750
    assert payload["rates"]["requests_per_s"] > 0.0
    assert payload["rates"]["requests_per_s_60s"] > 0.0
    assert payload["rates"]["generated_tokens_per_s"] > 0.0
    assert round(payload["rates"]["average_decode_tok_s"], 2) == 33.33
    assert payload["rates"]["active_decode_tok_s"] == last["decode_tok_s"]
    assert len(payload["recent_requests"]) == 1
    assert payload["recent_requests"][0]["request_id"] == 12
    assert payload["recent_requests"][0]["cache_status"] == "WARM"
    assert payload["recent_requests"][0]["adaptive_block_cycles"] == 19
    assert payload["recent_requests"][0]["adaptive_metrics"] == {
        "cycles_by_block": {"4": 19, "16": 25}
    }
    assert payload["recent_requests"][0]["cycle_profile_totals_us"] == {
        "cycle_total": 900_000.0,
        "cycle_wall": 910_000.0,
        "yield_pause": 10_000.0,
    }
    assert payload["recent_requests"][0]["tokens_per_cycle"] == 2.27


def test_metrics_average_decode_ignores_non_decode_requests(monkeypatch):
    _reset_live_metrics_state()
    runtime_config = _configure_metrics()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)

    finalize_request_observability(
        request_id=20,
        summary_event=_summary_event(
            generation_tokens=100,
            acceptance_ratio=0.8,
            cycles_completed=20,
            tokens_per_cycle=5.0,
        ),
        request_start_ns=0,
        request_done_ns=4_000_000_000,
        first_token_ns=1_000_000_000,
        prefill_done_ns=1_000_000_000,
        prompt_token_count=1000,
        live_token_count=100,
        cache_lookup_ms=0.0,
        cache_hit_tokens=0,
        cache_insert_ms=0.0,
        finish_reason="stop",
        max_tokens=100,
        runtime_config=runtime_config,
        diagnostics=None,
    )
    payload = get_live_metrics_payload()
    assert payload["totals"]["generated_tokens"] == 100
    assert payload["totals"]["decode_s"] == 3.0
    assert round(payload["rates"]["average_decode_tok_s"], 2) == 33.33

    record_target_only_request(
        request_id=21,
        mode_used="ar_fastpath",
        wall_ms=500.0,
        max_tokens=16,
        diagnostics=None,
    )
    payload = get_live_metrics_payload()
    assert payload["totals"]["generated_tokens"] == 100
    assert payload["totals"]["decode_s"] == 3.0
    assert round(payload["rates"]["average_decode_tok_s"], 2) == 33.33

    finalize_request_observability(
        request_id=22,
        summary_event=_summary_event(
            generation_tokens=0,
            acceptance_ratio=0.0,
            cycles_completed=0,
            tokens_per_cycle=0.0,
        ),
        request_start_ns=0,
        request_done_ns=1_200_000_000,
        first_token_ns=None,
        prefill_done_ns=1_000_000_000,
        prompt_token_count=1000,
        live_token_count=0,
        cache_lookup_ms=0.0,
        cache_hit_tokens=0,
        cache_insert_ms=0.0,
        finish_reason="stop",
        max_tokens=16,
        runtime_config=runtime_config,
        diagnostics=None,
    )
    payload = get_live_metrics_payload()
    assert payload["last_request"]["request_id"] == 22
    assert payload["last_request"]["generated_tokens"] == 0
    assert payload["last_request"]["decode_s"] == 0.2
    assert payload["last_request"]["decode_tok_s"] == 0.0
    assert payload["totals"]["generated_tokens"] == 100
    assert payload["totals"]["decode_s"] == 3.0
    assert round(payload["rates"]["average_decode_tok_s"], 2) == 33.33

    _reset_live_metrics_state()
    payload = get_live_metrics_payload()
    assert payload["totals"]["decode_s"] == 0.0
    assert payload["rates"]["average_decode_tok_s"] is None


def test_prompt_regime_distinguishes_text_completion_from_chat():
    tokenizer = SimpleNamespace(has_chat_template=True)
    args = SimpleNamespace(
        chat_template_args={"enable_thinking": True},
        use_default_chat_template=True,
        chat_template="template",
    )

    completion = build_prompt_regime(
        args,
        tokenizer,
        SimpleNamespace(request_type="text"),
    )
    chat = build_prompt_regime(
        args,
        tokenizer,
        SimpleNamespace(request_type="chat"),
    )

    assert completion["request_type"] == "text"
    assert completion["chat_template"] is False
    assert completion["chat_template_args"] == {}
    assert chat["request_type"] == "chat"
    assert chat["chat_template"] is True
    assert chat["enable_thinking"] is True

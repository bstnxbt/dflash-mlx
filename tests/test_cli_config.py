# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

import dflash_mlx.metal_limits as metal_limits
from dflash_mlx.server import model_provider
from dflash_mlx.server.config import (
    MetalLimitConfig,
    build_parser,
    configure_metal_limits,
    normalize_cli_args,
)

_RUNTIME_ENV_KEYS = (
    "DFLASH_PREFILL_STEP_SIZE",
    "DFLASH_DRAFT_SINK_SIZE",
    "DFLASH_DRAFT_WINDOW_SIZE",
    "DFLASH_VERIFY_LEN_CAP",
    "DFLASH_PREFIX_CACHE",
    "DFLASH_PREFIX_CACHE_MAX_ENTRIES",
    "DFLASH_PREFIX_CACHE_MAX_BYTES",
    "DFLASH_CLEAR_CACHE_BOUNDARIES",
    "DFLASH_MAX_SNAPSHOT_TOKENS",
    "DFLASH_PREFIX_CACHE_L2_ENABLED",
    "DFLASH_PREFIX_CACHE_L2_DIR",
    "DFLASH_PREFIX_CACHE_L2_MAX_BYTES",
    "DFLASH_TARGET_FA_WINDOW",
    "DFLASH_MAX_CTX",
    "DFLASH_VERIFY_MODE",
    "DFLASH_VERIFY_LINEAR",
    "DFLASH_VERIFY_QMM",
    "DFLASH_QUANTIZE_KV_CACHE",
)

def _clear_runtime_env(monkeypatch):
    for key in _RUNTIME_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

def test_serve_cli_wires_runtime_env_flags(monkeypatch):
    _clear_runtime_env(monkeypatch)

    args = build_parser().parse_args(
        [
            "--model",
            "m",
            "--prefill-step-size",
            "4096",
            "--clear-cache-boundaries",
        ]
    )
    normalize_cli_args(args)

    assert args.runtime_config.prefill_step_size == 4096
    assert args.runtime_config.clear_cache_boundaries is True

def test_serve_cli_thinking_uses_tokenizer_default(monkeypatch):
    _clear_runtime_env(monkeypatch)
    parser = build_parser()

    args = parser.parse_args(["--model", "m"])
    normalize_cli_args(args)

    assert args.enable_thinking is False
    assert args.chat_template_args == {}
    assert args.fastpath_max_tokens == 0

    args = parser.parse_args(["--model", "m", "--enable-thinking"])
    normalize_cli_args(args)
    assert args.chat_template_args["enable_thinking"] is True

def test_serve_cli_fastpath_max_tokens_zero_is_accepted(monkeypatch):
    _clear_runtime_env(monkeypatch)

    args = build_parser().parse_args(["--model", "m", "--fastpath-max-tokens", "0"])
    normalize_cli_args(args)

    assert args.fastpath_max_tokens == 0

def test_serve_cli_chat_template_args_can_enable_thinking(monkeypatch):
    _clear_runtime_env(monkeypatch)

    args = build_parser().parse_args(
        ["--model", "m", "--chat-template-args", '{"enable_thinking":true}']
    )
    normalize_cli_args(args)

    assert args.chat_template_args["enable_thinking"] is True

@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "m", "--memory-waterfall"],
        ["--model", "m", "--bench-log-dir", "/tmp/dflash-logs"],
    ],
)
def test_serve_rejects_removed_diagnostics_aliases(argv):
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)

    assert exc.value.code == 2


@pytest.mark.parametrize(
    "env_key",
    [
        "DFLASH_PREFIX_CACHE",
        "DFLASH_CLEAR_CACHE_BOUNDARIES",
        "DFLASH_PREFIX_CACHE_L2_ENABLED",
    ],
)
def test_runtime_bool_env_rejects_invalid_values(monkeypatch, env_key):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv(env_key, "maybe")
    args = build_parser().parse_args(["--model", "m"])

    with pytest.raises(SystemExit) as exc:
        normalize_cli_args(args)

    assert env_key in str(exc.value)


def test_model_provider_adds_mlx_lm_server_boundary_defaults(monkeypatch):
    _clear_runtime_env(monkeypatch)
    target_ops = object()
    draft_backend = object()
    load_calls: list[dict] = []

    class FakeGroup:
        def size(self):
            return 1

    monkeypatch.setattr(model_provider.mx.distributed, "init", lambda: FakeGroup())
    def fake_load_runtime_bundle(**kwargs):
        load_calls.append(kwargs)
        return SimpleNamespace(
            target_model=SimpleNamespace(parameters=lambda: []),
            tokenizer=SimpleNamespace(chat_template=None, default_chat_template=None),
            draft_model=SimpleNamespace(parameters=lambda: []),
            target_meta={},
            draft_meta={"draft_quant_source": "model_default"},
            draft_backend=draft_backend,
            target_ops=target_ops,
            resolved_draft_ref="draft",
            effective_draft_quant="w4",
        )

    monkeypatch.setattr(model_provider, "load_runtime_bundle", fake_load_runtime_bundle)
    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)

    for attr in model_provider._MLX_LM_SERVER_DEFAULTS:
        assert not hasattr(args, attr)

    provider = model_provider.DFlashModelProvider(args)

    assert provider.cli_args is args
    for attr, default in model_provider._MLX_LM_SERVER_DEFAULTS.items():
        assert getattr(args, attr) == default
    provider.load("default_model")
    assert provider.target_ops is target_ops
    assert provider.draft_backend is draft_backend
    assert "split_full_attention_sdpa" not in load_calls[0]
    assert provider.effective_draft_quant == "w4"
    assert provider.draft_meta["draft_quant_source"] == "model_default"


def test_model_provider_preserves_dflash_fields_if_parent_preloads(monkeypatch):
    _clear_runtime_env(monkeypatch)
    target_ops = object()
    draft_backend = object()

    def fake_parent_init(self, cli_args):
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None
        self._model_map = {"default_model": cli_args.model}
        self.load("default_model")

    monkeypatch.setattr(model_provider.mlx_server.ModelProvider, "__init__", fake_parent_init)
    monkeypatch.setattr(
        model_provider,
        "load_runtime_bundle",
        lambda **_kwargs: SimpleNamespace(
            target_model=SimpleNamespace(parameters=lambda: []),
            tokenizer=SimpleNamespace(chat_template=None, default_chat_template=None),
            draft_model=SimpleNamespace(parameters=lambda: []),
            target_meta={},
            draft_meta={"draft_quant_source": "none"},
            draft_backend=draft_backend,
            target_ops=target_ops,
            resolved_draft_ref="draft",
            effective_draft_quant=None,
        ),
    )

    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)
    provider = model_provider.DFlashModelProvider(args)

    assert provider.model_key == ("m", None, "draft")
    assert provider.target_ops is target_ops
    assert provider.draft_backend is draft_backend


def test_model_provider_materialization_failure_does_not_publish_loaded_state(monkeypatch):
    _clear_runtime_env(monkeypatch)

    class FakeGroup:
        def size(self):
            return 1

    target_ops = object()
    draft_backend = object()
    target_model = SimpleNamespace(parameters=lambda: ["target-param"])
    draft_model = SimpleNamespace(parameters=lambda: ["draft-param"])

    monkeypatch.setattr(model_provider.mx.distributed, "init", lambda: FakeGroup())
    monkeypatch.setattr(
        model_provider,
        "load_runtime_bundle",
        lambda **_kwargs: SimpleNamespace(
            target_model=target_model,
            tokenizer=SimpleNamespace(chat_template=None, default_chat_template=None),
            draft_model=draft_model,
            target_meta={},
            draft_meta={"draft_quant_source": "model_default"},
            draft_backend=draft_backend,
            target_ops=target_ops,
            resolved_draft_ref="draft",
            effective_draft_quant="w4",
        ),
    )
    monkeypatch.setattr(
        model_provider.mx,
        "eval",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad weights")),
    )

    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)
    provider = model_provider.DFlashModelProvider(args)

    with pytest.raises(RuntimeError, match="DFlash weight materialization failed"):
        provider.load("default_model")

    assert provider.model is None
    assert provider.tokenizer is None
    assert provider.draft_model is None
    assert provider.draft_backend is None
    assert provider.draft_meta is None
    assert provider.effective_draft_quant is None
    assert provider.target_meta is None
    assert provider.target_ops is None
    assert provider.model_key is None


def test_wait_for_initial_model_load_requires_complete_runtime_bundle():
    provider = SimpleNamespace(
        model_key=("m", None, "draft"),
        target_ops=None,
        draft_backend=object(),
    )

    with pytest.raises(RuntimeError, match="complete runtime bundle"):
        model_provider.wait_for_initial_model_load(
            provider,
            timeout_s=0.0,
            poll_interval_s=0.0,
        )

def test_serve_cli_prefill_step_size_is_not_decorative(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFILL_STEP_SIZE", "8192")
    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)
    assert args.runtime_config.prefill_step_size == 8192

    args = build_parser().parse_args(["--model", "m", "--prefill-step-size", "2048"])
    normalize_cli_args(args)
    assert args.runtime_config.prefill_step_size == 2048

def test_serve_cli_threads_draft_and_verify_runtime_flags(monkeypatch):
    _clear_runtime_env(monkeypatch)
    args = build_parser().parse_args(
        [
            "--model",
            "m",
            "--draft-sink-size",
            "32",
            "--draft-window-size",
            "512",
            "--verify-len-cap",
            "8",
        ]
    )
    normalize_cli_args(args)

    assert args.runtime_config.draft_sink_size == 32
    assert args.runtime_config.draft_window_size == 512
    assert args.runtime_config.verify_len_cap == 8
    assert args.runtime_context.runtime.draft_sink_size == 32
    assert args.runtime_context.runtime.draft_window_size == 512
    assert args.runtime_context.runtime.verify_len_cap == 8

def test_diagnostics_basic_resolves_default_artifact_dir(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    args = build_parser().parse_args(["--model", "m", "--diagnostics", "basic"])
    normalize_cli_args(args)

    out = Path(args.diagnostics_dir_resolved)
    assert out.parts[:3] == (".artifacts", "dflash", "diagnostics")
    assert out.name.endswith("-serve-basic")
    assert args.diagnostics_config.run_dir == out
    assert args.diagnostics_config.trace.log_dir == out
    assert args.diagnostics_config.trace.cycle_events is False
    assert (out / "manifest.json").exists()
    assert (out / "invocation.json").exists()
    assert (out / "effective_config.json").exists()
    assert (out / "post_events.jsonl").exists()
    assert (out / "cache_events.jsonl").exists()
    assert not (out / "cycle_events.jsonl").exists()

def test_diagnostics_full_enables_memory_and_cycle_log(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    args = build_parser().parse_args(["--model", "m", "--diagnostics", "full"])
    normalize_cli_args(args)

    out = Path(args.diagnostics_dir_resolved)
    assert out.parts[:3] == (".artifacts", "dflash", "diagnostics")
    assert out.name.endswith("-serve-full")
    assert not hasattr(args.runtime_config, "memory_waterfall")
    assert args.diagnostics_config.run_dir == out
    assert args.diagnostics_config.memory_waterfall is True
    assert args.diagnostics_config.trace.log_dir == out
    assert args.diagnostics_config.trace.cycle_events is True
    assert (out / "cycle_events.jsonl").exists()

def test_diagnostics_off_creates_no_artifact_dir(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)

    assert not Path(".artifacts").exists()
    assert not hasattr(args, "diagnostics_dir_resolved")
    assert args.diagnostics_config.mode == "off"
    assert args.diagnostics_config.trace.log_dir is None

def test_diagnostics_dir_override_is_exact(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    out = tmp_path / "custom-diagnostics"

    args = build_parser().parse_args(
        ["--model", "m", "--diagnostics", "basic", "--diagnostics-dir", str(out)]
    )
    normalize_cli_args(args)

    assert Path(args.diagnostics_dir_resolved) == out
    assert args.diagnostics_config.run_dir == out
    assert args.diagnostics_config.trace.log_dir == out
    assert (out / "manifest.json").exists()

def test_diagnostics_manifest_contains_required_fields(monkeypatch, tmp_path):
    _clear_runtime_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    args = build_parser().parse_args(["--model", "m", "--draft", "d", "--diagnostics", "basic"])
    normalize_cli_args(args)

    manifest = (Path(args.diagnostics_dir_resolved) / "manifest.json").read_text()
    for text in (
        '"kind": "diagnostics"',
        '"argv":',
        '"cwd":',
        '"git_sha":',
        '"git_dirty":',
        '"timestamp":',
        '"python_version":',
        '"platform":',
        '"model": "m"',
        '"draft": "d"',
        '"output_schema_version": 1',
    ):
        assert text in manifest

    effective = json.loads(
        (Path(args.diagnostics_dir_resolved) / "effective_config.json").read_text()
    )
    invocation = json.loads(
        (Path(args.diagnostics_dir_resolved) / "invocation.json").read_text()
    )
    assert "diagnostics" not in effective
    assert "diagnostics_dir" not in effective
    assert "memory_waterfall" not in effective
    assert invocation["diagnostics"]["mode"] == "basic"
    assert invocation["diagnostics"]["memory_waterfall"] is False

def test_serve_cli_parses_memory_limit_flags():
    args = build_parser().parse_args(
        ["--model", "m", "--wired-limit", "48GB", "--cache-limit", "8GB"]
    )
    assert args.wired_limit == 48 * 1024**3
    assert args.cache_limit == 8 * 1024**3

def test_serve_cli_rejects_invalid_memory_limit():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "m", "--wired-limit", "bad"])

def test_serve_cli_parses_prefix_cache_byte_suffixes(monkeypatch):
    _clear_runtime_env(monkeypatch)

    args = build_parser().parse_args(
        [
            "--model",
            "m",
            "--prefix-cache-max-bytes",
            "2GB",
            "--prefix-cache-l2-max-bytes",
            "50GB",
        ]
    )
    normalize_cli_args(args)

    assert args.runtime_config.prefix_cache_max_bytes == 2 * 1024**3
    assert args.runtime_config.prefix_cache_l2_max_bytes == 50 * 1024**3

def test_serve_cli_parses_prefix_cache_byte_suffix_env(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFIX_CACHE_MAX_BYTES", "2GB")
    monkeypatch.setenv("DFLASH_PREFIX_CACHE_L2_MAX_BYTES", "50GB")

    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)

    assert args.runtime_config.prefix_cache_max_bytes == 2 * 1024**3
    assert args.runtime_config.prefix_cache_l2_max_bytes == 50 * 1024**3

def _install_metal_limit_probe(monkeypatch):
    calls = []
    recommended = 64 * 1024**3

    class Metal:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(metal_limits.mx, "metal", Metal())
    monkeypatch.setattr(
        metal_limits.mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": recommended},
    )
    monkeypatch.setattr(
        metal_limits.mx,
        "set_wired_limit",
        lambda value: calls.append(("wired", value)),
    )
    monkeypatch.setattr(
        metal_limits.mx,
        "set_cache_limit",
        lambda value: calls.append(("cache", value)),
    )
    return calls, recommended

def test_configure_metal_limits_uses_bounded_default(monkeypatch):
    calls, recommended = _install_metal_limit_probe(monkeypatch)

    args = build_parser().parse_args(["--model", "m"])
    limits = configure_metal_limits(args)

    assert calls == [("wired", recommended), ("cache", 4 * 1024**3)]
    assert limits.wired_request == "auto"
    assert limits.wired_bytes == recommended
    assert limits.cache_request == 4 * 1024**3
    assert limits.cache_bytes == 4 * 1024**3

def test_serve_default_uses_bounded_cache_limit(monkeypatch):
    _clear_runtime_env(monkeypatch)
    calls, recommended = _install_metal_limit_probe(monkeypatch)

    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)
    limits = configure_metal_limits(args)

    assert calls == [("wired", recommended), ("cache", 4 * 1024**3)]
    assert args.cache_limit is None
    assert limits.cache_request == 4 * 1024**3
    assert limits.cache_bytes == 4 * 1024**3

def test_explicit_auto_cache_limit_overrides_serve_default(monkeypatch):
    _clear_runtime_env(monkeypatch)
    calls, recommended = _install_metal_limit_probe(monkeypatch)

    args = build_parser().parse_args(["--model", "m", "--cache-limit", "auto"])
    normalize_cli_args(args)
    limits = configure_metal_limits(args)

    assert calls == [("wired", recommended), ("cache", recommended // 4)]
    assert limits.cache_request == "auto"
    assert limits.cache_bytes == recommended // 4

def test_explicit_none_cache_limit_overrides_serve_default(monkeypatch):
    _clear_runtime_env(monkeypatch)
    calls, recommended = _install_metal_limit_probe(monkeypatch)

    args = build_parser().parse_args(["--model", "m", "--cache-limit", "none"])
    normalize_cli_args(args)
    limits = configure_metal_limits(args)

    assert calls == [("wired", recommended)]
    assert limits.cache_request == "none"
    assert limits.cache_bytes is None
    assert limits.cache_applied is False

def test_explicit_byte_cache_limit_overrides_serve_default(monkeypatch):
    _clear_runtime_env(monkeypatch)
    calls, recommended = _install_metal_limit_probe(monkeypatch)

    args = build_parser().parse_args(["--model", "m", "--cache-limit", "8GB"])
    normalize_cli_args(args)
    limits = configure_metal_limits(args)

    assert calls == [("wired", recommended), ("cache", 8 * 1024**3)]
    assert limits.cache_request == 8 * 1024**3
    assert limits.cache_bytes == 8 * 1024**3

def test_configure_metal_limits_supports_none_and_explicit_cache(monkeypatch):
    calls, _recommended = _install_metal_limit_probe(monkeypatch)

    args = build_parser().parse_args(
        ["--model", "m", "--wired-limit", "none", "--cache-limit", "8GB"]
    )
    limits = configure_metal_limits(args)

    assert calls == [("cache", 8 * 1024**3)]
    assert limits.wired_request == "none"
    assert limits.wired_bytes is None
    assert limits.wired_applied is False
    assert limits.cache_bytes == 8 * 1024**3
    assert limits.cache_applied is True


def test_configure_metal_limits_handles_missing_recommended_key(monkeypatch):
    calls = []

    class Metal:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(metal_limits.mx, "metal", Metal())
    monkeypatch.setattr(
        metal_limits.mx,
        "device_info",
        lambda: {"architecture": "applegpu_g13s"},
    )
    monkeypatch.setattr(
        metal_limits.mx,
        "set_wired_limit",
        lambda value: calls.append(("wired", value)),
    )
    monkeypatch.setattr(
        metal_limits.mx,
        "set_cache_limit",
        lambda value: calls.append(("cache", value)),
    )

    args = build_parser().parse_args(["--model", "m"])
    limits = configure_metal_limits(args)

    assert calls == [("cache", 4 * 1024**3)]
    assert limits.metal_available is True
    assert limits.recommended_bytes is None
    assert limits.wired_applied is False
    assert limits.cache_bytes == 4 * 1024**3
    assert limits.cache_applied is True
    assert "max_recommended_working_set_size" in str(limits.warning)


def test_configure_metal_limits_applies_explicit_limits_without_recommended_key(
    monkeypatch,
):
    calls = []

    class Metal:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(metal_limits.mx, "metal", Metal())
    monkeypatch.setattr(
        metal_limits.mx,
        "device_info",
        lambda: {"architecture": "applegpu_g13s"},
    )
    monkeypatch.setattr(
        metal_limits.mx,
        "set_wired_limit",
        lambda value: calls.append(("wired", value)),
    )
    monkeypatch.setattr(
        metal_limits.mx,
        "set_cache_limit",
        lambda value: calls.append(("cache", value)),
    )

    args = build_parser().parse_args(
        ["--model", "m", "--wired-limit", "48GB", "--cache-limit", "8GB"]
    )
    limits = configure_metal_limits(args)

    assert calls == [("wired", 48 * 1024**3), ("cache", 8 * 1024**3)]
    assert limits.recommended_bytes is None
    assert limits.wired_bytes == 48 * 1024**3
    assert limits.cache_bytes == 8 * 1024**3
    assert limits.wired_applied is True
    assert limits.cache_applied is True
    assert "max_recommended_working_set_size" in str(limits.warning)


def test_startup_banner_prints_resolved_metal_limits(monkeypatch, capsys):
    from dflash_mlx.server.runtime import ServerRuntime

    limits = MetalLimitConfig(
        metal_available=True,
        recommended_bytes=64 * 1024**3,
        wired_request="auto",
        wired_bytes=64 * 1024**3,
        wired_applied=True,
        cache_request=8 * 1024**3,
        cache_bytes=8 * 1024**3,
        cache_applied=True,
    )
    provider = SimpleNamespace(
        model_key=("target", None, "draft"),
        effective_draft_quant="w4",
        draft_meta={"draft_quant_source": "model_default"},
        target_meta={},
        cli_args=SimpleNamespace(
            model="target",
            draft_model=None,
            runtime_config=None,
            metal_limits=limits,
            chat_template_args={"enable_thinking": True},
            fastpath_max_tokens=0,
        ),
    )

    runtime = ServerRuntime(
        host="127.0.0.1",
        port=8000,
        model_provider=provider,
        version="test-version",
    )
    runtime.print_startup_banner()

    err = capsys.readouterr().err
    assert "Draft quant:  w4 (model_default)" in err
    assert "Thinking:     enabled" in err
    assert "Fast path:    off" in err
    assert "Wired limit: auto -> 64.0 GiB" in err
    assert "Cache limit: 8.0 GiB -> 8.0 GiB" in err

    provider.cli_args.chat_template_args = {}
    provider.cli_args.fastpath_max_tokens = 64
    runtime.print_startup_banner()
    err = capsys.readouterr().err
    assert "Thinking:     disabled" in err
    assert "Fast path:    AR <= 64 tokens" in err

@pytest.mark.parametrize(
    "argv,error",
    [
        (["--model", "m", "--prefill-step-size", "0"], "--prefill-step-size"),
        (["--model", "m", "--draft-sink-size", "-1"], "draft_sink_size"),
        (["--model", "m", "--draft-window-size", "0"], "draft_window_size"),
        (["--model", "m", "--verify-len-cap", "-1"], "verify_len_cap"),
        (["--model", "m", "--fastpath-max-tokens", "-1"], "--fastpath-max-tokens"),
    ],
)
def test_serve_cli_rejects_invalid_runtime_flags(argv, error):
    args = build_parser().parse_args(argv)
    with pytest.raises(SystemExit, match=error):
        normalize_cli_args(args)

def test_serve_default_sets_product_runtime(monkeypatch):
    _clear_runtime_env(monkeypatch)
    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)
    cfg = args.runtime_config
    assert cfg.prefill_step_size == 2048
    assert cfg.draft_sink_size == 64
    assert cfg.draft_window_size == 1024
    assert cfg.verify_len_cap == 0
    assert cfg.prefix_cache is True
    assert cfg.prefix_cache_max_entries == 8
    assert cfg.prefix_cache_max_bytes == 8 * 1024**3
    assert cfg.clear_cache_boundaries is True
    assert cfg.max_snapshot_tokens == 32000
    assert cfg.prefix_cache_l2 is True
    assert cfg.verify_mode == "adaptive"

def test_cli_explicit_overrides_default_runtime(monkeypatch):
    _clear_runtime_env(monkeypatch)
    args = build_parser().parse_args(["--model", "m", "--prefill-step-size", "2048"])
    normalize_cli_args(args)
    assert args.runtime_config.prefill_step_size == 2048

def test_env_overrides_default_runtime(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFILL_STEP_SIZE", "2048")
    monkeypatch.setenv("DFLASH_DRAFT_WINDOW_SIZE", "512")
    monkeypatch.setenv("DFLASH_VERIFY_LEN_CAP", "4")
    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)
    assert args.runtime_config.prefill_step_size == 2048
    assert args.runtime_config.draft_window_size == 512
    assert args.runtime_config.verify_len_cap == 4

def test_cli_overrides_env(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFILL_STEP_SIZE", "1024")
    args = build_parser().parse_args(["--model", "m", "--prefill-step-size", "8192"])
    normalize_cli_args(args)
    assert args.runtime_config.prefill_step_size == 8192

def test_verify_mode_off_disables_custom_verify(monkeypatch):
    _clear_runtime_env(monkeypatch)
    args = build_parser().parse_args(["--model", "m", "--verify-mode", "off"])
    normalize_cli_args(args)
    assert "DFLASH_VERIFY_LINEAR" not in os.environ
    assert "DFLASH_VERIFY_QMM" not in os.environ
    assert args.runtime_config.verify_mode == "off"
    assert args.runtime_context.verify.mode == "off"

@pytest.mark.parametrize(
    "argv,error",
    [
        (["--model", "m", "--prefix-cache-max-entries", "0"], "prefix_cache_max_entries"),
        (["--model", "m", "--prefix-cache-max-bytes", "-1"], "prefix_cache_max_bytes"),
        (["--model", "m", "--max-snapshot-tokens", "-1"], "max_snapshot_tokens"),
        (
            ["--model", "m", "--prefix-cache-l2", "--prefix-cache-l2-dir", ""],
            "prefix_cache_l2_dir",
        ),
    ],
)
def test_runtime_validation_errors(argv, error):
    args = build_parser().parse_args(argv)
    with pytest.raises(SystemExit, match=error):
        normalize_cli_args(args)

def test_target_fa_window_disables_prefix_cache(monkeypatch):
    _clear_runtime_env(monkeypatch)
    args = build_parser().parse_args(["--model", "m", "--target-fa-window", "2048"])
    normalize_cli_args(args)
    assert args.runtime_config.target_fa_window == 2048
    assert args.runtime_config.prefix_cache is False
    assert args.runtime_config.prefix_cache_l2 is False

def test_prefix_cache_disabled_disables_l2(monkeypatch):
    _clear_runtime_env(monkeypatch)
    args = build_parser().parse_args(
        ["--model", "m", "--no-prefix-cache", "--prefix-cache-l2"]
    )
    normalize_cli_args(args)
    assert args.runtime_config.prefix_cache is False
    assert args.runtime_config.prefix_cache_l2 is False


def test_serve_cli_quantize_kv_cache_default_off(monkeypatch):
    _clear_runtime_env(monkeypatch)

    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)

    assert args.runtime_config.quantize_kv_cache is False


def test_serve_cli_quantize_kv_cache_flag(monkeypatch):
    _clear_runtime_env(monkeypatch)

    args = build_parser().parse_args(["--model", "m", "--quantize-kv-cache"])
    normalize_cli_args(args)

    assert args.runtime_config.quantize_kv_cache is True


def test_serve_cli_quantize_kv_cache_env(monkeypatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("DFLASH_QUANTIZE_KV_CACHE", "1")

    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)

    assert args.runtime_config.quantize_kv_cache is True


def test_quantize_kv_cache_rejects_target_fa_window():
    from dataclasses import replace

    from dflash_mlx.runtime.config import (
        DEFAULT_RUNTIME_CONFIG,
        validate_runtime_config,
    )

    cfg = replace(DEFAULT_RUNTIME_CONFIG, quantize_kv_cache=True, target_fa_window=4096)
    with pytest.raises(ValueError, match="target_fa_window"):
        validate_runtime_config(cfg)


def test_offline_surfaces_include_quantize_kv_cache():
    from dflash_mlx.runtime.config import (
        BENCHMARK_RUNTIME_FIELDS,
        GENERATE_RUNTIME_FIELDS,
    )

    assert "quantize_kv_cache" in GENERATE_RUNTIME_FIELDS
    assert "quantize_kv_cache" in BENCHMARK_RUNTIME_FIELDS

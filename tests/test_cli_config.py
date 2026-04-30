# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import importlib
import os
import tomllib
from types import SimpleNamespace
from pathlib import Path

import pytest

import dflash_mlx.metal_limits as metal_limits
from dflash_mlx.server.config import (
    MetalLimitConfig,
    build_parser,
    configure_metal_limits,
    normalize_cli_args,
)

_PROFILE_ENV_KEYS = (
    "DFLASH_RUNTIME_PROFILE",
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
    "DFLASH_VERIFY_MODE",
    "DFLASH_VERIFY_LINEAR",
    "DFLASH_VERIFY_QMM",
    "DFLASH_DIAGNOSTICS_DIR",
)

def _clear_profile_env(monkeypatch):
    for key in _PROFILE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

def test_serve_cli_wires_runtime_env_flags(monkeypatch):
    _clear_profile_env(monkeypatch)

    args = build_parser().parse_args(
        [
            "--model",
            "m",
            "--prefill-step-size",
            "4096",
            "--clear-cache-boundaries",
            "--memory-waterfall",
            "--bench-log-dir",
            "/tmp/dflash-logs",
        ]
    )
    normalize_cli_args(args)

    assert args.runtime_config.prefill_step_size == 4096
    assert args.runtime_config.clear_cache_boundaries is True
    assert args.runtime_config.memory_waterfall is True
    assert args.runtime_config.bench_log_dir == "/tmp/dflash-logs"
    assert args.diagnostics_config.memory_waterfall is True
    assert args.diagnostics_config.trace.log_dir == Path("/tmp/dflash-logs")

def test_serve_cli_prefill_step_size_is_not_decorative(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFILL_STEP_SIZE", "8192")
    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)
    assert args.runtime_config.prefill_step_size == 8192

    args = build_parser().parse_args(["--model", "m", "--prefill-step-size", "2048"])
    normalize_cli_args(args)
    assert args.runtime_config.prefill_step_size == 2048

def test_serve_cli_threads_draft_and_verify_runtime_flags(monkeypatch):
    _clear_profile_env(monkeypatch)
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
    _clear_profile_env(monkeypatch)
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

def test_diagnostics_full_enables_profile_memory_and_cycle_log(monkeypatch, tmp_path):
    _clear_profile_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    args = build_parser().parse_args(["--model", "m", "--diagnostics", "full"])
    normalize_cli_args(args)

    out = Path(args.diagnostics_dir_resolved)
    assert out.parts[:3] == (".artifacts", "dflash", "diagnostics")
    assert out.name.endswith("-serve-full")
    assert args.runtime_config.memory_waterfall is True
    assert args.diagnostics_config.run_dir == out
    assert args.diagnostics_config.memory_waterfall is True
    assert args.diagnostics_config.trace.log_dir == out
    assert args.diagnostics_config.trace.cycle_events is True
    assert (out / "cycle_events.jsonl").exists()

def test_diagnostics_off_creates_no_artifact_dir(monkeypatch, tmp_path):
    _clear_profile_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)

    assert not Path(".artifacts").exists()
    assert not hasattr(args, "diagnostics_dir_resolved")
    assert args.diagnostics_config.mode == "off"
    assert args.diagnostics_config.trace.log_dir is None

def test_diagnostics_dir_override_is_exact(monkeypatch, tmp_path):
    _clear_profile_env(monkeypatch)
    out = tmp_path / "custom-diagnostics"

    args = build_parser().parse_args(
        ["--model", "m", "--diagnostics", "basic", "--diagnostics-dir", str(out)]
    )
    normalize_cli_args(args)

    assert Path(args.diagnostics_dir_resolved) == out
    assert args.diagnostics_config.run_dir == out
    assert args.diagnostics_config.trace.log_dir == out
    assert (out / "manifest.json").exists()

def test_diagnostics_rejects_ambiguous_bench_log_dir(monkeypatch, tmp_path):
    _clear_profile_env(monkeypatch)
    args = build_parser().parse_args(
        [
            "--model",
            "m",
            "--diagnostics",
            "basic",
            "--diagnostics-dir",
            str(tmp_path / "diag"),
            "--bench-log-dir",
            str(tmp_path / "logs"),
        ]
    )
    with pytest.raises(SystemExit, match="diagnostics-dir"):
        normalize_cli_args(args)

def test_diagnostics_rejects_bench_log_dir_without_diagnostics_dir(monkeypatch, tmp_path):
    _clear_profile_env(monkeypatch)
    args = build_parser().parse_args(
        [
            "--model",
            "m",
            "--diagnostics",
            "basic",
            "--bench-log-dir",
            str(tmp_path / "logs"),
        ]
    )
    with pytest.raises(SystemExit, match="diagnostics and --bench-log-dir"):
        normalize_cli_args(args)

def test_diagnostics_manifest_contains_required_fields(monkeypatch, tmp_path):
    _clear_profile_env(monkeypatch)
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

def test_serve_cli_parses_memory_limit_flags():
    args = build_parser().parse_args(
        ["--model", "m", "--wired-limit", "48GB", "--cache-limit", "8GB"]
    )
    assert args.wired_limit == 48 * 1024**3
    assert args.cache_limit == 8 * 1024**3

def test_serve_cli_rejects_invalid_memory_limit():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "m", "--wired-limit", "bad"])

def test_configure_metal_limits_preserves_current_default(monkeypatch):
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

    args = build_parser().parse_args(["--model", "m"])
    limits = configure_metal_limits(args)

    assert calls == [("wired", recommended), ("cache", recommended // 4)]
    assert limits.wired_request == "auto"
    assert limits.wired_bytes == recommended
    assert limits.cache_request == "auto"
    assert limits.cache_bytes == recommended // 4

def test_configure_metal_limits_supports_none_and_explicit_cache(monkeypatch):
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

def test_startup_banner_prints_resolved_metal_limits(monkeypatch, capsys):
    from dflash_mlx import serve

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
        cli_args=SimpleNamespace(
            model="target",
            draft_model=None,
            runtime_config=None,
            metal_limits=limits,
        ),
    )

    serve._print_startup_banner(port=8000, model_provider=provider)

    err = capsys.readouterr().err
    assert "Wired limit: auto -> 64.0 GiB" in err
    assert "Cache limit: 8.0 GiB -> 8.0 GiB" in err

@pytest.mark.parametrize(
    "argv,error",
    [
        (["--model", "m", "--prefill-step-size", "0"], "--prefill-step-size"),
        (["--model", "m", "--draft-sink-size", "-1"], "draft_sink_size"),
        (["--model", "m", "--draft-window-size", "0"], "draft_window_size"),
        (["--model", "m", "--verify-len-cap", "-1"], "verify_len_cap"),
        (["--model", "m", "--bench-log-dir", ""], "--bench-log-dir"),
    ],
)
def test_serve_cli_rejects_invalid_runtime_flags(argv, error):
    args = build_parser().parse_args(argv)
    with pytest.raises(SystemExit, match=error):
        normalize_cli_args(args)

def test_project_scripts_import():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert "dflash-opencode-bench" not in scripts
    assert "dflash-pi-bench" not in scripts
    for target in scripts.values():
        module_name, func_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, func_name))

def test_profile_balanced_sets_product_defaults(monkeypatch):
    _clear_profile_env(monkeypatch)
    args = build_parser().parse_args(["--model", "m"])
    normalize_cli_args(args)
    cfg = args.runtime_config
    assert cfg.profile == "balanced"
    assert cfg.prefill_step_size == 4096
    assert cfg.draft_sink_size == 64
    assert cfg.draft_window_size == 1024
    assert cfg.verify_len_cap == 0
    assert cfg.prefix_cache is True
    assert cfg.prefix_cache_max_entries == 4
    assert cfg.prefix_cache_max_bytes == 8 * 1024**3
    assert cfg.clear_cache_boundaries is False
    assert cfg.max_snapshot_tokens == 24000
    assert cfg.prefix_cache_l2 is False
    assert cfg.verify_mode == "auto"

@pytest.mark.parametrize(
    "profile,prefill,l2",
    [
        ("fast", "8192", "0"),
        ("low-memory", "1024", "0"),
        ("long-session", "4096", "1"),
    ],
)
def test_profiles_set_expected_values(monkeypatch, profile, prefill, l2):
    _clear_profile_env(monkeypatch)
    args = build_parser().parse_args(["--model", "m", "--profile", profile])
    normalize_cli_args(args)
    assert args.runtime_config.profile == profile
    assert args.runtime_config.prefill_step_size == int(prefill)
    assert args.runtime_config.prefix_cache_l2 is (l2 == "1")

def test_cli_explicit_overrides_profile(monkeypatch):
    _clear_profile_env(monkeypatch)
    args = build_parser().parse_args(
        ["--model", "m", "--profile", "fast", "--prefill-step-size", "2048"]
    )
    normalize_cli_args(args)
    assert args.runtime_config.profile == "fast"
    assert args.runtime_config.prefill_step_size == 2048

def test_env_overrides_profile(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFILL_STEP_SIZE", "2048")
    monkeypatch.setenv("DFLASH_DRAFT_WINDOW_SIZE", "512")
    monkeypatch.setenv("DFLASH_VERIFY_LEN_CAP", "4")
    args = build_parser().parse_args(["--model", "m", "--profile", "fast"])
    normalize_cli_args(args)
    assert args.runtime_config.prefill_step_size == 2048
    assert args.runtime_config.draft_window_size == 512
    assert args.runtime_config.verify_len_cap == 4

def test_cli_overrides_env(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFILL_STEP_SIZE", "1024")
    args = build_parser().parse_args(["--model", "m", "--prefill-step-size", "8192"])
    normalize_cli_args(args)
    assert args.runtime_config.prefill_step_size == 8192

def test_verify_mode_off_disables_custom_verify(monkeypatch):
    _clear_profile_env(monkeypatch)
    args = build_parser().parse_args(["--model", "m", "--verify-mode", "off"])
    normalize_cli_args(args)
    assert "DFLASH_VERIFY_LINEAR" not in os.environ
    assert "DFLASH_VERIFY_QMM" not in os.environ
    assert args.runtime_config.verify_mode == "off"
    assert args.runtime_context.verify.mode == "off"

def test_list_profiles_output_stable(capsys):
    args = build_parser().parse_args(["--list-profiles"])
    with pytest.raises(SystemExit) as exc:
        normalize_cli_args(args)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "profile      prefill" in out
    assert "draft_window" in out and "verify_cap" in out
    assert "balanced" in out and "4096" in out and "64+1024" in out and "4x8GiB" in out
    assert "fast" in out and "8192" in out and "4x16GiB" in out
    assert "low-memory" in out and "1024" in out and "2x2GiB" in out
    assert "long-session" in out and "on/50GiB" in out

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
def test_profile_validation_errors(argv, error):
    args = build_parser().parse_args(argv)
    with pytest.raises(SystemExit, match=error):
        normalize_cli_args(args)

def test_target_fa_window_disables_prefix_cache(monkeypatch):
    _clear_profile_env(monkeypatch)
    args = build_parser().parse_args(["--model", "m", "--target-fa-window", "2048"])
    normalize_cli_args(args)
    assert args.runtime_config.target_fa_window == 2048
    assert args.runtime_config.prefix_cache is False
    assert args.runtime_config.prefix_cache_l2 is False

def test_prefix_cache_disabled_disables_l2(monkeypatch):
    _clear_profile_env(monkeypatch)
    args = build_parser().parse_args(
        ["--model", "m", "--no-prefix-cache", "--prefix-cache-l2"]
    )
    normalize_cli_args(args)
    assert args.runtime_config.prefix_cache is False
    assert args.runtime_config.prefix_cache_l2 is False

# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dflash_mlx import doctor
from dflash_mlx.runtime.config import EffectiveRuntimeConfig
from dflash_mlx.runtime.chip_detect import chip_profile_from_device_info
from dflash_mlx.server.config import build_parser as build_server_parser

_DOCTOR_ENV_KEYS = (
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
    "DFLASH_VERIFY_VARIANT",
    "DFLASH_VERIFY_MAX_N",
    "DFLASH_VERIFY_QMM_KPARTS",
    "DFLASH_VERIFY_INCLUDE",
)

def _clear_doctor_env(monkeypatch):
    for key in _DOCTOR_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

def _json_run(argv, capsys):
    code = doctor.run([*argv, "--json"])
    out = capsys.readouterr().out
    return code, json.loads(out)

def test_doctor_json_stable(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    code, report = _json_run([], capsys)
    assert code in (0, 2)
    assert set(report) == {
        "checks",
        "effective_config",
        "model",
        "packages",
        "python",
        "summary",
    }
    assert report["effective_config"]["values"]["prefill_step_size"] == 2048
    assert report["effective_config"]["sources"]["prefill_step_size"] == "default"

def test_doctor_does_not_expose_internal_diagnostics_aliases():
    parser = doctor.build_parser()
    help_text = parser.format_help()

    assert "--memory-waterfall" not in help_text
    assert "--bench-log-dir" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["--bench-log-dir", "/tmp/dflash-logs"])


def test_doctor_and_serve_share_runtime_config_flags():
    runtime_dests = set(EffectiveRuntimeConfig.__dataclass_fields__)

    def runtime_actions(parser):
        return {
            action.dest: tuple(action.option_strings)
            for action in parser._actions
            if action.dest in runtime_dests
        }

    assert runtime_actions(doctor.build_parser()) == runtime_actions(build_server_parser())


def test_doctor_invalid_bool_env_is_fatal(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFIX_CACHE", "maybe")

    code, report = _json_run([], capsys)

    assert code == 1
    assert report["effective_config"]["resolved"] is False
    config_check = next(
        check for check in report["checks"] if check["name"] == "runtime_config"
    )
    assert config_check["status"] == "fatal"
    assert "DFLASH_PREFIX_CACHE" in config_check["details"]["error"]


def test_doctor_model_draft_resolution(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    code, report = _json_run(["--model", "mlx-community/Qwen3.6-27B-4bit"], capsys)
    assert code in (0, 2)
    model_check = next(check for check in report["checks"] if check["name"] == "model")
    assert model_check["status"] == "ok"
    assert model_check["details"]["draft"] == "z-lab/Qwen3.6-27B-DFlash"

def test_doctor_load_model_uses_runtime_bundle(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    calls = []

    import dflash_mlx.runtime.bundle as runtime_bundle

    def fake_load_runtime_bundle(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            target_model=object(),
            tokenizer=object(),
            target_meta={},
            draft_model=object(),
            draft_meta={},
            draft_backend=object(),
            target_ops=object(),
            resolved_model_ref="Qwen/Qwen3.5-9B",
            resolved_draft_ref="manual/draft",
        )

    monkeypatch.setattr(runtime_bundle, "load_runtime_bundle", fake_load_runtime_bundle)

    code, report = _json_run(
        ["--model", "Qwen/Qwen3.5-9B", "--draft", "manual/draft", "--load-model"],
        capsys,
    )

    assert code in (0, 2)
    assert calls
    assert calls[0]["model_ref"] == "Qwen/Qwen3.5-9B"
    assert calls[0]["draft_ref"] == "manual/draft"
    assert calls[0]["verify_config"].mode == "adaptive"
    load_check = next(check for check in report["checks"] if check["name"] == "load_model")
    assert load_check["status"] == "ok"
    assert load_check["message"] == "runtime bundle loads"
    assert load_check["details"] == {
        "model": "Qwen/Qwen3.5-9B",
        "draft": "manual/draft",
    }


def test_doctor_scripts_reject_non_callable_entrypoint(monkeypatch):
    import dflash_mlx.cli as cli

    monkeypatch.setattr(cli, "main", object())

    check = doctor._check_scripts_importable()

    assert check.status == "fatal"
    assert "not callable" in check.details["dflash"]["error"]


def test_doctor_load_model_rejects_incomplete_runtime_bundle(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)

    import dflash_mlx.runtime.bundle as runtime_bundle

    def fake_load_runtime_bundle(**_kwargs):
        return SimpleNamespace(
            resolved_model_ref="Qwen/Qwen3.5-9B",
            resolved_draft_ref="manual/draft",
        )

    monkeypatch.setattr(runtime_bundle, "load_runtime_bundle", fake_load_runtime_bundle)

    code, report = _json_run(
        ["--model", "Qwen/Qwen3.5-9B", "--draft", "manual/draft", "--load-model"],
        capsys,
    )

    assert code == 1
    load_check = next(check for check in report["checks"] if check["name"] == "load_model")
    assert load_check["status"] == "fatal"
    assert "runtime bundle is incomplete" in load_check["details"]["error"]
    assert "target_model" in load_check["details"]["error"]


def test_doctor_warns_on_target_fa_window(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    code, report = _json_run(["--target-fa-window", "2048"], capsys)
    assert code == 2
    warning = next(
        check for check in report["checks"] if check["name"] == "target_fa_window"
    )
    assert warning["status"] == "warning"
    assert report["effective_config"]["values"]["prefix_cache"] is False


def test_doctor_warns_on_old_apple_chip(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    monkeypatch.setattr(
        doctor,
        "detect_chip",
        lambda: chip_profile_from_device_info(
            {"architecture": "applegpu_g13s"},
            metal_available=True,
            macos_version="15.0",
        ),
    )

    code, report = _json_run([], capsys)

    assert code == 2
    bf16 = next(check for check in report["checks"] if check["name"] == "old_apple_bf16")
    nax = next(check for check in report["checks"] if check["name"] == "nax_unavailable")
    assert bf16["status"] == "warning"
    assert bf16["details"]["family"] == "M1"
    assert bf16["details"]["tier"] == "max"
    assert nax["status"] == "warning"


def test_doctor_reports_chip_detection_error(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    monkeypatch.setattr(
        doctor,
        "detect_chip",
        lambda: (_ for _ in ()).throw(RuntimeError("device info failed")),
    )

    code, report = _json_run([], capsys)

    assert code == 2
    check = next(check for check in report["checks"] if check["name"] == "chip_profile")
    assert check["status"] == "warning"
    assert check["details"]["error"] == "device info failed"


def test_doctor_warns_on_internal_verify_env(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")
    code, report = _json_run([], capsys)
    assert code == 2
    verify = next(
        check for check in report["checks"] if check["name"] == "internal_verify_env"
    )
    assert verify["status"] == "warning"
    assert verify["details"]["active"] == {"DFLASH_VERIFY_QMM": "1"}

def test_doctor_strict_promotes_warnings(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")
    code, report = _json_run(["--strict"], capsys)
    assert code == 1
    assert report["summary"]["warnings"] >= 1

def test_doctor_runtime_flag_precedence(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFILL_STEP_SIZE", "2048")
    code, report = _json_run(["--prefill-step-size", "1024"], capsys)
    assert code in (0, 2)
    assert report["effective_config"]["values"]["prefill_step_size"] == 1024
    assert report["effective_config"]["sources"]["prefill_step_size"] == "cli"

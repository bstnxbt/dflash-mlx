# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json

from dflash_mlx import doctor

_DOCTOR_ENV_KEYS = (
    "DFLASH_RUNTIME_PROFILE",
    "DFLASH_PREFILL_STEP_SIZE",
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
    assert report["effective_config"]["values"]["profile"] == "balanced"
    assert report["effective_config"]["sources"]["profile"] == "default"

def test_doctor_model_draft_resolution(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    code, report = _json_run(["--model", "mlx-community/Qwen3.6-27B-4bit"], capsys)
    assert code in (0, 2)
    model_check = next(check for check in report["checks"] if check["name"] == "model")
    assert model_check["status"] == "ok"
    assert model_check["details"]["draft"] == "z-lab/Qwen3.6-27B-DFlash"

def test_doctor_warns_on_target_fa_window(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    code, report = _json_run(["--target-fa-window", "2048"], capsys)
    assert code == 2
    warning = next(
        check for check in report["checks"] if check["name"] == "target_fa_window"
    )
    assert warning["status"] == "warning"
    assert report["effective_config"]["values"]["prefix_cache"] is False

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

def test_doctor_profile_precedence(monkeypatch, capsys):
    _clear_doctor_env(monkeypatch)
    monkeypatch.setenv("DFLASH_PREFILL_STEP_SIZE", "2048")
    code, report = _json_run(
        ["--profile", "fast", "--prefill-step-size", "1024"], capsys
    )
    assert code in (0, 2)
    assert report["effective_config"]["values"]["prefill_step_size"] == 1024
    assert report["effective_config"]["sources"]["prefill_step_size"] == "cli"

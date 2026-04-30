# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

from dflash_mlx import cli

def test_root_help_contains_product_commands(capsys):
    assert cli.run(["--help"]) == 0
    out = capsys.readouterr().out
    assert "serve" in out
    assert "generate" in out
    assert "benchmark" in out
    assert "doctor" in out
    assert "profiles" in out

def test_profiles_command_lists_profiles(capsys):
    assert cli.run(["profiles"]) == 0
    out = capsys.readouterr().out
    assert "profile      prefill" in out
    assert "balanced" in out and "4096" in out
    assert "fast" in out and "8192" in out
    assert "low-memory" in out and "1024" in out
    assert "long-session" in out and "on/50GiB" in out

def test_models_command_lists_draft_registry(capsys):
    assert cli.run(["models"]) == 0
    out = capsys.readouterr().out
    assert "Qwen3.6-27B" in out
    assert "z-lab/Qwen3.6-27B-DFlash" in out

@pytest.mark.parametrize(
    "argv,module_name,prog,forwarded",
    [
        (["serve", "--model", "m"], "dflash_mlx.serve", "dflash serve", ["--model", "m"]),
        (
            ["generate", "--model", "m", "--prompt", "p"],
            "dflash_mlx.generate",
            "dflash generate",
            ["--model", "m", "--prompt", "p"],
        ),
        (
            ["doctor", "--json"],
            "dflash_mlx.doctor",
            "dflash doctor",
            ["--json"],
        ),
        (
            ["benchmark", "--prompt", "p"],
            "dflash_mlx.benchmark",
            "dflash benchmark",
            ["--prompt", "p"],
        ),
    ],
)
def test_root_dispatch(monkeypatch, argv, module_name, prog, forwarded):
    calls = []

    def fake_run_module_main(seen_module, seen_prog, seen_args):
        calls.append((seen_module, seen_prog, list(seen_args)))
        return 0

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)
    assert cli.run(argv) == 0
    assert calls == [(module_name, prog, forwarded)]

def test_root_rejects_missing_subcommand_generate_form(capsys):
    assert cli.run(["--model", "m", "--prompt", "p"]) == 2
    err = capsys.readouterr().err
    assert "dflash generate" in err

def test_pyproject_scripts_import():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["dflash"] == "dflash_mlx.cli:main"
    assert set(scripts) == {"dflash"}
    for target in scripts.values():
        module_name, func_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, func_name))

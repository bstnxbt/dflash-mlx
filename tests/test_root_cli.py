# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dflash_mlx import cli

def test_models_command_lists_draft_registry(capsys):
    assert cli.run(["models"]) == 0
    out = capsys.readouterr().out
    assert "Qwen3.6-27B" in out
    assert "z-lab/Qwen3.6-27B-DFlash" in out
    assert "Qwen3.8-27B" in out
    assert "z-lab/Qwen3.8-27B-DFlash2" in out
    assert "gemma-4-31b-it" in out
    assert "z-lab/gemma-4-31B-it-DFlash" in out
    assert "gemma-4-26b-a4b-it" in out
    assert "z-lab/gemma-4-26B-A4B-it-DFlash" in out

def test_root_dispatch(monkeypatch):
    calls = []

    def fake_run_module_main(seen_module, seen_prog, seen_args):
        calls.append((seen_module, seen_prog, list(seen_args)))
        return 0

    monkeypatch.setattr(cli, "_run_module_main", fake_run_module_main)
    cases = [
        (["serve", "--model", "m"], "dflash_mlx.serve", "dflash serve", ["--model", "m"]),
        (
            ["generate", "--model", "m", "--prompt", "p"],
            "dflash_mlx.generate",
            "dflash generate",
            ["--model", "m", "--prompt", "p"],
        ),
        (["doctor", "--json"], "dflash_mlx.doctor", "dflash doctor", ["--json"]),
        (
            ["benchmark", "--prompt", "p"],
            "dflash_mlx.benchmark",
            "dflash benchmark",
            ["--prompt", "p"],
        ),
    ]
    for argv, module_name, prog, forwarded in cases:
        assert cli.run(argv) == 0
        assert calls[-1] == (module_name, prog, forwarded)

def test_root_rejects_missing_subcommand_generate_form(capsys):
    assert cli.run(["--model", "m", "--prompt", "p"]) == 2
    err = capsys.readouterr().err
    assert "dflash generate" in err

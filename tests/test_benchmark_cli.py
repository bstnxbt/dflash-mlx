# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from pathlib import Path

import pytest

from dflash_mlx import benchmark
from dflash_mlx.runtime_context import build_offline_runtime_context

def test_benchmark_help_documents_public_flags(capsys):
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    expected = [
        "--max-tokens INT",
        "Number of tokens to generate. Default: 64.",
        "--block-tokens INT",
        "DFlash speculative verify block size. Default: 16.",
        "--ctx INT",
        "Build an approximate long-context prompt of INT tokens. Default: 0.",
        "--no-memory",
        "Omit peak memory medians from the summary. Default: memory summary enabled.",
        "--repeat INT",
        "Number of measured runs. Default: 1.",
        "--cooldown SECONDS",
        "Sleep between measured runs. Default: 10.",
        "--model HF_REF_OR_PATH",
        "Target model. Default: auto-resolved default target.",
        "--draft HF_REF_OR_PATH",
        "DFlash draft model. Default: auto-resolved from target.",
        "--no-chat-template",
        "Default: chat template enabled.",
        "--quantize-draft",
        "Default: disabled.",
        "--no-eos",
        "Default: EOS enabled.",
        "--split-sdpa",
        "--no-split-sdpa",
        "--target-fa-window INT",
        "Default: 0 = full KV.",
        "--draft-sink-size INT",
        "Default: 64.",
        "--draft-window-size INT",
        "Default: 1024.",
        "--verify-len-cap INT",
        "Default: 0 = block size.",
        "--out PATH",
        ".artifacts/dflash/benchmarks/<timestamp>-<mode>-<model>",
    ]
    for text in expected:
        assert text in out
    assert "--matrix" not in out
    assert "--memory" not in out
    assert "--agentic" not in out

@pytest.mark.parametrize("flag", ["--matrix", "--memory", "--agentic"])
def test_benchmark_rejects_removed_public_flags(flag):
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([flag])

    assert exc.value.code == 2

def test_benchmark_invocation_records_explicit_and_effective_values():
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--prompt",
            "p",
            "--model",
            "target-alias",
            "--draft",
            "draft-alias",
            "--max-tokens",
            "8",
            "--ctx",
            "65536",
            "--no-memory",
            "--out",
            "/tmp/result.json",
            "--no-chat-template",
            "--no-split-sdpa",
            "--draft-sink-size",
            "32",
            "--draft-window-size",
            "512",
            "--verify-len-cap",
            "8",
        ]
    )
    args.repeat = 1
    config = {
        "model": "resolved-target",
        "draft": "resolved-draft",
    }
    invocation = benchmark._build_invocation(
        args,
        Path("/tmp/result.json"),
        [
            "dflash benchmark",
            "--prompt",
            "p",
            "--model",
            "target-alias",
            "--draft",
            "draft-alias",
            "--max-tokens",
            "8",
            "--ctx",
            "65536",
            "--no-memory",
            "--out",
            "/tmp/result.json",
            "--no-chat-template",
            "--no-split-sdpa",
            "--draft-sink-size",
            "32",
            "--draft-window-size",
            "512",
            "--verify-len-cap",
            "8",
        ],
        config,
    )

    assert invocation["output_path"] == "/tmp/result.json"
    assert invocation["output_dir"] == "/tmp/result.json"
    assert invocation["command"].startswith("dflash benchmark --prompt p")
    assert invocation["protocol_order"] == ["baseline", "dflash"]
    assert invocation["same_prompt_token_ids"] is True
    assert invocation["primary_metric"] == "post_prefill_generation_tps"
    assert invocation["explicit_flags"]["model"] == "resolved-target"
    assert invocation["explicit_flags"]["draft"] == "resolved-draft"
    assert invocation["explicit_flags"]["max_tokens"] == 8
    assert invocation["explicit_flags"]["ctx"] == 65536
    assert invocation["explicit_flags"]["no_memory"] is True
    assert invocation["explicit_flags"]["out"] == "/tmp/result.json"
    assert invocation["explicit_flags"]["draft_sink_size"] == 32
    assert invocation["explicit_flags"]["draft_window_size"] == 512
    assert invocation["explicit_flags"]["verify_len_cap"] == 8
    assert invocation["effective"]["model"] == "resolved-target"
    assert invocation["effective"]["draft"] == "resolved-draft"
    assert invocation["effective"]["include_memory"] is False
    assert invocation["effective"]["use_chat_template"] is False
    assert invocation["effective"]["split_sdpa"] is False

def test_benchmark_default_output_dir_is_artifact_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    parser = benchmark.build_parser()
    args = parser.parse_args(["--model", "mlx-community/Qwen3.6-27B-4bit"])
    args.repeat = 1

    out = benchmark.create_run_dir("benchmark", benchmark._benchmark_label(args))

    assert out.parts[:3] == (".artifacts", "dflash", "benchmarks")
    assert "benchmark-results" not in str(out).replace("/", "-")
    assert not str(out).startswith("/tmp")

def test_benchmark_runtime_context_uses_product_verify_config():
    context = build_offline_runtime_context(
        target_fa_window=2048,
        draft_sink_size=32,
        draft_window_size=512,
        verify_len_cap=8,
    )

    assert context.runtime.target_fa_window == 2048
    assert context.runtime.draft_sink_size == 32
    assert context.runtime.draft_window_size == 512
    assert context.runtime.verify_len_cap == 8
    assert context.runtime.prefix_cache is False
    assert context.verify.mode == "auto"
    assert context.verify.enable_qmm is True

def test_benchmark_runtime_context_is_required():
    stream = benchmark.stream_dflash_generate()
    with pytest.raises(ValueError, match="runtime_context is required"):
        next(stream)

def test_benchmark_help_has_no_legacy_default_paths(capsys):
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "benchmark/results" not in out
    assert "/tmp" not in out

def test_public_docs_do_not_use_internal_benchmark_modules_as_normal_path():
    for path in (
        Path("README.md"),
        Path("docs/cli.md"),
        Path("docs/benchmarking.md"),
    ):
        text = path.read_text()
        assert "python -m tools.benchmarks" not in text
        assert "bash tools/benchmarks" not in text
        assert "benchmark/results/<" not in text

def test_public_docs_mention_artifact_policy_and_public_commands():
    docs = "\n".join(
        Path(path).read_text()
        for path in ("docs/cli.md", "docs/benchmarking.md", "docs/observability.md")
    )
    assert "dflash serve --diagnostics basic" in docs
    assert "dflash serve --diagnostics full" in docs
    assert "dflash benchmark --model" in docs
    assert ".artifacts/dflash/diagnostics" in docs
    assert ".artifacts/dflash/benchmarks" in docs

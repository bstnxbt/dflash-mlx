# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import replace
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from dflash_mlx import benchmark
from dflash_mlx import benchmark_report
from dflash_mlx import benchmark_suites
import dflash_mlx.runtime.config as runtime_config
from dflash_mlx.engine.events import (
    CycleCompleteEvent,
    PrefillCompleteEvent,
    SummaryEvent,
    TokenEvent,
)
from dflash_mlx.metal_limits import MetalLimitConfig
from dflash_mlx.runtime.context import build_offline_runtime_context

def test_prompt_slug_distinguishes_long_prompts_with_same_prefix():
    prefix = "same prefix " * 8
    first = benchmark_suites.slugify_prompt_id(prefix + "alpha")
    second = benchmark_suites.slugify_prompt_id(prefix + "beta")

    assert first != second
    assert first.startswith("same-prefix")
    assert second.startswith("same-prefix")

def test_benchmark_parser_exposes_public_flags(capsys):
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {
        "--suite",
        "--limit",
        "--ctx-tokens",
        "--prompt-file",
        "--shuffle",
        "--seed",
        "--prompt",
        "--max-tokens",
        "--block-tokens",
        "--no-memory",
        "--repeat",
        "--cooldown",
        "--wired-limit",
        "--cache-limit",
        "--model",
        "--draft",
        "--no-chat-template",
        "--draft-quant",
        "--no-eos",
        "--target-fa-window",
        "--draft-sink-size",
        "--draft-window-size",
        "--verify-len-cap",
        "--verify-mode",
        "--only-dflash",
        "--out",
    } <= option_strings
    assert "--matrix" not in out
    assert "--memory" not in out
    assert "--agentic" not in out

def test_benchmark_rejects_removed_public_flags():
    parser = benchmark.build_parser()
    for flag in ("--matrix", "--memory", "--agentic"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([flag])

        assert exc.value.code == 2

def test_benchmark_finalize_uses_offline_runtime_validation():
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "m",
            "--draft-window-size",
            "0",
        ]
    )

    with pytest.raises(
        ValueError,
        match="--draft-window-size / draft_window_size must be > 0",
    ):
        benchmark._finalize_benchmark_args(args, ["--model", "m"])

def test_benchmark_cli_reports_offline_runtime_validation_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(
            [
                "--model",
                "m",
                "--draft-window-size",
                "0",
            ],
            prog="dflash benchmark",
        )

    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "--draft-window-size must be > 0" in err
    assert "draft_window_size" not in err
    assert "Traceback" not in err


def test_benchmark_cli_requires_model_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--max-tokens", "1"], prog="dflash benchmark")

    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "--model" in err
    assert "Traceback" not in err


def test_benchmark_main_applies_requested_metal_limits(monkeypatch, tmp_path):
    calls: list[dict] = []

    def fake_apply_metal_limits(**kwargs):
        calls.append(kwargs)
        return {
            "metal_available": True,
            "wired_request": kwargs["wired_request"],
            "cache_request": kwargs["cache_request"],
        }

    def fake_suite(*, prompts, args):
        assert prompts[0].id == "smoke-default"
        assert args.metal_limits["cache_request"] == 4 * 1024**3
        return {
            "hardware": {},
            "config": {
                "suite": args.suite,
                "model": args.model,
                "draft": "auto-draft",
            },
            "prompts": [],
            "runs": [],
            "summary": {},
        }

    monkeypatch.setattr(benchmark, "apply_metal_limits", fake_apply_metal_limits)
    monkeypatch.setattr(benchmark, "benchmark_suite", fake_suite)

    benchmark.main(
        [
            "--model",
            "m",
            "--wired-limit",
            "48GB",
            "--cache-limit",
            "4GB",
            "--out",
            str(tmp_path / "bench"),
        ],
        prog="dflash benchmark",
    )

    assert calls == [
        {
            "wired_request": 48 * 1024**3,
            "cache_request": 4 * 1024**3,
        }
    ]
    assert (tmp_path / "bench" / "invocation.json").exists()


def test_benchmark_invocation_records_explicit_and_effective_values():
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--suite",
            "longctx",
            "--limit",
            "1",
            "--prompt",
            "p",
            "--cooldown",
            "120",
            "--wired-limit",
            "48GB",
            "--cache-limit",
            "4GB",
            "--model",
            "target-alias",
            "--draft",
            "draft-alias",
            "--max-tokens",
            "8",
            "--ctx-tokens",
            "65536",
            "--no-memory",
            "--out",
            "/tmp/result.json",
            "--no-chat-template",
            "--draft-sink-size",
            "32",
            "--draft-window-size",
            "512",
            "--verify-len-cap",
            "8",
            "--verify-mode",
            "ddtree",
        ]
    )
    args = benchmark._finalize_benchmark_args(
        args,
        ["--suite", "longctx", "--limit", "1", "--ctx-tokens", "65536"],
    )
    config = {
        "model": "resolved-target",
        "draft": "resolved-draft",
        "draft_quant": "w4",
    }
    invocation = benchmark._build_invocation(
        args,
        Path("/tmp/result.json"),
        [
            "dflash benchmark",
            "--suite",
            "longctx",
            "--limit",
            "1",
            "--prompt",
            "p",
            "--cooldown",
            "120",
            "--wired-limit",
            "48GB",
            "--cache-limit",
            "4GB",
            "--model",
            "target-alias",
            "--draft",
            "draft-alias",
            "--max-tokens",
            "8",
            "--ctx-tokens",
            "65536",
            "--no-memory",
            "--out",
            "/tmp/result.json",
            "--no-chat-template",
            "--draft-sink-size",
            "32",
            "--draft-window-size",
            "512",
            "--verify-len-cap",
            "8",
            "--verify-mode",
            "ddtree",
        ],
        config,
    )

    assert invocation["output_path"] == "/tmp/result.json"
    assert invocation["output_dir"] == "/tmp/result.json"
    assert invocation["command"].startswith("dflash benchmark --suite longctx")
    assert invocation["protocol_order"] == ["baseline", "dflash"]
    assert invocation["same_prompt_token_ids"] is True
    assert invocation["primary_metric"] == "post_prefill_generation_tps"
    assert invocation["explicit_flags"]["model"] == "resolved-target"
    assert invocation["explicit_flags"]["draft"] == "resolved-draft"
    assert invocation["explicit_flags"]["suite"] == "longctx"
    assert invocation["explicit_flags"]["limit"] == 1
    assert invocation["explicit_flags"]["max_tokens"] == 8
    assert invocation["explicit_flags"]["ctx_tokens"] == 65536
    assert invocation["explicit_flags"]["no_memory"] is True
    assert invocation["explicit_flags"]["cooldown"] == 120
    assert invocation["explicit_flags"]["wired_limit"] == 48 * 1024**3
    assert invocation["explicit_flags"]["cache_limit"] == 4 * 1024**3
    assert invocation["explicit_flags"]["out"] == "/tmp/result.json"
    assert invocation["explicit_flags"]["draft_sink_size"] == 32
    assert invocation["explicit_flags"]["draft_window_size"] == 512
    assert invocation["explicit_flags"]["verify_len_cap"] == 8
    assert invocation["explicit_flags"]["verify_mode"] == "ddtree"
    assert invocation["effective"]["model"] == "resolved-target"
    assert invocation["effective"]["draft"] == "resolved-draft"
    assert invocation["effective"]["suite"] == "longctx"
    assert invocation["effective"]["ctx_tokens"] == 65536
    assert invocation["effective"]["shuffle"] is False
    assert invocation["effective"]["seed"] == 0
    assert invocation["effective"]["include_memory"] is False
    assert invocation["effective"]["wired_limit"] == 48 * 1024**3
    assert invocation["effective"]["cache_limit"] == 4 * 1024**3
    assert invocation["effective"]["use_chat_template"] is False
    assert invocation["effective"]["draft_quant"] == "w4"

def test_benchmark_default_output_dir_is_artifact_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--model", "mlx-community/Qwen3.6-27B-4bit"]),
        ["--model", "mlx-community/Qwen3.6-27B-4bit"],
    )

    out = benchmark.create_run_dir("benchmark", benchmark._benchmark_label(args))

    assert out.parts[:3] == (".artifacts", "dflash", "benchmarks")
    assert "benchmark-results" not in str(out).replace("/", "-")
    assert not str(out).startswith("/tmp")

def test_benchmark_default_suite_is_smoke():
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(parser.parse_args([]), [])
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert args.suite == "smoke"
    assert args.limit == 1
    assert args.cache_limit == "auto"
    assert [prompt.id for prompt in prompts] == ["smoke-default"]

def test_benchmark_longctx_defaults_to_bounded_cache_limit():
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "longctx", "--model", "m"]),
        ["--suite", "longctx", "--model", "m"],
    )

    assert args.cache_limit == 4 * 1024**3

def test_benchmark_explicit_auto_cache_limit_overrides_longctx_default():
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(
            ["--suite", "longctx", "--model", "m", "--cache-limit", "auto"]
        ),
        ["--suite", "longctx", "--model", "m", "--cache-limit", "auto"],
    )

    assert args.cache_limit == "auto"

def _fake_hf_rows(suite: str, count: int = 5) -> list[dict[str, str]]:
    if suite == "humaneval":
        return [
            {"task_id": f"HumanEval/{idx}", "prompt": f"def f_{idx}():\n    pass\n"}
            for idx in range(count)
        ]
    if suite == "gsm8k":
        return [{"question": f"question {idx}"} for idx in range(count)]
    if suite == "aime25":
        return [
            {"id": str(idx), "problem": f"aime problem {idx}", "answer": str(idx)}
            for idx in range(count)
        ]
    return [
        {"problem_id": f"problem-{idx}", "problem": f"problem text {idx}"}
        for idx in range(count)
    ]

@pytest.mark.parametrize(
    ("suite", "expected_call", "expected_ids"),
    [
        (
            "humaneval",
            ("openai_humaneval", None, "test"),
            ["humaneval:HumanEval/0", "humaneval:HumanEval/1", "humaneval:HumanEval/2"],
        ),
        (
            "gsm8k",
            ("gsm8k", "main", "test"),
            ["gsm8k:0", "gsm8k:1", "gsm8k:2"],
        ),
        (
            "math500",
            ("HuggingFaceH4/MATH-500", None, "test"),
            ["math500:problem-0", "math500:problem-1", "math500:problem-2"],
        ),
        (
            "aime25",
            ("math-ai/aime25", None, "test"),
            ["aime25:0", "aime25:1", "aime25:2"],
        ),
    ],
)
def test_benchmark_named_suite_loads_hf_dataset(monkeypatch, suite, expected_call, expected_ids):
    calls = []

    def fake_load(name, config, split):
        calls.append((name, config, split))
        return _fake_hf_rows(suite)

    monkeypatch.setattr(benchmark_suites, "load_hf_dataset", fake_load)
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", suite, "--limit", "3"]),
        ["--suite", suite, "--limit", "3"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert calls == [expected_call]
    assert len(prompts) == 3
    assert [prompt.suite for prompt in prompts] == [suite, suite, suite]
    assert [prompt.source for prompt in prompts] == ["hf", "hf", "hf"]
    assert [prompt.id for prompt in prompts] == expected_ids

def test_benchmark_hf_default_selection_is_not_shuffled(monkeypatch):
    monkeypatch.setattr(
        benchmark_suites,
        "load_hf_dataset",
        lambda name, config, split: _fake_hf_rows("gsm8k", count=5),
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--limit", "3"]),
        ["--suite", "gsm8k", "--limit", "3"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert [prompt.id for prompt in prompts] == ["gsm8k:0", "gsm8k:1", "gsm8k:2"]

def test_benchmark_hf_shuffle_is_explicit_and_seeded(monkeypatch):
    monkeypatch.setattr(
        benchmark_suites,
        "load_hf_dataset",
        lambda name, config, split: _fake_hf_rows("gsm8k", count=6),
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--limit", "3", "--shuffle", "--seed", "7"]),
        ["--suite", "gsm8k", "--limit", "3", "--shuffle", "--seed", "7"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)
    expected_indices = list(range(6))
    random.Random(7).shuffle(expected_indices)

    assert [prompt.id for prompt in prompts] == [
        f"gsm8k:{idx}" for idx in expected_indices[:3]
    ]

def test_benchmark_missing_datasets_has_clean_error(monkeypatch):
    def raise_import_error():
        raise ImportError("no datasets")

    monkeypatch.setattr(benchmark_suites, "datasets_load_dataset", raise_import_error)
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--limit", "1"]),
        ["--suite", "gsm8k", "--limit", "1"],
    )

    with pytest.raises(RuntimeError, match="Install datasets to use --suite humaneval/gsm8k/math500/aime25"):
        benchmark_suites.resolve_benchmark_prompts(args)

def test_benchmark_aime25_prompt_keeps_expected_answer(monkeypatch):
    monkeypatch.setattr(
        benchmark_suites,
        "load_hf_dataset",
        lambda name, config, split: _fake_hf_rows("aime25", count=2),
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "aime25", "--limit", "1"]),
        ["--suite", "aime25", "--limit", "1"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert args.limit == 1
    assert prompts[0].id == "aime25:0"
    assert prompts[0].expected_answer == "0"
    assert prompts[0].prompt.startswith("Solve this AIME 2025 problem.")
    assert "aime problem 0" in prompts[0].prompt
    assert "\\boxed{}" in prompts[0].prompt

def test_benchmark_aime25_defaults_to_full_generation_budget():
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "aime25", "--limit", "1"]),
        ["--suite", "aime25", "--limit", "1"],
    )
    explicit = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "aime25", "--limit", "1", "--max-tokens", "128"]),
        ["--suite", "aime25", "--limit", "1", "--max-tokens", "128"],
    )
    smoke = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "smoke"]),
        ["--suite", "smoke"],
    )

    assert args.max_tokens == 65536
    assert explicit.max_tokens == 128
    assert smoke.max_tokens == 64

def test_benchmark_local_suites_do_not_call_hf(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("HF loader should not be called")

    monkeypatch.setattr(benchmark_suites, "load_hf_dataset", fail)
    parser = benchmark.build_parser()
    smoke = benchmark._finalize_benchmark_args(parser.parse_args(["--suite", "smoke"]), ["--suite", "smoke"])
    longctx = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "longctx", "--ctx-tokens", "1024"]),
        ["--suite", "longctx", "--ctx-tokens", "1024"],
    )

    assert benchmark_suites.resolve_benchmark_prompts(smoke)[0].source == "smoke"
    assert benchmark_suites.resolve_benchmark_prompts(longctx)[0].source == "synthetic"

def test_benchmark_longctx_resolves_configured_context():
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "longctx", "--ctx-tokens", "65536"]),
        ["--suite", "longctx", "--ctx-tokens", "65536"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert args.ctx_tokens == 65536
    assert args.limit == 1
    assert prompts[0].id == "longctx-65536"
    assert len(prompts[0].prompt) > len(benchmark_suites.DEFAULT_PROMPT)

def test_benchmark_prompt_file_loads_jsonl_deterministically(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"id":"b","suite":"custom","prompt":"second"}\n'
        '{"id":"a","suite":"custom","prompt":"first"}\n'
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--prompt-file", str(prompt_file), "--limit", "1"]),
        ["--suite", "gsm8k", "--prompt-file", str(prompt_file), "--limit", "1"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert [prompt.id for prompt in prompts] == ["b"]
    assert prompts[0].prompt == "second"
    assert prompts[0].source == "jsonl"

def test_benchmark_prompt_file_bypasses_hf(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise AssertionError("HF loader should not be called")

    monkeypatch.setattr(benchmark_suites, "load_hf_dataset", fail)
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"id":"local","suite":"gsm8k","prompt":"local prompt"}\n')
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--prompt-file", str(prompt_file)]),
        ["--suite", "gsm8k", "--prompt-file", str(prompt_file)],
    )

    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert [prompt.id for prompt in prompts] == ["local"]

def test_benchmark_summary_markdown_handles_missing_metrics():
    result = {
        "config": {
            "suite": "smoke",
            "benchmark_mode": "smoke",
            "model": "target",
            "draft": "draft",
            "max_tokens": 8,
            "block_tokens": 16,
            "prompt_count": 1,
            "draft_quant": "w4",
            "prompt_tokenization_mode": "chat_template",
        },
        "summary": {
            "prompt_tok_avg": None,
            "baseline_tps_median": None,
            "dflash_tps_median": None,
            "speedup_median": None,
            "dflash_ttft_ms_median": None,
            "dflash_peak_memory_gb_median": None,
            "acceptance_ratio_median": None,
            "prefix_saved_tokens": None,
        },
        "prompts": [],
    }

    text = benchmark_report.summary_markdown(result)

    assert (
        "| smoke | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | "
        "n/a | n/a | n/a |"
    ) in text
    assert "- draft_quant: w4" in text
    assert "- verify_mode: None" in text


def test_benchmark_uses_effective_runtime_default_draft_quant_label():
    assert (
        benchmark._effective_draft_quant_label(
            None,
            {"draft_quant_spec": "w4"},
        )
        == "w4"
    )
    assert (
        benchmark._effective_draft_quant_label(
            "none",
            {"draft_quant_spec": None},
        )
        == "none"
    )


def test_benchmark_manifest_config_fields_are_suite_aware(monkeypatch):
    monkeypatch.setattr(
        benchmark_suites,
        "load_hf_dataset",
        lambda name, config, split: _fake_hf_rows("humaneval", count=3),
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(
            [
                "--suite",
                "humaneval",
                "--limit",
                "2",
                "--model",
                "m",
                "--shuffle",
                "--seed",
                "9",
                "--cache-limit",
                "4GB",
            ]
        ),
        [
            "--suite",
            "humaneval",
            "--limit",
            "2",
            "--model",
            "m",
            "--shuffle",
            "--seed",
            "9",
            "--cache-limit",
            "4GB",
        ],
    )
    args.metal_limits = MetalLimitConfig(
        metal_available=True,
        recommended_bytes=64 * 1024**3,
        wired_request="auto",
        wired_bytes=64 * 1024**3,
        wired_applied=True,
        cache_request=4 * 1024**3,
        cache_bytes=4 * 1024**3,
        cache_applied=True,
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)
    result = benchmark_report.suite_report(
        prompts=prompts,
        prompt_reports=[
            {
                "hardware": {},
                "config": {
                    "model": "m",
                    "draft": "d",
                    "prompt_id": prompts[0].id,
                    "prompt_suite": prompts[0].suite,
                    "prompt_tokens": 10,
                },
                "runs": [],
                "summary": {},
            },
            {
                "hardware": {},
                "config": {
                    "model": "m",
                    "draft": "d",
                    "prompt_id": prompts[1].id,
                    "prompt_suite": prompts[1].suite,
                    "prompt_tokens": 12,
                },
                "runs": [],
                "summary": {},
            },
        ],
        args=args,
        include_memory=True,
    )

    assert result["config"]["suite"] == "humaneval"
    assert result["config"]["prompt_source"] == "hf"
    assert result["config"]["hf_dataset_name"] == "openai_humaneval"
    assert result["config"]["hf_dataset_config"] is None
    assert result["config"]["hf_dataset_split"] == "test"
    assert result["config"]["shuffle"] is True
    assert result["config"]["seed"] == 9
    assert result["config"]["hf_shuffle_seed"] == 9
    assert result["config"]["prompt_ids"] == [prompt.id for prompt in prompts]
    assert result["config"]["selected_row_indices"] == [prompt.row_index for prompt in prompts]
    assert result["config"]["prompt_count"] == 2
    assert result["config"]["ctx_tokens"] == 0
    assert result["config"]["cache_limit"] == 4 * 1024**3
    assert result["config"]["metal_limits"]["cache_bytes"] == 4 * 1024**3
    assert result["runs"] == []

def test_benchmark_runs_json_rows_keep_provenance():
    prompts = [
        benchmark_suites.BenchmarkPrompt("p0", "smoke", "prompt", "smoke"),
    ]
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(
            [
                "--suite",
                "smoke",
                "--model",
                "m",
                "--draft",
                "d",
                "--draft-quant",
                "w4",
                "--verify-mode",
                "ddtree",
            ]
        ),
        [
            "--suite",
            "smoke",
            "--model",
            "m",
            "--draft",
            "d",
            "--draft-quant",
            "w4",
            "--verify-mode",
            "ddtree",
        ],
    )
    result = benchmark_report.suite_report(
        prompts=prompts,
        prompt_reports=[
            {
                "hardware": {},
                "config": {
                    "model": "m",
                    "draft": "d",
                    "git_hash": "abc123",
                    "prompt_id": "p0",
                    "prompt_suite": "smoke",
                    "prompt_tokens": 8,
                },
                "runs": [
                    {
                        "run": 1,
                        "baseline": {"generation_tps": 1.0},
                        "dflash": {"generation_tps": 2.0},
                        "speedup": 2.0,
                    }
                ],
                "summary": {},
            }
        ],
        args=args,
        include_memory=True,
    )

    row = result["runs"][0]
    assert row["suite"] == "smoke"
    assert row["prompt_id"] == "p0"
    assert row["model"] == "m"
    assert row["draft"] == "d"
    assert row["draft_quant"] == "w4"
    assert row["git_hash"] == "abc123"
    assert row["prompt_tokenization_mode"] == "chat_template"
    assert row["max_tokens"] == 64
    assert row["verify_mode"] == "ddtree"


def test_benchmark_runtime_context_uses_product_verify_config():
    context = build_offline_runtime_context(
        prefill_step_size=2048,
        target_fa_window=2048,
        draft_sink_size=32,
        draft_window_size=512,
        verify_len_cap=8,
        verify_mode="adaptive",
    )

    assert context.runtime.prefill_step_size == 2048
    assert context.runtime.target_fa_window == 2048
    assert context.runtime.draft_sink_size == 32
    assert context.runtime.draft_window_size == 512
    assert context.runtime.verify_len_cap == 8
    assert context.runtime.prefix_cache is False
    assert context.verify.mode == "adaptive"
    assert context.verify.enable_qmm is True


def test_benchmark_prefill_step_size_cli_reaches_invocation(tmp_path):
    parser = benchmark.build_parser()
    argv = ["dflash benchmark", "--model", "m", "--prefill-step-size", "2048"]
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(argv[1:]),
        argv[1:],
    )

    invocation = benchmark._build_invocation(args, tmp_path, argv, {})

    assert args.prefill_step_size == 2048
    assert invocation["explicit_flags"]["prefill_step_size"] == 2048
    assert invocation["effective"]["prefill_step_size"] == 2048

def test_benchmark_direct_defaults_follow_runtime_default(monkeypatch):
    monkeypatch.setattr(
        runtime_config,
        "DEFAULT_RUNTIME_CONFIG",
        replace(runtime_config.DEFAULT_RUNTIME_CONFIG, draft_window_size=1536),
    )

    values = benchmark._offline_runtime_values()

    assert values["draft_window_size"] == 1536

class _BenchmarkTokenizer:
    eos_token_ids = []
    eos_token_id = None

    def encode(self, prompt):
        return [1, 2, 3]

    def decode(self, tokens):
        if isinstance(tokens, int):
            return f"T{tokens}"
        return "".join(f"T{token}" for token in tokens)


class _ClosableBenchmarkStream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


class _BindableBenchmarkDraft:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.bound = False
        self.bound_target = None
        self.bound_ops = None

    def bind_target_model(self, target_model, *, target_ops):
        if self.events is not None:
            self.events.append("bind")
        self.bound = True
        self.bound_target = target_model
        self.bound_ops = target_ops


def _fake_baseline_result() -> dict:
    return {
        "elapsed_us": 2_000_000.0,
        "prefill_us": 1_000_000.0,
        "prefill_tok_s": 3.0,
        "prompt_token_count": 3,
        "generated_token_ids": [7],
        "generation_tokens": 1,
        "generation_tps": 1.0,
        "peak_memory_gb": 1.0,
        "memory_end": {"phys_footprint_gb": 2.0, "mlx_cache_gb": 0.5},
    }


def _fake_dflash_result() -> dict:
    return {
        "elapsed_us": 1_000_000.0,
        "phase_timings_us": {"prefill": 100_000.0, "verify": 50_000.0},
        "ttft_us": 100_000.0,
        "prefill_us": 100_000.0,
        "prompt_token_count": 3,
        "logical_ctx_tokens": 5,
        "physical_prefill_tokens": 3,
        "prefill_tokens_restored": 2,
        "prefill_tokens_computed": 3,
        "generated_token_ids": [7],
        "generation_tokens": 1,
        "acceptance_ratio": 1.0,
        "cycles_completed": 1,
        "memory_end": {"phys_footprint_gb": 3.0, "mlx_cache_gb": 0.75},
    }


def test_benchmark_aime25_scoring_ignores_unclosed_reasoning():
    result = {
        "generated_text": "<think>the answer is \\boxed{60}",
    }

    benchmark._attach_aime25_score(result, "60")

    assert result["score"]["prediction"] is None
    assert result["score"]["correct"] is False
    assert result["scored_text_chars"] == 0


def test_benchmark_aime25_scoring_uses_final_content_after_reasoning():
    result = {
        "generated_text": "<think>wrong \\boxed{10}</think>\nFinal answer: \\boxed{16}",
    }

    benchmark._attach_aime25_score(result, "016")

    assert result["score"]["prediction"] == "16"
    assert result["score"]["expected"] == "16"
    assert result["score"]["correct"] is True
    assert result["score"]["answer_source"] == "final_line"
    assert result["scored_text_sha1"]
    assert "Final answer" in result["scored_text_tail"]


def test_benchmark_aime25_scoring_prefers_explicit_final_answer_over_prior_boxed():
    result = {
        "generated_text": "A false intermediate claim: \\boxed{10}\nFinal answer: 16",
    }

    benchmark._attach_aime25_score(result, "16")

    assert result["score"]["prediction"] == "16"
    assert result["score"]["correct"] is True
    assert result["score"]["answer_source"] == "final_line"


def test_benchmark_aime25_scoring_requires_explicit_answer():
    result = {
        "generated_text": "I checked 15 cases and then 16 cases, but did not finalize.",
    }

    benchmark._attach_aime25_score(result, "16")

    assert result["score"]["prediction"] is None
    assert result["score"]["correct"] is False
    assert result["score"]["answer_source"] is None


def test_benchmark_omits_baseline_peak_memory_when_reset_fails(monkeypatch, capsys):
    def reset_fails():
        raise RuntimeError("reset failed")

    monkeypatch.setattr(benchmark.mx, "reset_peak_memory", reset_fails, raising=False)
    monkeypatch.setattr(
        benchmark,
        "mlx_stream_generate",
        lambda *_args, **_kwargs: iter(
            [
                SimpleNamespace(
                    token=7,
                    prompt_tokens=3,
                    prompt_tps=100.0,
                    generation_tokens=1,
                    generation_tps=10.0,
                    peak_memory=42.0,
                )
            ]
        ),
    )

    result = benchmark._generate_stock_baseline_once(
        target_model=object(),
        tokenizer=_BenchmarkTokenizer(),
        prompt="prompt",
        max_new_tokens=1,
        no_eos=False,
        use_chat_template=False,
    )

    assert result["peak_memory_gb"] is None
    assert "baseline peak memory reset failed" in capsys.readouterr().err


def test_benchmark_omits_dflash_peak_memory_when_reset_fails(monkeypatch, capsys):
    def reset_fails():
        raise RuntimeError("reset failed")

    stream = _ClosableBenchmarkStream(
        [
            PrefillCompleteEvent(
                prefill_us=100_000.0,
                prompt_token_count=3,
                logical_ctx_tokens=3,
                physical_prefill_tokens=3,
                prefill_tokens_restored=0,
                prefill_tokens_computed=3,
            ),
            TokenEvent(
                token_id=7,
                generated_tokens=1,
                acceptance_ratio=1.0,
                cycles_completed=1,
            ),
            SummaryEvent(
                elapsed_us=1_000_000.0,
                prompt_token_count=3,
                generated_token_ids=(7,),
                generation_tokens=1,
                accepted_from_draft=1,
                acceptance_ratio=1.0,
                cycles_completed=1,
                phase_timings_us={"prefill": 100_000.0},
                peak_memory_gb=42.0,
            ),
        ]
    )
    monkeypatch.setattr(benchmark.mx, "reset_peak_memory", reset_fails, raising=False)
    monkeypatch.setattr(benchmark, "stream_dflash_generate", lambda **_kwargs: stream)

    result = benchmark._generate_dflash_stream_once(
        target_model=object(),
        target_ops=object(),
        tokenizer=_BenchmarkTokenizer(),
        draft_model=object(),
        draft_backend=object(),
        prompt="prompt",
        max_new_tokens=1,
        use_chat_template=False,
        block_tokens=16,
        stop_token_ids=[],
        suppress_token_ids=None,
        runtime_context=object(),
    )

    assert stream.closed is True
    assert result["peak_memory_gb"] is None
    assert result["logical_ctx_tokens"] == 3
    assert result["physical_prefill_tokens"] == 3
    assert result["prefill_tokens_computed"] == 3
    assert "dflash peak memory reset failed" in capsys.readouterr().err


def test_benchmark_memory_summary_includes_end_breakdown():
    result = {
        "runs": [
            {
                "baseline": {
                    "peak_memory_gb": 1.0,
                    "memory_end": {"phys_footprint_gb": 2.0, "mlx_cache_gb": 0.5},
                },
                "dflash": {
                    "peak_memory_gb": 4.0,
                    "memory_end": {"phys_footprint_gb": 5.0, "mlx_cache_gb": 0.75},
                },
            }
        ],
        "summary": {},
    }

    benchmark._attach_memory_summary(result)

    assert result["summary"]["baseline_peak_memory_gb_median"] == 1.0
    assert result["summary"]["dflash_peak_memory_gb_median"] == 4.0
    assert result["summary"]["baseline_memory_end_phys_footprint_gb_median"] == 2.0
    assert result["summary"]["dflash_memory_end_mlx_cache_gb_median"] == 0.75


def test_benchmark_report_aggregates_aime25_scores():
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "aime25", "--model", "m", "--only-dflash"]),
        ["--suite", "aime25", "--model", "m", "--only-dflash"],
    )
    prompts = [
        benchmark_suites.BenchmarkPrompt(
            "aime25:0",
            "aime25",
            "p0",
            "hf",
            expected_answer="70",
        ),
        benchmark_suites.BenchmarkPrompt(
            "aime25:1",
            "aime25",
            "p1",
            "hf",
            expected_answer="588",
        ),
    ]
    prompt_reports = [
        {
            "hardware": {},
            "config": {"prompt_id": prompts[0].id, "prompt_tokens": 3},
            "runs": [
                {
                    "baseline": {},
                    "dflash": {"score": {"correct": True}},
                }
            ],
            "summary": {
                "baseline_tps_median": None,
                "dflash_tps_median": 10.0,
                "dflash_score": 1.0,
            },
        },
        {
            "hardware": {},
            "config": {"prompt_id": prompts[1].id, "prompt_tokens": 4},
            "runs": [
                {
                    "baseline": {},
                    "dflash": {"score": {"correct": True}},
                }
            ],
            "summary": {
                "baseline_tps_median": None,
                "dflash_tps_median": 20.0,
                "dflash_score": 1.0,
            },
        },
    ]

    result = benchmark_report.suite_report(
        prompts=prompts,
        prompt_reports=prompt_reports,
        args=args,
        include_memory=False,
    )

    assert result["config"]["only_dflash"] is True
    assert result["summary"]["baseline_score"] is None
    assert result["summary"]["baseline_correct"] is None
    assert result["summary"]["dflash_score"] == 1.0
    assert result["summary"]["dflash_correct"] == 2
    assert result["summary"]["score_count"] == 2


def test_benchmark_rejects_stale_dict_engine_event(monkeypatch):
    stream = _ClosableBenchmarkStream(
        [
            PrefillCompleteEvent(
                prefill_us=100_000.0,
                prompt_token_count=3,
                logical_ctx_tokens=3,
                physical_prefill_tokens=3,
                prefill_tokens_restored=0,
                prefill_tokens_computed=3,
            ),
            {"event": "token", "token_id": 7},
        ]
    )
    monkeypatch.setattr(benchmark, "_reset_peak_memory_for_benchmark", lambda _mode: True)
    monkeypatch.setattr(benchmark, "stream_dflash_generate", lambda **_kwargs: stream)

    with pytest.raises(TypeError, match="Unsupported DFlash engine event: dict"):
        benchmark._generate_dflash_stream_once(
            target_model=object(),
            target_ops=object(),
            tokenizer=_BenchmarkTokenizer(),
            draft_model=object(),
            draft_backend=object(),
            prompt="prompt",
            max_new_tokens=1,
            use_chat_template=False,
            block_tokens=16,
            stop_token_ids=[],
            suppress_token_ids=None,
            runtime_context=object(),
        )

    assert stream.closed is True


def test_summary_event_keeps_cycle_profile_typed_until_serialization():
    cycle = CycleCompleteEvent(
        cycle=1,
        block_len=16,
        commit_count=8,
        acceptance_len=8,
        draft_us=1.0,
        verify_us=2.0,
        acceptance_us=3.0,
        hidden_extraction_us=4.0,
        commit_us=7.0,
        rollback_us=5.0,
        other_us=6.0,
        cycle_total_us=21.0,
        cycle_wall_us=25.0,
        yield_pause_us=4.0,
        prefix_len=128,
        draft_source="dflash",
        candidate_count=1,
    )
    summary = SummaryEvent(
        elapsed_us=1_000_000.0,
        prompt_token_count=3,
        generated_token_ids=(7,),
        generation_tokens=1,
        accepted_from_draft=1,
        acceptance_ratio=1.0,
        cycles_completed=1,
        phase_timings_us={"prefill": 100_000.0},
        cycle_profile_us=(cycle,),
    )

    assert summary.cycle_profile_us == (cycle,)
    payload = summary.to_payload()
    assert payload["cycle_profile_us"] == [cycle.to_payload()]
    assert payload["cycle_profile_us"][0]["cycle_engine_us"] == 21.0
    assert payload["cycle_profile_us"][0]["cycle_wall_us"] == 25.0
    assert payload["cycle_profile_us"][0]["yield_pause_us"] == 4.0
    assert payload["cycle_profile_us"][0]["prefix_len"] == 128
    assert payload["cycle_profile_us"][0]["draft_source"] == "dflash"


def test_cycle_event_token_id_capture_fields_round_trip():
    base = dict(
        cycle=1,
        block_len=16,
        commit_count=3,
        acceptance_len=2,
        draft_us=1.0,
        verify_us=2.0,
        acceptance_us=3.0,
        hidden_extraction_us=4.0,
        rollback_us=5.0,
        other_us=6.0,
        cycle_total_us=21.0,
    )
    captured = CycleCompleteEvent(
        **base,
        proposed_ids=(11, 22, 33, 44),
        posterior_ids=(11, 22, 99, 77),
        committed_ids=(11, 22, 99),
        draft_topk_ids=((22, 5), (33, 7)),
        draft_topk_logprobs=((-0.125, -2.5), (-0.0625, -3.0)),
        posterior_top2_ids=((11, 2), (22, 4), (99, 6), (77, 8)),
        posterior_top2_logprobs=(
            (-0.25, -1.5),
            (-0.5, -2.0),
            (-0.75, -2.25),
            (-1.0, -2.625),
        ),
    )
    payload = captured.to_payload()
    assert payload["proposed_ids"] == [11, 22, 33, 44]
    assert payload["posterior_ids"] == [11, 22, 99, 77]
    assert payload["committed_ids"] == [11, 22, 99]
    assert payload["draft_topk_ids"] == [[22, 5], [33, 7]]
    assert payload["draft_topk_logprobs"] == [[-0.125, -2.5], [-0.0625, -3.0]]
    assert payload["posterior_top2_ids"] == [[11, 2], [22, 4], [99, 6], [77, 8]]
    assert payload["posterior_top2_logprobs"] == [
        [-0.25, -1.5],
        [-0.5, -2.0],
        [-0.75, -2.25],
        [-1.0, -2.625],
    ]

    uncaptured = CycleCompleteEvent(**base)
    payload = uncaptured.to_payload()
    assert "proposed_ids" not in payload
    assert "posterior_ids" not in payload
    assert "committed_ids" not in payload
    assert "draft_topk_ids" not in payload
    assert "draft_topk_logprobs" not in payload
    assert "posterior_top2_ids" not in payload
    assert "posterior_top2_logprobs" not in payload


def test_benchmark_cleanup_failure_is_reported(monkeypatch, capsys):
    def clear_cache_fails():
        raise RuntimeError("clear failed")

    def metal_clear_cache_fails():
        raise RuntimeError("metal clear failed")

    monkeypatch.setattr(benchmark.mx, "clear_cache", clear_cache_fails, raising=False)
    monkeypatch.setattr(
        benchmark.mx,
        "metal",
        SimpleNamespace(clear_cache=metal_clear_cache_fails),
        raising=False,
    )

    benchmark._release_loaded_models()

    err = capsys.readouterr().err
    assert "MLX cache cleanup failed" in err
    assert "clear failed" in err
    assert "metal clear failed" in err


def test_benchmark_cleanup_primary_failure_with_fallback_success_is_reported(monkeypatch, capsys):
    fallback_calls = 0

    def clear_cache_fails():
        raise RuntimeError("clear failed")

    def metal_clear_cache_succeeds():
        nonlocal fallback_calls
        fallback_calls += 1

    monkeypatch.setattr(benchmark.mx, "clear_cache", clear_cache_fails, raising=False)
    monkeypatch.setattr(
        benchmark.mx,
        "metal",
        SimpleNamespace(clear_cache=metal_clear_cache_succeeds),
        raising=False,
    )

    benchmark._release_loaded_models()

    err = capsys.readouterr().err
    assert fallback_calls == 1
    assert "used metal.clear_cache after clear_cache failed" in err
    assert "clear failed" in err


def _patch_benchmark_target(monkeypatch, *, resolved_model_ref: str = "target"):
    target = object()
    tokenizer = _BenchmarkTokenizer()
    monkeypatch.setattr(
        benchmark,
        "_load_pristine_target_bundle",
        lambda model_ref: (object(), tokenizer, {"resolved_model_ref": model_ref}),
    )
    monkeypatch.setattr(
        benchmark,
        "_generate_stock_baseline_once",
        lambda **kwargs: _fake_baseline_result(),
    )
    return target, tokenizer, {
        "resolved_model_ref": resolved_model_ref,
    }


def _run_one_fake_benchmark(
    *,
    draft_model_ref: str | None = None,
) -> dict:
    return benchmark._run_once_sequential(
        prompt="prompt",
        max_new_tokens=1,
        block_tokens=16,
        use_chat_template=False,
        target_model_ref="target",
        draft_model_ref=draft_model_ref,
        draft_quant=None,
        no_eos=True,
    )


def test_benchmark_dflash_path_binds_draft_before_generation(monkeypatch):
    events: list[str] = []
    target, tokenizer, target_meta = _patch_benchmark_target(monkeypatch)
    draft = _BindableBenchmarkDraft(events)
    draft_backend = object()
    ops = object()
    bundle_calls: list[dict] = []

    def fake_load_runtime_bundle(**kwargs):
        bundle_calls.append(kwargs)
        draft.bind_target_model(target, target_ops=ops)
        return SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=draft,
            draft_meta={"resolved_model_ref": "auto-draft"},
            draft_backend=draft_backend,
            target_ops=ops,
        )

    monkeypatch.setattr(benchmark, "load_runtime_bundle", fake_load_runtime_bundle)

    def fake_generate(**kwargs):
        events.append("generate")
        assert kwargs["draft_model"].bound is True
        assert kwargs["draft_model"].bound_target is target
        assert kwargs["draft_model"].bound_ops is ops
        assert kwargs["draft_backend"] is draft_backend
        assert kwargs["target_ops"] is ops
        return _fake_dflash_result()

    monkeypatch.setattr(benchmark, "_generate_dflash_stream_once", fake_generate)

    report = _run_one_fake_benchmark()

    assert events == ["bind", "generate"]
    assert len(bundle_calls) == 1
    assert bundle_calls[0]["model_ref"] == "target"
    assert bundle_calls[0]["draft_ref"] is None
    assert bundle_calls[0]["draft_quant"] is None
    assert bundle_calls[0]["verify_config"] is not None
    assert "split_full_attention_sdpa" not in bundle_calls[0]
    assert report["draft_meta"]["resolved_model_ref"] == "auto-draft"


def test_benchmark_only_dflash_skips_pristine_baseline(monkeypatch):
    target = object()
    tokenizer = _BenchmarkTokenizer()
    draft = object()
    draft_backend = object()
    ops = object()

    def fail_pristine(_model_ref):
        raise AssertionError("baseline should not load in --only-dflash mode")

    def fake_load_runtime_bundle(**kwargs):
        return SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta={"resolved_model_ref": kwargs["model_ref"]},
            draft_model=draft,
            draft_meta={"resolved_model_ref": "auto-draft"},
            draft_backend=draft_backend,
            target_ops=ops,
        )

    monkeypatch.setattr(benchmark, "_load_pristine_target_bundle", fail_pristine)
    monkeypatch.setattr(benchmark, "load_runtime_bundle", fake_load_runtime_bundle)
    monkeypatch.setattr(benchmark, "_generate_dflash_stream_once", lambda **_kwargs: _fake_dflash_result())

    report = benchmark.benchmark_once(
        prompt="prompt",
        max_new_tokens=1,
        block_tokens=16,
        use_chat_template=False,
        target_model_ref="target",
        draft_model_ref=None,
        no_eos=True,
        only_dflash=True,
    )

    assert report["config"]["only_dflash"] is True
    assert report["runs"][0]["baseline"]["generation_tps"] is None
    assert report["runs"][0]["baseline"]["prefill_tokens_physical"] is None
    assert report["runs"][0]["dflash"]["generation_tps"] > 0
    assert report["runs"][0]["speedup"] is None
    assert report["summary"]["baseline_tps_median"] is None


def test_benchmark_cooldown_applies_between_baseline_and_dflash(monkeypatch):
    events: list[str] = []
    target, tokenizer, target_meta = _patch_benchmark_target(monkeypatch)

    def fake_baseline(**_kwargs):
        events.append("baseline")
        return _fake_baseline_result()

    def fake_sleep(seconds):
        events.append(f"sleep:{seconds}")

    def fake_load_runtime_bundle(**_kwargs):
        events.append("load_dflash")
        return SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=_BindableBenchmarkDraft(),
            draft_meta={"resolved_model_ref": "auto-draft"},
            draft_backend=object(),
            target_ops=object(),
        )

    def fake_dflash(**_kwargs):
        events.append("dflash")
        return _fake_dflash_result()

    monkeypatch.setattr(benchmark, "_generate_stock_baseline_once", fake_baseline)
    monkeypatch.setattr(benchmark.time, "sleep", fake_sleep)
    monkeypatch.setattr(benchmark, "load_runtime_bundle", fake_load_runtime_bundle)
    monkeypatch.setattr(benchmark, "_generate_dflash_stream_once", fake_dflash)

    report = benchmark.benchmark_once(
        prompt="prompt",
        max_new_tokens=1,
        block_tokens=16,
        use_chat_template=False,
        target_model_ref="target",
        draft_model_ref=None,
        no_eos=True,
        cooldown=60,
    )

    assert events == ["baseline", "sleep:60", "load_dflash", "dflash"]
    assert report["config"]["cooldown"] == 60


def test_benchmark_cooldown_applies_between_repeats(monkeypatch):
    events: list[str] = []
    target, tokenizer, target_meta = _patch_benchmark_target(monkeypatch)

    monkeypatch.setattr(
        benchmark,
        "_generate_stock_baseline_once",
        lambda **_kwargs: events.append("baseline") or _fake_baseline_result(),
    )
    monkeypatch.setattr(
        benchmark.time,
        "sleep",
        lambda seconds: events.append(f"sleep:{seconds}"),
    )
    monkeypatch.setattr(
        benchmark,
        "load_runtime_bundle",
        lambda **_kwargs: events.append("load_dflash")
        or SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=_BindableBenchmarkDraft(),
            draft_meta={"resolved_model_ref": "auto-draft"},
            draft_backend=object(),
            target_ops=object(),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_generate_dflash_stream_once",
        lambda **_kwargs: events.append("dflash") or _fake_dflash_result(),
    )

    benchmark.benchmark_repeated(
        prompt="prompt",
        max_new_tokens=1,
        repeat=2,
        block_tokens=16,
        use_chat_template=False,
        target_model_ref="target",
        draft_model_ref=None,
        no_eos=True,
        cooldown=60,
    )

    assert events == [
        "baseline",
        "sleep:60",
        "load_dflash",
        "dflash",
        "sleep:60",
        "baseline",
        "sleep:60",
        "load_dflash",
        "dflash",
    ]


def test_benchmark_rejects_tokenizer_drift_between_baseline_and_dflash(monkeypatch):
    target, _tokenizer, target_meta = _patch_benchmark_target(monkeypatch)

    class _DriftTokenizer(_BenchmarkTokenizer):
        def encode(self, prompt):
            return [1, 2, 3, 4]

    monkeypatch.setattr(
        benchmark,
        "load_runtime_bundle",
        lambda **_kwargs: SimpleNamespace(
            target_model=target,
            tokenizer=_DriftTokenizer(),
            target_meta=target_meta,
            draft_model=_BindableBenchmarkDraft(),
            draft_meta={"resolved_model_ref": "auto-draft"},
            draft_backend=object(),
            target_ops=object(),
        ),
    )

    with pytest.raises(ValueError, match="Tokenizer drift"):
        benchmark.benchmark_once(
            prompt="prompt",
            max_new_tokens=1,
            block_tokens=16,
            use_chat_template=False,
            target_model_ref="target",
            draft_model_ref=None,
            no_eos=True,
        )


def test_benchmark_records_draft_load_dtype_in_config(monkeypatch):
    target, tokenizer, target_meta = _patch_benchmark_target(monkeypatch)
    monkeypatch.setattr(
        benchmark,
        "load_runtime_bundle",
        lambda **kwargs: SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=_BindableBenchmarkDraft(),
            draft_meta={
                "resolved_model_ref": "auto-draft",
                "draft_load_dtype": "float16",
                "draft_load_dtype_source": "old_apple_bf16_emulation",
            },
            draft_backend=object(),
            target_ops=object(),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_generate_dflash_stream_once",
        lambda **kwargs: _fake_dflash_result(),
    )

    report = benchmark.benchmark_once(
        prompt="prompt",
        max_new_tokens=1,
        block_tokens=16,
        use_chat_template=False,
        target_model_ref="target",
        draft_model_ref=None,
        draft_quant=None,
        no_eos=True,
        cooldown=0,
    )

    assert report["config"]["draft_load_dtype"] == "float16"
    assert (
        report["config"]["draft_load_dtype_source"]
        == "old_apple_bf16_emulation"
    )


def test_benchmark_preserves_dflash_phase_timings_in_artifact_entry(monkeypatch):
    target, tokenizer, target_meta = _patch_benchmark_target(monkeypatch)
    monkeypatch.setattr(
        benchmark,
        "load_runtime_bundle",
        lambda **kwargs: SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=_BindableBenchmarkDraft(),
            draft_meta={"resolved_model_ref": "auto-draft"},
            draft_backend=object(),
            target_ops=object(),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_generate_dflash_stream_once",
        lambda **kwargs: _fake_dflash_result(),
    )

    run = _run_one_fake_benchmark()
    run["dflash"]["adaptive_metrics"] = {"cycles_by_block": {"4": 1}}
    assert "generated_token_ids" not in run["dflash"]
    assert run["dflash"]["prefill_us"] == 100_000.0

    entry = benchmark._format_run_entry({**run, "run_index": 1})

    assert entry["baseline"]["prefill_tok_s"] == 3.0
    assert entry["baseline"]["prefill_tok_s_physical"] == 3.0
    assert entry["baseline"]["memory_end"]["mlx_cache_gb"] == 0.5
    assert entry["dflash"]["prefill_tok_s_physical"] == 30.0
    assert entry["dflash"]["prefill_tok_s_apparent"] == 50.0
    assert entry["dflash"]["memory_end"]["phys_footprint_gb"] == 3.0
    assert entry["dflash"]["prefill_tokens_restored"] == 2
    assert entry["dflash"]["prefill_tokens_computed"] == 3
    assert entry["dflash"]["phase_timings_us"] == {
        "prefill": 100_000.0,
        "verify": 50_000.0,
    }
    assert entry["dflash"]["adaptive_metrics"] == {"cycles_by_block": {"4": 1}}


def test_benchmark_explicit_draft_wins_over_registry(monkeypatch):
    loaded_bundle_args: list[dict] = []
    target, tokenizer, target_meta = _patch_benchmark_target(
        monkeypatch,
        resolved_model_ref="mlx-community/gemma-4-26b-a4b-it-4bit",
    )

    def fake_load_runtime_bundle(**kwargs):
        loaded_bundle_args.append(kwargs)
        return SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=_BindableBenchmarkDraft(),
            draft_meta={"resolved_model_ref": kwargs["draft_ref"]},
            draft_backend=object(),
            target_ops=object(),
        )

    monkeypatch.setattr(benchmark, "load_runtime_bundle", fake_load_runtime_bundle)
    monkeypatch.setattr(
        benchmark,
        "_generate_dflash_stream_once",
        lambda **kwargs: _fake_dflash_result(),
    )

    report = _run_one_fake_benchmark(draft_model_ref="manual/draft")

    assert [call["draft_ref"] for call in loaded_bundle_args] == ["manual/draft"]
    assert report["draft_meta"]["resolved_model_ref"] == "manual/draft"


def _patch_missing_auto_draft(monkeypatch) -> None:
    _patch_benchmark_target(monkeypatch, resolved_model_ref="unknown/model")
    monkeypatch.setattr(
        benchmark,
        "load_runtime_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("No DFlash draft model found for 'unknown/model'")
        ),
    )


def test_benchmark_missing_auto_draft_raises_clear_value_error(monkeypatch):
    _patch_missing_auto_draft(monkeypatch)

    with pytest.raises(
        ValueError,
        match="No DFlash draft model found for 'unknown/model'",
    ):
        _run_one_fake_benchmark()


def test_benchmark_cli_missing_auto_draft_exits_cleanly(monkeypatch, tmp_path, capsys):
    _patch_missing_auto_draft(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        benchmark.main(
            [
                "--model",
                "unknown/model",
                "--prompt",
                "prompt",
                "--max-tokens",
                "1",
                "--no-chat-template",
                "--no-memory",
                "--out",
                str(tmp_path / "missing-auto-draft"),
            ],
            prog="dflash benchmark",
        )

    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "No DFlash draft model found for 'unknown/model'" in err
    assert "Traceback" not in err

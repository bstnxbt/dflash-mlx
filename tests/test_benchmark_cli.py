# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import random
from pathlib import Path

import pytest

from dflash_mlx import benchmark
from dflash_mlx import benchmark_report
from dflash_mlx import benchmark_suites
from dflash_mlx.runtime_context import build_offline_runtime_context

def test_prompt_slug_distinguishes_long_prompts_with_same_prefix():
    prefix = "same prefix " * 8
    first = benchmark_suites.slugify_prompt_id(prefix + "alpha")
    second = benchmark_suites.slugify_prompt_id(prefix + "beta")

    assert first != second
    assert first.startswith("same-prefix")
    assert second.startswith("same-prefix")

def test_benchmark_help_documents_public_flags(capsys):
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    expected = [
        "--suite {smoke,humaneval,gsm8k,math500,longctx}",
        "Named runtime prompt suite. Default: smoke.",
        "--limit N",
        "Number of prompts to run from the selected suite.",
        "--ctx-tokens N",
        "Synthetic long-context token target for --suite longctx. Default: 8192.",
        "--prompt-file PATH",
        "JSONL prompt file override",
        "--shuffle",
        "Shuffle HF dataset rows before --limit selection. Default: disabled.",
        "--seed INT",
        "Shuffle seed used only with --shuffle. Default: 0.",
        "--max-tokens INT",
        "Number of tokens to generate. Default: 64.",
        "--block-tokens INT",
        "DFlash speculative verify block size. Default: 16.",
        "--ctx INT",
        "Existing shorthand for --ctx-tokens. Default: 0.",
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
        "--draft-quant SPEC",
        "Optional in-memory draft quantization, e.g. w4:gs64.",
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
        ".artifacts/dflash/benchmarks/<timestamp>-<suite>-<model>",
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
            "--suite",
            "longctx",
            "--limit",
            "1",
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
    args = benchmark._finalize_benchmark_args(args, ["--suite", "longctx", "--limit", "1", "--ctx", "65536"])
    config = {
        "model": "resolved-target",
        "draft": "resolved-draft",
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
    assert invocation["command"].startswith("dflash benchmark --suite longctx")
    assert invocation["protocol_order"] == ["baseline", "dflash"]
    assert invocation["same_prompt_token_ids"] is True
    assert invocation["primary_metric"] == "post_prefill_generation_tps"
    assert invocation["explicit_flags"]["model"] == "resolved-target"
    assert invocation["explicit_flags"]["draft"] == "resolved-draft"
    assert invocation["explicit_flags"]["suite"] == "longctx"
    assert invocation["explicit_flags"]["limit"] == 1
    assert invocation["explicit_flags"]["max_tokens"] == 8
    assert invocation["explicit_flags"]["ctx"] == 65536
    assert invocation["explicit_flags"]["no_memory"] is True
    assert invocation["explicit_flags"]["out"] == "/tmp/result.json"
    assert invocation["explicit_flags"]["draft_sink_size"] == 32
    assert invocation["explicit_flags"]["draft_window_size"] == 512
    assert invocation["explicit_flags"]["verify_len_cap"] == 8
    assert invocation["effective"]["model"] == "resolved-target"
    assert invocation["effective"]["draft"] == "resolved-draft"
    assert invocation["effective"]["suite"] == "longctx"
    assert invocation["effective"]["ctx_tokens"] == 65536
    assert invocation["effective"]["shuffle"] is False
    assert invocation["effective"]["seed"] == 0
    assert invocation["effective"]["include_memory"] is False
    assert invocation["effective"]["use_chat_template"] is False
    assert invocation["effective"]["split_sdpa"] is False

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
    assert [prompt.id for prompt in prompts] == ["smoke-default"]

def _fake_hf_rows(suite: str, count: int = 5) -> list[dict[str, str]]:
    if suite == "humaneval":
        return [
            {"task_id": f"HumanEval/{idx}", "prompt": f"def f_{idx}():\n    pass\n"}
            for idx in range(count)
        ]
    if suite == "gsm8k":
        return [{"question": f"question {idx}"} for idx in range(count)]
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

    with pytest.raises(RuntimeError, match="Install datasets to use --suite humaneval/gsm8k/math500"):
        benchmark_suites.resolve_benchmark_prompts(args)

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

    assert "| smoke | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in text

def test_benchmark_manifest_config_fields_are_suite_aware(monkeypatch):
    monkeypatch.setattr(
        benchmark_suites,
        "load_hf_dataset",
        lambda name, config, split: _fake_hf_rows("humaneval", count=3),
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "humaneval", "--limit", "2", "--model", "m", "--shuffle", "--seed", "9"]),
        ["--suite", "humaneval", "--limit", "2", "--model", "m", "--shuffle", "--seed", "9"],
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
    assert result["runs"] == []

def test_benchmark_runs_json_rows_keep_provenance():
    prompts = [
        benchmark_suites.BenchmarkPrompt("p0", "smoke", "prompt", "smoke"),
    ]
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "smoke", "--model", "m", "--draft", "d"]),
        ["--suite", "smoke", "--model", "m", "--draft", "d"],
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
    assert row["git_hash"] == "abc123"
    assert row["prompt_tokenization_mode"] == "chat_template"
    assert row["max_tokens"] == 64

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
    assert "dflash benchmark --suite smoke" in docs
    assert ".artifacts/dflash/diagnostics" in docs
    assert ".artifacts/dflash/benchmarks" in docs

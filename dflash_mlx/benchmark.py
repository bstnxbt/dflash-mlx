# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import argparse
import gc
import hashlib
import json
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import stream_generate as mlx_stream_generate
from mlx_lm.utils import load as load_pristine_target

from dflash_mlx.artifacts import create_run_dir, write_json, write_jsonl, write_manifest
from dflash_mlx.benchmark_report import (
    print_summary,
    suite_report,
    summary_markdown,
)
from dflash_mlx.benchmark_suites import (
    DEFAULT_CTX_TOKENS,
    DEFAULT_PROMPT,
    SUITE_CHOICES,
    BenchmarkPrompt,
    ctx_tokens as _ctx_tokens,
    default_limit_for_suite as _default_limit_for_suite,
    resolve_benchmark_prompts,
    slugify_prompt_id,
)
from dflash_mlx.draft_backend import DraftBackend
from dflash_mlx.engine.events import (
    PrefillCompleteEvent,
    SummaryEvent,
    TokenEvent,
    is_engine_event,
)
from dflash_mlx.metal_limits import apply_metal_limits, parse_memory_limit
from dflash_mlx.observability.memory import process_memory_snapshot
from dflash_mlx.runtime import (
    get_stop_token_ids,
    stream_dflash_generate,
)
from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.config import (
    BENCHMARK_RUNTIME_FIELDS,
    GiB,
    add_offline_runtime_arguments,
    offline_runtime_error_message,
    offline_runtime_kwargs,
)
from dflash_mlx.runtime.context import (
    RuntimeContext,
    build_offline_runtime_config,
    build_offline_runtime_context,
)
from dflash_mlx.runtime.loading import resolve_model_ref

CONTROLLED_FLAG_NAMES = (
    "suite",
    "limit",
    "ctx_tokens",
    "prompt_file",
    "shuffle",
    "seed",
    "prompt",
    "max_tokens",
    "block_tokens",
    "include_memory",
    "no_memory",
    "repeat",
    "cooldown",
    "wired_limit",
    "cache_limit",
    "model",
    "draft",
    "use_chat_template",
    "draft_quant",
    "no_eos",
    "prefill_step_size",
    "target_fa_window",
    "draft_sink_size",
    "draft_window_size",
    "verify_len_cap",
    "verify_mode",
    "copyspec_mode",
    "only_dflash",
    "out",
)
DEFAULT_MAX_TOKENS = 64
AIME25_DEFAULT_MAX_TOKENS = 65536

_AIME_INT_RE = re.compile(r"(?<!\d)([+-]?\d{1,4})(?!\d)")
_AIME_FINAL_RE = re.compile(
    r"(?:answer|final answer)\s*[:：]\s*(?:\\boxed\s*)?\{?\$?\s*([+-]?\d{1,4})",
    re.IGNORECASE,
)
_REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|reason|reasoning|thought)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_DETAILS_REASONING_RE = re.compile(
    r"<details\b(?=[^>]*\btype=[\"']reasoning[\"'])[^>]*>.*?</details\s*>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_OPEN_RE = re.compile(
    r"<(think|thinking|reason|reasoning|thought)\b[^>]*>",
    re.IGNORECASE,
)
_DETAILS_REASONING_OPEN_RE = re.compile(
    r"<details\b(?=[^>]*\btype=[\"']reasoning[\"'])[^>]*>",
    re.IGNORECASE,
)

class _BenchmarkHelpFormatter(argparse.RawTextHelpFormatter):
    def __init__(self, prog: str):
        super().__init__(prog, max_help_position=28, width=100)

def _git_hash_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"

def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"

def _hardware_info() -> dict[str, str]:
    return {
        "chip": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_gb": str(
            int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            // (1024**3)
        ),
        "mlx_version": mx.__version__,
        "mlx_lm_version": _package_version("mlx-lm"),
        "dflash_mlx_version": _package_version("dflash-mlx"),
        "python": platform.python_version(),
    }

def _get_thermal_pressure() -> str:
    try:
        out = subprocess.check_output(["pmset", "-g", "therm"], text=True, timeout=2)
        for line in out.splitlines():
            if "CPU_Scheduler_Limit" not in line:
                continue
            val = int(line.strip().split("=")[-1].strip())
            if val == 100:
                return "nominal"
            if val >= 80:
                return "fair"
            if val >= 50:
                return "serious"
            return "critical"
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return "unknown"
    return "unknown"

def _warn_if_throttled(thermal_pressure: str) -> None:
    if thermal_pressure == "nominal":
        return
    print(
        f"WARNING: thermal pressure is '{thermal_pressure}' — results may be throttled. "
        "Increase --cooldown or wait for chip to cool.",
        file=sys.stderr,
    )

def _warn_benchmark(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)

def _gb_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000.0

def _benchmark_memory_snapshot(label: str) -> dict[str, float | None] | None:
    try:
        snapshot = process_memory_snapshot(include_system_wired=False)
    except Exception as exc:
        _warn_benchmark(
            f"{label} memory snapshot failed: {type(exc).__name__}: {exc}; "
            "memory breakdown will be omitted."
        )
        return None
    return {
        "rss_gb": _gb_or_none(snapshot.get("rss_bytes")),
        "phys_footprint_gb": _gb_or_none(snapshot.get("phys_footprint_bytes")),
        "mlx_active_gb": _gb_or_none(snapshot.get("mlx_active_bytes")),
        "mlx_cache_gb": _gb_or_none(snapshot.get("mlx_cache_bytes")),
        "mlx_peak_gb": _gb_or_none(snapshot.get("mlx_peak_bytes")),
        "untracked_gb": _gb_or_none(snapshot.get("untracked_bytes")),
    }

def _reset_peak_memory_for_benchmark(label: str) -> bool:
    reset = getattr(mx, "reset_peak_memory", None)
    if reset is None:
        return False
    try:
        reset()
        return True
    except Exception as exc:
        _warn_benchmark(
            f"{label} peak memory reset failed: "
            f"{type(exc).__name__}: {exc}; peak memory will be omitted."
        )
        return False

def _peak_memory_gb_if_reset(reset_ok: bool) -> float | None:
    if not reset_ok or not hasattr(mx, "get_peak_memory"):
        return None
    return float(mx.get_peak_memory()) / 1e9

def _slugify_model_ref(model_ref: str | None) -> str:
    resolved = resolve_model_ref(model_ref, kind="target")
    label = Path(str(resolved)).name or str(resolved)
    label = re.sub(r"[^a-z0-9]+", "-", label.lower())
    label = re.sub(r"-+", "-", label).strip("-")
    return label or "model"

def _benchmark_mode(args: argparse.Namespace) -> str:
    suite = getattr(args, "suite", None)
    if suite:
        return str(suite)
    if _ctx_tokens(args) > 0:
        return "longctx"
    if int(args.repeat) > 1:
        return "repeated"
    return "smoke"

def _benchmark_label(args: argparse.Namespace) -> str:
    return f"{_benchmark_mode(args)}-{_slugify_model_ref(args.model)}"

def _finalize_benchmark_args(
    args: argparse.Namespace,
    argv_tokens: Sequence[str] | None = None,
) -> argparse.Namespace:
    tokens = list(argv_tokens or [])
    suite_explicit = "--suite" in tokens
    max_tokens_explicit = "--max-tokens" in tokens
    if args.repeat is None:
        args.repeat = 1
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.ctx_tokens is not None and args.ctx_tokens < 0:
        raise ValueError("--ctx-tokens must be >= 0")
    if args.ctx_tokens is not None and args.suite == "smoke" and not suite_explicit:
        args.suite = "longctx"
    if args.suite == "longctx" and (args.ctx_tokens is None or args.ctx_tokens == 0):
        args.ctx_tokens = DEFAULT_CTX_TOKENS
    if args.ctx_tokens is None:
        args.ctx_tokens = 0
    args.seed = int(args.seed)
    if args.limit is None:
        args.limit = _default_limit_for_suite(args.suite)
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.suite == "aime25" and not max_tokens_explicit:
        args.max_tokens = AIME25_DEFAULT_MAX_TOKENS
    if args.cache_limit is None:
        args.cache_limit = 4 * GiB if args.suite == "longctx" else "auto"
    build_offline_runtime_config(**offline_runtime_kwargs(args, BENCHMARK_RUNTIME_FIELDS))
    return args

def _strip_generation_payload(result: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(result)
    cleaned.pop("generated_token_ids", None)
    phase_timings = dict(cleaned.get("phase_timings_us", {}) or {})
    if "prefill" in phase_timings and "prefill_us" not in cleaned:
        cleaned["prefill_us"] = float(phase_timings["prefill"])
    return cleaned

def _compact_phase_timings(result: dict[str, Any]) -> dict[str, float]:
    timings = dict(result.get("phase_timings_us", {}) or {})
    return {str(key): float(value) for key, value in timings.items()}

def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _tokens_per_second(token_count: Any, elapsed_us: Any) -> float | None:
    tokens = _int_or_none(token_count)
    elapsed = _float_or_none(elapsed_us)
    if tokens is None or elapsed is None or elapsed <= 0.0:
        return None
    return tokens / (elapsed / 1_000_000.0)

def _decode_token_ids(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    try:
        return str(tokenizer.decode(token_ids))
    except TypeError:
        return "".join(str(tokenizer.decode(int(token))) for token in token_ids)

def _aime_normalize_answer(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().replace(",", "")
    value = value.strip("$ .")
    if not re.fullmatch(r"[+-]?\d+", value):
        return value or None
    sign = "-" if value.startswith("-") else ""
    digits = value[1:] if value[:1] in "+-" else value
    return sign + (digits.lstrip("0") or "0")

def _boxed_values(text: str) -> list[str]:
    values: list[str] = []
    marker = "\\boxed"
    i = 0
    while True:
        start = text.find(marker, i)
        if start < 0:
            return values
        brace = text.find("{", start + len(marker))
        if brace < 0:
            i = start + len(marker)
            continue
        depth = 0
        for j in range(brace, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    values.append(text[brace + 1 : j])
                    i = j + 1
                    break
        else:
            return values

def _content_without_reasoning_blocks(text: str) -> str:
    content = _DETAILS_REASONING_RE.sub("", str(text or ""))
    content = _REASONING_BLOCK_RE.sub("", content)
    cut_points = [
        match.start()
        for pattern in (_DETAILS_REASONING_OPEN_RE, _REASONING_OPEN_RE)
        if (match := pattern.search(content)) is not None
    ]
    if cut_points:
        content = content[: min(cut_points)]
    return content

def _extract_aime25_answer(text: str) -> tuple[str | None, str | None]:
    content = _content_without_reasoning_blocks(text)
    for line in reversed([line.strip() for line in content.splitlines() if line.strip()]):
        match = _AIME_FINAL_RE.search(line)
        if match is not None:
            return _aime_normalize_answer(match.group(1)), "final_line"
    boxed = _boxed_values(content)
    if boxed:
        return _aime_normalize_answer(boxed[-1]), "boxed"
    return None, None

def _attach_aime25_score(result: dict[str, Any], expected_answer: str) -> None:
    generated_text = str(result.pop("generated_text", ""))
    scored_text = _content_without_reasoning_blocks(generated_text)
    prediction, source = _extract_aime25_answer(generated_text)
    expected = _aime_normalize_answer(expected_answer)
    result["generated_text_chars"] = len(generated_text)
    result["scored_text_chars"] = len(scored_text)
    result["scored_text_sha1"] = hashlib.sha1(
        scored_text.encode("utf-8")
    ).hexdigest()
    result["scored_text_tail"] = scored_text[-1000:]
    result["score"] = {
        "kind": "aime25_exact",
        "expected": expected,
        "prediction": prediction,
        "correct": bool(prediction is not None and expected is not None and prediction == expected),
        "answer_source": source,
    }

def _format_run_entry(run: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(run["baseline"])
    dflash = dict(run["dflash"])
    baseline_prefill_us = baseline.get("prefill_us")
    baseline_prefill_tok_s = _float_or_none(
        baseline.get("prefill_tok_s")
    ) or _tokens_per_second(baseline.get("prompt_token_count"), baseline_prefill_us)
    dflash_prefill_us = dflash.get("prefill_us")
    dflash_logical_prefill_tokens = _int_or_none(
        dflash.get("logical_ctx_tokens", dflash.get("prompt_token_count"))
    )
    dflash_physical_prefill_tokens = _int_or_none(
        dflash.get(
            "physical_prefill_tokens",
            dflash.get("prefill_tokens_computed", dflash.get("prompt_token_count")),
        )
    )
    dflash_entry = {
        "ttft_ms": float(run["dflash_ttft_ms"]),
        "generation_tps": float(run["dflash_generation_tps"]),
        "prefill_tok_s_physical": _tokens_per_second(
            dflash_physical_prefill_tokens, dflash_prefill_us
        ),
        "prefill_tok_s_apparent": _tokens_per_second(
            dflash_logical_prefill_tokens, dflash_prefill_us
        ),
        "prefill_tokens_physical": dflash_physical_prefill_tokens,
        "prefill_tokens_restored": _int_or_none(dflash.get("prefill_tokens_restored")),
        "prefill_tokens_computed": _int_or_none(dflash.get("prefill_tokens_computed")),
        "logical_ctx_tokens": dflash_logical_prefill_tokens,
        "accepted_from_draft": _int_or_none(dflash.get("accepted_from_draft")),
        "tokens_per_cycle": float(dflash.get("tokens_per_cycle", 0.0)),
        "cycles": int(dflash.get("cycles_completed", 0)),
        "acceptance_ratio": float(dflash.get("acceptance_ratio", 0.0)),
        "acceptance_first_20_avg": float(dflash.get("acceptance_first_20_avg", 0.0)),
        "acceptance_last_20_avg": float(dflash.get("acceptance_last_20_avg", 0.0)),
        "peak_memory_gb": dflash.get("peak_memory_gb"),
    }
    phase_timings_us = _compact_phase_timings(dflash)
    if phase_timings_us:
        dflash_entry["phase_timings_us"] = phase_timings_us
    baseline_memory_end = (
        dict(baseline["memory_end"]) if baseline.get("memory_end") is not None else None
    )
    if dflash.get("memory_end") is not None:
        dflash_entry["memory_end"] = dict(dflash["memory_end"])
    if baseline.get("score") is not None:
        baseline["score"] = dict(baseline["score"])
    if dflash.get("score") is not None:
        dflash_entry["score"] = dict(dflash["score"])
    if dflash.get("adaptive_metrics") is not None:
        dflash_entry["adaptive_metrics"] = dict(dflash["adaptive_metrics"])
    if baseline.get("generated_text_chars") is not None:
        baseline["generated_text_chars"] = int(baseline["generated_text_chars"])
    if baseline.get("scored_text_chars") is not None:
        baseline["scored_text_chars"] = int(baseline["scored_text_chars"])
    if dflash.get("generated_text_chars") is not None:
        dflash_entry["generated_text_chars"] = int(dflash["generated_text_chars"])
    if dflash.get("scored_text_chars") is not None:
        dflash_entry["scored_text_chars"] = int(dflash["scored_text_chars"])
    baseline_tps = _float_or_none(run.get("baseline_generation_tps"))
    baseline_ttft = _float_or_none(run.get("baseline_ttft_ms"))
    return {
        "run": int(run["run_index"]),
        "thermal_pressure": str(run.get("thermal_pressure", "unknown")),
        "baseline": {
            "ttft_ms": baseline_ttft,
            "generation_tps": baseline_tps,
            "prefill_tok_s": baseline_prefill_tok_s,
            "prefill_tok_s_physical": baseline_prefill_tok_s,
            "prefill_tok_s_apparent": baseline_prefill_tok_s,
            "prefill_tokens_physical": _int_or_none(baseline.get("prompt_token_count")),
            "peak_memory_gb": baseline.get("peak_memory_gb"),
            "memory_end": baseline_memory_end,
            **(
                {
                    "score": baseline["score"],
                    "generated_text_chars": baseline.get("generated_text_chars"),
                    "scored_text_chars": baseline.get("scored_text_chars"),
                }
                if baseline.get("score") is not None
                else {}
            ),
        },
        "dflash": dflash_entry,
        "speedup": float(run["generation_speedup_vs_baseline"]) if run["generation_speedup_vs_baseline"] is not None else None,
    }


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _build_config(
    *,
    prompt: str,
    prompt_tokens: int,
    max_new_tokens: int,
    block_tokens: int,
    repeat: int,
    cooldown: int,
    model: str,
    draft: str,
    use_chat_template: bool,
    draft_quant: str | None,
    draft_load_dtype: str | None,
    draft_load_dtype_source: str | None,
    no_eos: bool,
    prefill_step_size: int,
    target_fa_window: int,
    draft_sink_size: int,
    draft_window_size: int,
    verify_len_cap: int,
    verify_mode: str,
    copyspec_mode: str,
    only_dflash: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "draft": draft,
        "max_tokens": int(max_new_tokens),
        "block_tokens": int(block_tokens),
        "repeat": int(repeat),
        "cooldown": int(cooldown),
        "use_chat_template": bool(use_chat_template),
        "draft_quant": draft_quant,
        "draft_load_dtype": draft_load_dtype,
        "draft_load_dtype_source": draft_load_dtype_source,
        "no_eos": bool(no_eos),
        "prefill_step_size": int(prefill_step_size),
        "target_fa_window": int(target_fa_window),
        "draft_sink_size": int(draft_sink_size),
        "draft_window_size": int(draft_window_size),
        "verify_len_cap": int(verify_len_cap),
        "verify_mode": str(verify_mode),
        "copyspec_mode": str(copyspec_mode),
        "only_dflash": bool(only_dflash),
        "prompt": prompt,
        "prompt_tokens": int(prompt_tokens),
        "prompt_id": slugify_prompt_id(prompt),
        "git_hash": _git_hash_short(),
    }

def _offline_runtime_values(
    *,
    prefill_step_size: int | None = None,
    target_fa_window: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    verify_len_cap: int | None = None,
    verify_mode: str | None = None,
    copyspec_mode: str | None = None,
) -> dict[str, Any]:
    cfg = build_offline_runtime_config(
        prefill_step_size=prefill_step_size,
        target_fa_window=target_fa_window,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
        verify_mode=verify_mode,
        copyspec_mode=copyspec_mode,
    )
    return {
        "prefill_step_size": int(cfg.prefill_step_size),
        "target_fa_window": int(cfg.target_fa_window),
        "draft_sink_size": int(cfg.draft_sink_size),
        "draft_window_size": int(cfg.draft_window_size),
        "verify_len_cap": int(cfg.verify_len_cap),
        "verify_mode": str(cfg.verify_mode),
        "copyspec_mode": str(cfg.copyspec_mode),
    }

def _build_single_case_report(
    *,
    prompt: str,
    max_new_tokens: int,
    block_tokens: int,
    repeat: int,
    cooldown: int,
    runs: list[dict[str, Any]],
    model: str,
    draft: str,
    use_chat_template: bool,
    draft_quant: str | None,
    draft_load_dtype: str | None,
    draft_load_dtype_source: str | None,
    no_eos: bool,
    prefill_step_size: int,
    target_fa_window: int,
    draft_sink_size: int,
    draft_window_size: int,
    verify_len_cap: int,
    verify_mode: str,
    copyspec_mode: str,
    only_dflash: bool,
) -> dict[str, Any]:
    run_entries = [_format_run_entry(run) for run in runs]
    baseline_tps_values = [
        float(value)
        for run in runs
        if (value := _float_or_none(run.get("baseline_generation_tps"))) is not None
    ]
    dflash_tps_values = [
        float(value)
        for run in runs
        if (value := _float_or_none(run.get("dflash_generation_tps"))) is not None
    ]
    speedup_values = [float(run["generation_speedup_vs_baseline"]) for run in runs if run["generation_speedup_vs_baseline"] is not None]
    acceptance_ratio_values = [float(run["dflash"]["acceptance_ratio"]) for run in runs]
    baseline_score_values = [
        bool(score.get("correct"))
        for run in run_entries
        if (score := run.get("baseline", {}).get("score")) is not None
    ]
    dflash_score_values = [
        bool(score.get("correct"))
        for run in run_entries
        if (score := run.get("dflash", {}).get("score")) is not None
    ]
    baseline_prefill_tok_s_values = [
        value
        for run in run_entries
        if (value := run.get("baseline", {}).get("prefill_tok_s")) is not None
    ]
    dflash_prefill_physical_values = [
        value
        for run in run_entries
        if (value := run.get("dflash", {}).get("prefill_tok_s_physical")) is not None
    ]
    dflash_prefill_apparent_values = [
        value
        for run in run_entries
        if (value := run.get("dflash", {}).get("prefill_tok_s_apparent")) is not None
    ]
    prompt_tokens = (
        int(runs[0]["baseline"].get("prompt_token_count") or runs[0]["dflash"].get("prompt_token_count") or 0)
        if runs
        else 0
    )
    return {
        "hardware": _hardware_info(),
        "config": _build_config(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            block_tokens=block_tokens,
            repeat=repeat,
            cooldown=cooldown,
            model=model,
            draft=draft,
            use_chat_template=use_chat_template,
            draft_quant=draft_quant,
            draft_load_dtype=draft_load_dtype,
            draft_load_dtype_source=draft_load_dtype_source,
            no_eos=no_eos,
            prefill_step_size=prefill_step_size,
            target_fa_window=target_fa_window,
            draft_sink_size=draft_sink_size,
            draft_window_size=draft_window_size,
            verify_len_cap=verify_len_cap,
            verify_mode=verify_mode,
            copyspec_mode=copyspec_mode,
            only_dflash=only_dflash,
        ),
        "runs": run_entries,
        "summary": {
            "baseline_tps_median": statistics.median(baseline_tps_values) if baseline_tps_values else None,
            "dflash_tps_median": statistics.median(dflash_tps_values) if dflash_tps_values else None,
            "dflash_tps_min": min(dflash_tps_values) if dflash_tps_values else None,
            "dflash_tps_max": max(dflash_tps_values) if dflash_tps_values else None,
            "speedup_median": statistics.median(speedup_values) if speedup_values else None,
            "baseline_prefill_tok_s_median": (
                statistics.median(baseline_prefill_tok_s_values)
                if baseline_prefill_tok_s_values
                else None
            ),
            "dflash_prefill_tok_s_physical_median": (
                statistics.median(dflash_prefill_physical_values)
                if dflash_prefill_physical_values
                else None
            ),
            "dflash_prefill_tok_s_apparent_median": (
                statistics.median(dflash_prefill_apparent_values)
                if dflash_prefill_apparent_values
                else None
            ),
            "acceptance_ratio_median": statistics.median(acceptance_ratio_values) if acceptance_ratio_values else None,
            "baseline_score": (
                sum(1 for value in baseline_score_values if value) / len(baseline_score_values)
                if baseline_score_values
                else None
            ),
            "baseline_correct": (
                sum(1 for value in baseline_score_values if value)
                if baseline_score_values
                else None
            ),
            "dflash_score": (
                sum(1 for value in dflash_score_values if value) / len(dflash_score_values)
                if dflash_score_values
                else None
            ),
            "dflash_correct": (
                sum(1 for value in dflash_score_values if value)
                if dflash_score_values
                else None
            ),
            "score_count": len(dflash_score_values) or len(baseline_score_values) or None,
        },
    }

def _attach_memory_summary(result: dict[str, Any]) -> None:
    baseline_peaks = [
        float(run.get("baseline", {}).get("peak_memory_gb"))
        for run in result.get("runs", [])
        if run.get("baseline", {}).get("peak_memory_gb") is not None
    ]
    dflash_peaks = [
        float(run.get("dflash", {}).get("peak_memory_gb"))
        for run in result.get("runs", [])
        if run.get("dflash", {}).get("peak_memory_gb") is not None
    ]
    baseline_phys = _memory_end_values(result, "baseline", "phys_footprint_gb")
    dflash_phys = _memory_end_values(result, "dflash", "phys_footprint_gb")
    baseline_cache = _memory_end_values(result, "baseline", "mlx_cache_gb")
    dflash_cache = _memory_end_values(result, "dflash", "mlx_cache_gb")
    result.setdefault("summary", {}).update(
        {
            "baseline_peak_memory_gb_median": statistics.median(baseline_peaks)
            if baseline_peaks
            else None,
            "dflash_peak_memory_gb_median": statistics.median(dflash_peaks)
            if dflash_peaks
            else None,
            "baseline_memory_end_phys_footprint_gb_median": (
                statistics.median(baseline_phys) if baseline_phys else None
            ),
            "dflash_memory_end_phys_footprint_gb_median": (
                statistics.median(dflash_phys) if dflash_phys else None
            ),
            "baseline_memory_end_mlx_cache_gb_median": (
                statistics.median(baseline_cache) if baseline_cache else None
            ),
            "dflash_memory_end_mlx_cache_gb_median": (
                statistics.median(dflash_cache) if dflash_cache else None
            ),
        }
    )

def _memory_end_values(
    result: dict[str, Any],
    side: str,
    key: str,
) -> list[float]:
    values: list[float] = []
    for run in result.get("runs", []):
        memory = run.get(side, {}).get("memory_end") or {}
        value = memory.get(key)
        if value is not None:
            values.append(float(value))
    return values

def _speedup(baseline_elapsed: float, dflash_elapsed: float) -> float | None:
    return baseline_elapsed / dflash_elapsed if dflash_elapsed > 0.0 else None

def _generation_speedup(baseline_tps: float, dflash_tps: float) -> float | None:
    return dflash_tps / baseline_tps if baseline_tps > 0.0 else None

def _ttft_ms_from_baseline(result: dict[str, Any]) -> float:
    return float(result.get("prefill_us", 0.0)) / 1_000.0

def _ttft_ms_from_dflash(result: dict[str, Any]) -> float:
    ttft_us = result.get("ttft_us")
    if ttft_us is not None:
        return float(ttft_us) / 1_000.0
    phase_timings = dict(result.get("phase_timings_us", {}))
    return float(phase_timings.get("prefill", 0.0)) / 1_000.0

def _generation_tps_from_baseline(result: dict[str, Any]) -> float:
    if "generation_tps" in result:
        return float(result["generation_tps"])
    elapsed_us = float(result.get("elapsed_us", 0.0))
    prefill_us = float(result.get("prefill_us", 0.0))
    generation_tokens = int(result.get("generation_tokens", 0))
    generation_us = max(0.0, elapsed_us - prefill_us)
    return (generation_tokens / (generation_us / 1e6)) if generation_us > 0.0 else 0.0

def _generation_tps_from_dflash(result: dict[str, Any]) -> float:
    elapsed_us = float(result.get("elapsed_us", 0.0))
    phase_timings = dict(result.get("phase_timings_us", {}))
    prefill_us = float(phase_timings.get("prefill", 0.0))
    generation_tokens = int(result.get("generation_tokens", 0))
    generation_us = max(0.0, elapsed_us - prefill_us)
    return (generation_tokens / (generation_us / 1e6)) if generation_us > 0.0 else 0.0

def _load_pristine_target_bundle(model_ref: str | None):
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    model, tokenizer, config = load_pristine_target(resolved_ref, lazy=True, return_config=True)
    return model, tokenizer, {"resolved_model_ref": resolved_ref, "config": config}

def _prompt_tokens_for(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))

def _generate_stock_baseline_once(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    no_eos: bool,
    use_chat_template: bool = True,
    prompt_tokens_override: list[int] | None = None,
    capture_text: bool = False,
) -> dict[str, Any]:
    memory_reset_ok = _reset_peak_memory_for_benchmark("baseline")

    if prompt_tokens_override is not None:
        baseline_input: Any = list(prompt_tokens_override)
    elif use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        baseline_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        baseline_input = prompt

    original_eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    original_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if no_eos:
        try:
            tokenizer.eos_token_ids = set()
        except (AttributeError, TypeError, ValueError):
            tokenizer.eos_token_ids = []
        try:
            tokenizer.eos_token_id = None
        except (AttributeError, TypeError, ValueError) as exc:
            _warn_benchmark(
                f"tokenizer eos_token_id could not be cleared for --no-eos: {exc}"
            )

    generated_token_ids: list[int] = []
    final_response = None
    start_ns = time.perf_counter_ns()
    try:
        for response in mlx_stream_generate(
            target_model,
            tokenizer,
            baseline_input,
            max_tokens=max_new_tokens,
        ):
            final_response = response
            generated_token_ids.append(int(response.token))
    finally:
        if no_eos:
            tokenizer.eos_token_ids = original_eos_token_ids
            tokenizer.eos_token_id = original_eos_token_id

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    if final_response is None:
        prompt_tokens = len(tokenizer.encode(prompt))
        return {
            "elapsed_us": elapsed_us,
            "prefill_us": 0.0,
            "prompt_token_count": prompt_tokens,
            "generated_token_ids": [],
            "generation_tokens": 0,
            **({"generated_text": ""} if capture_text else {}),
            "peak_memory_gb": _peak_memory_gb_if_reset(memory_reset_ok),
            "memory_end": _benchmark_memory_snapshot("baseline"),
        }

    prompt_tokens = int(final_response.prompt_tokens)
    prompt_tps = float(final_response.prompt_tps)
    generation_tokens = int(final_response.generation_tokens)
    generation_tps = float(final_response.generation_tps)
    prefill_us = (prompt_tokens / prompt_tps) * 1e6 if prompt_tps > 0.0 else 0.0
    return {
        "elapsed_us": elapsed_us,
        "prefill_us": prefill_us,
        "prefill_tok_s": prompt_tps,
        "prompt_token_count": prompt_tokens,
        "generated_token_ids": generated_token_ids,
        "generation_tokens": generation_tokens,
        "generation_tps": generation_tps,
        **(
            {"generated_text": _decode_token_ids(tokenizer, generated_token_ids)}
            if capture_text
            else {}
        ),
        "peak_memory_gb": (
            float(final_response.peak_memory) if memory_reset_ok else None
        ),
        "memory_end": _benchmark_memory_snapshot("baseline"),
    }

def _generate_dflash_stream_once(
    *,
    target_model: Any,
    target_ops: Any,
    tokenizer: Any,
    draft_model: Any,
    draft_backend: DraftBackend,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool,
    block_tokens: int | None,
    stop_token_ids: list[int] | None,
    suppress_token_ids: list[int] | None,
    runtime_context: RuntimeContext,
    prompt_tokens_override: list[int] | None = None,
    capture_text: bool = False,
) -> dict[str, Any]:
    memory_reset_ok = _reset_peak_memory_for_benchmark("dflash")

    start_ns = time.perf_counter_ns()
    first_token_us: float | None = None
    prefill: PrefillCompleteEvent | None = None
    summary: SummaryEvent | None = None
    stream = stream_dflash_generate(
        target_model=target_model,
        target_ops=target_ops,
        tokenizer=tokenizer,
        draft_model=draft_model,
        draft_backend=draft_backend,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
        block_tokens=block_tokens,
        stop_token_ids=stop_token_ids,
        suppress_token_ids=suppress_token_ids,
        prompt_tokens_override=prompt_tokens_override,
        runtime_context=runtime_context,
    )
    try:
        for event in stream:
            if isinstance(event, PrefillCompleteEvent):
                prefill = event
                continue
            if isinstance(event, TokenEvent):
                if first_token_us is None:
                    first_token_us = (time.perf_counter_ns() - start_ns) / 1_000.0
                continue
            if isinstance(event, SummaryEvent):
                summary = event
                continue
            if not is_engine_event(event):
                raise TypeError(f"Unsupported DFlash engine event: {type(event).__name__}")
    finally:
        stream.close()

    if summary is None:
        raise RuntimeError("DFlash stream did not yield a summary event")
    summary_payload = summary.to_payload()
    if capture_text:
        summary_payload["generated_text"] = _decode_token_ids(
            tokenizer,
            [int(token) for token in summary_payload.get("generated_token_ids", [])],
        )
    if prefill is not None:
        summary_payload.update(prefill.to_payload())
    if not memory_reset_ok:
        summary_payload["peak_memory_gb"] = None
    summary_payload["memory_end"] = _benchmark_memory_snapshot("dflash")

    summary_payload["ttft_us"] = (
        first_token_us
        if first_token_us is not None
        else float(dict(summary.phase_timings_us).get("prefill", 0.0))
    )
    return summary_payload

def _release_loaded_models() -> None:
    gc.collect()
    clear_error: Exception | None = None
    if hasattr(mx, "clear_cache"):
        try:
            mx.clear_cache()
            return
        except Exception as exc:
            clear_error = exc
    if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        try:
            mx.metal.clear_cache()
            if clear_error is not None:
                _warn_benchmark(
                    f"MLX cache cleanup used metal.clear_cache after "
                    f"clear_cache failed: {clear_error}"
                )
            return
        except Exception as exc:
            if clear_error is None:
                clear_error = exc
            else:
                _warn_benchmark(
                    f"MLX cache cleanup failed: clear_cache={clear_error}; "
                    f"metal.clear_cache={exc}"
                )
                return
    if clear_error is not None:
        _warn_benchmark(f"MLX cache cleanup failed: {clear_error}")

def _run_once_sequential(
    *,
    prompt: str,
    max_new_tokens: int,
    block_tokens: int,
    use_chat_template: bool,
    target_model_ref: str | None,
    draft_model_ref: str | None,
    draft_quant: str | None,
    no_eos: bool,
    prefill_step_size: int | None = None,
    target_fa_window: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    verify_len_cap: int | None = None,
    verify_mode: str | None = None,
    copyspec_mode: str | None = None,
    expected_answer: str | None = None,
    cooldown: int = 0,
    only_dflash: bool = False,
) -> dict[str, Any]:
    capture_text = expected_answer is not None
    baseline: dict[str, Any]
    pristine_meta: dict[str, Any] | None = None
    prompt_tokens: list[int] | None = None
    if only_dflash:
        baseline = {
            "elapsed_us": 0.0,
            "prefill_us": 0.0,
            "prompt_token_count": None,
            "generated_token_ids": [],
            "generation_tokens": 0,
            "peak_memory_gb": None,
            "memory_end": None,
        }
    else:
        pristine_target_model, pristine_tokenizer, pristine_meta = _load_pristine_target_bundle(
            target_model_ref
        )
        prompt_tokens = _prompt_tokens_for(
            pristine_tokenizer,
            prompt,
            use_chat_template=use_chat_template,
        )
        try:
            baseline = _generate_stock_baseline_once(
                target_model=pristine_target_model,
                tokenizer=pristine_tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                no_eos=no_eos,
                use_chat_template=use_chat_template,
                prompt_tokens_override=prompt_tokens,
                capture_text=capture_text,
            )
        finally:
            del pristine_target_model
            del pristine_tokenizer
            _release_loaded_models()
        if cooldown > 0:
            time.sleep(cooldown)

    target_model = None
    tokenizer = None
    draft_model = None
    try:
        runtime_context = build_offline_runtime_context(
            prefill_step_size=prefill_step_size,
            target_fa_window=target_fa_window,
            draft_sink_size=draft_sink_size,
            draft_window_size=draft_window_size,
            verify_len_cap=verify_len_cap,
            verify_mode=verify_mode,
            copyspec_mode=copyspec_mode,
        )
        bundle = load_runtime_bundle(
            model_ref=target_model_ref,
            draft_ref=draft_model_ref,
            draft_quant=draft_quant,
            verify_config=runtime_context.verify,
        )
        target_model = bundle.target_model
        tokenizer = bundle.tokenizer
        target_meta = bundle.target_meta
        draft_model = bundle.draft_model
        draft_meta = bundle.draft_meta
        draft_backend = bundle.draft_backend
        target_ops = bundle.target_ops

        dflash_prompt_tokens = _prompt_tokens_for(
            tokenizer,
            prompt,
            use_chat_template=use_chat_template,
        )
        if prompt_tokens is None:
            prompt_tokens = dflash_prompt_tokens
        else:
            if prompt_tokens != dflash_prompt_tokens:
                raise ValueError(
                    "Tokenizer drift between pristine and DFlash bundles: "
                    f"{len(prompt_tokens)} vs {len(dflash_prompt_tokens)} tokens"
                )
        dflash_eos_token_ids = get_stop_token_ids(tokenizer)
        dflash_stop_token_ids = [] if no_eos else dflash_eos_token_ids
        dflash_suppress_token_ids = dflash_eos_token_ids if no_eos else None
        dflash = _generate_dflash_stream_once(
            target_model=target_model,
            target_ops=target_ops,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            block_tokens=block_tokens,
            stop_token_ids=dflash_stop_token_ids,
            suppress_token_ids=dflash_suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            runtime_context=runtime_context,
            capture_text=capture_text,
        )
    finally:
        if target_model is not None:
            del target_model
        if tokenizer is not None:
            del tokenizer
        if draft_model is not None:
            del draft_model
        _release_loaded_models()

    if expected_answer is not None and not only_dflash:
        _attach_aime25_score(baseline, expected_answer)
    if expected_answer is not None:
        _attach_aime25_score(dflash, expected_answer)

    baseline_elapsed = float(baseline["elapsed_us"])
    dflash_elapsed = float(dflash["elapsed_us"])
    baseline_generation_tps = None if only_dflash else _generation_tps_from_baseline(baseline)
    dflash_generation_tps = _generation_tps_from_dflash(dflash)
    return {
        "baseline": _strip_generation_payload(baseline),
        "dflash": _strip_generation_payload(dflash),
        "speedup_vs_baseline": (
            None if only_dflash else _speedup(baseline_elapsed, dflash_elapsed)
        ),
        "baseline_ttft_ms": None if only_dflash else _ttft_ms_from_baseline(baseline),
        "dflash_ttft_ms": _ttft_ms_from_dflash(dflash),
        "baseline_generation_tps": baseline_generation_tps,
        "dflash_generation_tps": dflash_generation_tps,
        "generation_speedup_vs_baseline": (
            None
            if baseline_generation_tps is None
            else _generation_speedup(baseline_generation_tps, dflash_generation_tps)
        ),
        "token_match": (
            None
            if only_dflash
            else baseline["generated_token_ids"] == dflash["generated_token_ids"]
        ),
        "target_meta": target_meta,
        "draft_meta": draft_meta,
        "pristine_target_meta": pristine_meta,
    }


def _effective_draft_quant_label(
    requested: str | None,
    draft_meta: dict[str, Any] | None,
) -> str | None:
    effective = (draft_meta or {}).get("draft_quant_spec")
    if effective is not None:
        return str(effective)
    return requested


def benchmark_once(
    *,
    prompt: str,
    max_new_tokens: int,
    block_tokens: int | None = None,
    use_chat_template: bool,
    target_model_ref: str | None,
    draft_model_ref: str | None,
    draft_quant: str | None = None,
    no_eos: bool = False,
    prefill_step_size: int | None = None,
    target_fa_window: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    verify_len_cap: int | None = None,
    verify_mode: str | None = None,
    copyspec_mode: str | None = None,
    cooldown: int = 10,
    expected_answer: str | None = None,
    only_dflash: bool = False,
) -> dict[str, Any]:
    runtime_values = _offline_runtime_values(
        prefill_step_size=prefill_step_size,
        target_fa_window=target_fa_window,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
        verify_mode=verify_mode,
        copyspec_mode=copyspec_mode,
    )
    thermal_pressure = _get_thermal_pressure()
    _warn_if_throttled(thermal_pressure)
    result = _run_once_sequential(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        block_tokens=block_tokens,
        use_chat_template=use_chat_template,
        target_model_ref=target_model_ref,
        draft_model_ref=draft_model_ref,
        draft_quant=draft_quant,
        no_eos=no_eos,
        expected_answer=expected_answer,
        cooldown=cooldown,
        only_dflash=only_dflash,
        **runtime_values,
    )
    target_meta = result.pop("target_meta")
    draft_meta = result.pop("draft_meta")
    result.pop("pristine_target_meta", None)
    result["run_index"] = 1
    result["thermal_pressure"] = thermal_pressure
    return _build_single_case_report(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        block_tokens=block_tokens if block_tokens is not None else 16,
        repeat=1,
        cooldown=cooldown,
        runs=[result],
        model=target_meta["resolved_model_ref"],
        draft=draft_meta["resolved_model_ref"],
        use_chat_template=use_chat_template,
        draft_quant=_effective_draft_quant_label(draft_quant, draft_meta),
        draft_load_dtype=draft_meta.get("draft_load_dtype"),
        draft_load_dtype_source=draft_meta.get("draft_load_dtype_source"),
        no_eos=no_eos,
        only_dflash=only_dflash,
        **runtime_values,
    )

def benchmark_repeated(
    *,
    prompt: str,
    max_new_tokens: int,
    repeat: int = 1,
    block_tokens: int | None = None,
    use_chat_template: bool = False,
    target_model_ref: str | None = None,
    draft_model_ref: str | None = None,
    draft_quant: str | None = None,
    no_eos: bool = False,
    prefill_step_size: int | None = None,
    target_fa_window: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    verify_len_cap: int | None = None,
    verify_mode: str | None = None,
    copyspec_mode: str | None = None,
    cooldown: int = 10,
    expected_answer: str | None = None,
    only_dflash: bool = False,
) -> dict[str, Any]:
    runtime_values = _offline_runtime_values(
        prefill_step_size=prefill_step_size,
        target_fa_window=target_fa_window,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
        verify_mode=verify_mode,
        copyspec_mode=copyspec_mode,
    )
    target_meta: dict[str, Any] | None = None
    draft_meta: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []

    for run_index in range(1, repeat + 1):
        thermal_pressure = _get_thermal_pressure()
        _warn_if_throttled(thermal_pressure)
        run = _run_once_sequential(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            block_tokens=block_tokens,
            use_chat_template=use_chat_template,
            target_model_ref=target_model_ref,
            draft_model_ref=draft_model_ref,
            draft_quant=draft_quant,
            no_eos=no_eos,
            expected_answer=expected_answer,
            cooldown=cooldown,
            only_dflash=only_dflash,
            **runtime_values,
        )
        if target_meta is None:
            target_meta = run.pop("target_meta")
        else:
            run.pop("target_meta", None)
        if draft_meta is None:
            draft_meta = run.pop("draft_meta")
        else:
            run.pop("draft_meta", None)
        run.pop("pristine_target_meta", None)
        run["run_index"] = run_index
        run["thermal_pressure"] = thermal_pressure
        runs.append(run)
        if cooldown > 0 and run_index < repeat:
            time.sleep(cooldown)

    return _build_single_case_report(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        block_tokens=block_tokens if block_tokens is not None else 16,
        repeat=repeat,
        cooldown=cooldown,
        runs=runs,
        model=target_meta["resolved_model_ref"] if target_meta is not None else "",
        draft=draft_meta["resolved_model_ref"] if draft_meta is not None else "",
        use_chat_template=use_chat_template,
        draft_quant=_effective_draft_quant_label(draft_quant, draft_meta),
        draft_load_dtype=(
            draft_meta.get("draft_load_dtype") if draft_meta is not None else None
        ),
        draft_load_dtype_source=(
            draft_meta.get("draft_load_dtype_source")
            if draft_meta is not None
            else None
        ),
        no_eos=no_eos,
        only_dflash=only_dflash,
        **runtime_values,
    )

def _sustained_summary(gens: list[dict[str, Any]]) -> dict[str, Any]:
    if not gens:
        return {"generations": 0}
    fresh = float(gens[0]["decode_tok_s"])
    total_s = float(gens[-1]["t_start_s"]) + float(gens[-1]["wall_s"])
    # plateau = generations starting in the final 5 minutes, once at least
    # 5 minutes of continuous load have elapsed (the M5-class cliff sits
    # near 3 minutes; anything later is the power-envelope equilibrium).
    plateau_start = max(300.0, total_s - 300.0)
    plateau_rows = [g for g in gens if float(g["t_start_s"]) >= plateau_start]
    plateau = (
        sum(float(g["decode_tok_s"]) for g in plateau_rows) / len(plateau_rows)
        if plateau_rows
        else None
    )
    cliff_s = None
    for g in gens:
        if float(g["decode_tok_s"]) < 0.8 * fresh:
            cliff_s = float(g["t_start_s"])
            break
    out: dict[str, Any] = {
        "generations": len(gens),
        "elapsed_s": round(total_s, 1),
        "fresh_tok_s": round(fresh, 2),
        "cliff_s": round(cliff_s, 1) if cliff_s is not None else None,
    }
    if plateau is not None:
        out["plateau_tok_s"] = round(plateau, 2)
        out["throttle_factor"] = round(fresh / plateau, 3) if plateau > 0 else None
    return out


def _sustained_summary_markdown(result: dict[str, Any]) -> str:
    s = result.get("sustained", {})
    cfg = result.get("config", {})
    lines = [
        "# DFlash sustained benchmark",
        "",
        f"- model: {cfg.get('model')}",
        f"- draft: {cfg.get('draft')}",
        f"- minutes: {cfg.get('sustained_minutes')}",
        f"- generations: {s.get('generations')}  (elapsed {s.get('elapsed_s')}s)",
        f"- fresh decode: {s.get('fresh_tok_s')} tok/s",
        f"- plateau decode (last 5 min): {s.get('plateau_tok_s')} tok/s",
        f"- throttle factor (fresh/plateau): {s.get('throttle_factor')}",
        f"- cliff onset (first gen < 0.8x fresh): {s.get('cliff_s')}s",
        "",
        "| gen | t_start_s | decode_tok_s | tokens | tpc |",
        "| --- | --- | --- | --- | --- |",
    ]
    for g in result.get("runs", []):
        lines.append(
            f"| {g['gen_index']} | {g['t_start_s']:.0f} | {g['decode_tok_s']:.1f} "
            f"| {g['generation_tokens']} | {g.get('tokens_per_cycle', 0.0):.2f} |"
        )
    return "\n".join(lines) + "\n"


def benchmark_sustained(
    *,
    prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Continuous-load leg: one bundle, back-to-back generations for N minutes.

    Single-run benchmarks measure the fresh-GPU regime only; sustained decode
    settles at the power-envelope plateau (measured 1.5-1.8x slower on M5 Max
    after ~3 min). This reports both regimes plus the cliff onset.
    """
    minutes = float(args.sustained_minutes)
    if minutes <= 0:
        raise ValueError("--sustained-minutes must be > 0")
    runtime_values = _offline_runtime_values(
        prefill_step_size=args.prefill_step_size,
        target_fa_window=args.target_fa_window,
        draft_sink_size=args.draft_sink_size,
        draft_window_size=args.draft_window_size,
        verify_len_cap=args.verify_len_cap,
        verify_mode=args.verify_mode,
        copyspec_mode=args.copyspec_mode,
    )
    runtime_context = build_offline_runtime_context(**runtime_values)
    bundle = load_runtime_bundle(
        model_ref=args.model,
        draft_ref=args.draft,
        draft_quant=args.draft_quant,
        verify_config=runtime_context.verify,
    )
    eos_token_ids = get_stop_token_ids(bundle.tokenizer)
    stop_token_ids = [] if args.no_eos else eos_token_ids
    suppress_token_ids = eos_token_ids if args.no_eos else None
    gens: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < minutes * 60.0:
        t_start = time.perf_counter() - t0
        payload = _generate_dflash_stream_once(
            target_model=bundle.target_model,
            target_ops=bundle.target_ops,
            tokenizer=bundle.tokenizer,
            draft_model=bundle.draft_model,
            draft_backend=bundle.draft_backend,
            prompt=prompt,
            max_new_tokens=args.max_tokens,
            use_chat_template=not args.no_chat_template,
            block_tokens=args.block_tokens,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            runtime_context=runtime_context,
        )
        wall_s = (time.perf_counter() - t0) - t_start
        prefill_us = float(dict(payload.get("phase_timings_us", {})).get("prefill", 0.0))
        decode_s = max(float(payload.get("elapsed_us", 0.0)) - prefill_us, 1.0) / 1e6
        tokens = int(payload.get("generation_tokens", 0))
        gens.append(
            {
                "gen_index": len(gens),
                "t_start_s": round(t_start, 2),
                "wall_s": round(wall_s, 2),
                "generation_tokens": tokens,
                "decode_tok_s": round(tokens / decode_s, 2),
                "tokens_per_cycle": float(payload.get("tokens_per_cycle", 0.0)),
                "acceptance_ratio": float(payload.get("acceptance_ratio", 0.0)),
                "thermal_pressure": _get_thermal_pressure(),
            }
        )
    target_meta = bundle.target_meta or {}
    draft_meta = bundle.draft_meta or {}
    return {
        "config": {
            "model": target_meta.get("resolved_model_ref", args.model),
            "draft": draft_meta.get("resolved_model_ref", args.draft),
            "sustained_minutes": minutes,
            "max_tokens": int(args.max_tokens),
            "block_tokens": args.block_tokens,
            "suite": "sustained",
            **runtime_values,
        },
        "sustained": _sustained_summary(gens),
        "runs": gens,
    }


def benchmark_suite(
    *,
    prompts: list[BenchmarkPrompt],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_reports: list[dict[str, Any]] = []
    common_kwargs = {
        "block_tokens": args.block_tokens,
        "use_chat_template": not args.no_chat_template,
        "target_model_ref": args.model,
        "draft_model_ref": args.draft,
        "draft_quant": args.draft_quant,
        "no_eos": args.no_eos,
        "prefill_step_size": args.prefill_step_size,
        "target_fa_window": args.target_fa_window,
        "draft_sink_size": args.draft_sink_size,
        "draft_window_size": args.draft_window_size,
        "verify_len_cap": args.verify_len_cap,
        "verify_mode": args.verify_mode,
        "copyspec_mode": args.copyspec_mode,
        "cooldown": args.cooldown,
        "only_dflash": bool(args.only_dflash),
    }
    for prompt in prompts:
        if args.repeat > 1:
            report = benchmark_repeated(
                prompt=prompt.prompt,
                max_new_tokens=args.max_tokens,
                repeat=args.repeat,
                expected_answer=prompt.expected_answer,
                **common_kwargs,
            )
        else:
            report = benchmark_once(
                prompt=prompt.prompt,
                max_new_tokens=args.max_tokens,
                expected_answer=prompt.expected_answer,
                **common_kwargs,
            )
        report["config"]["prompt_id"] = prompt.id
        report["config"]["prompt_suite"] = prompt.suite
        if prompt.expected_answer is not None:
            report["config"]["expected_answer"] = prompt.expected_answer
        if not args.no_memory:
            _attach_memory_summary(report)
        prompt_reports.append(report)
    return suite_report(
        prompts=prompts,
        prompt_reports=prompt_reports,
        args=args,
        include_memory=not bool(args.no_memory),
    )

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Benchmark baseline MLX vs DFlash MLX runtime.",
        formatter_class=_BenchmarkHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        choices=SUITE_CHOICES,
        default="smoke",
        help=(
            "Named benchmark prompt suite. Default: smoke.\n"
            "humaneval/gsm8k/math500/aime25 load HF datasets; aime25 also "
            "records exact-answer score fields."
        ),
    )
    parser.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Number of prompts to run from the selected suite. Default: 1 for smoke/longctx, 30 for aime25, 10 otherwise.",
    )
    parser.add_argument(
        "--ctx-tokens",
        metavar="N",
        type=int,
        default=None,
        help="Synthetic long-context token target for --suite longctx. Default: 8192.",
    )
    parser.add_argument(
        "--prompt-file",
        metavar="PATH",
        default=None,
        help='JSONL prompt file override with rows like {"id":"...","suite":"...","prompt":"..."}.',
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle HF dataset rows before --limit selection. Default: disabled.",
    )
    parser.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        default=0,
        help="Shuffle seed used only with --shuffle. Default: 0.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=f"Prompt to benchmark. Default: {DEFAULT_PROMPT!r}.",
    )
    parser.add_argument(
        "--max-tokens",
        metavar="INT",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Number of tokens to generate. Default: 64, or 65536 for --suite aime25.",
    )
    parser.add_argument(
        "--block-tokens",
        metavar="INT",
        type=int,
        default=16,
        help="DFlash speculative verify block size. Default: 16.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Omit peak memory medians from the summary. Default: memory summary enabled.",
    )
    parser.add_argument(
        "--repeat",
        metavar="INT",
        type=int,
        default=None,
        help="Number of measured runs. Default: 1.",
    )
    parser.add_argument(
        "--cooldown",
        metavar="SECONDS",
        type=int,
        default=10,
        help="Sleep between baseline/DFlash legs and repeated measured runs. Default: 10.",
    )
    parser.add_argument(
        "--sustained-minutes",
        metavar="MINUTES",
        type=float,
        default=None,
        help=(
            "Continuous-load mode: one model load, back-to-back DFlash "
            "generations for N minutes; reports fresh vs plateau decode tok/s "
            "and the throttle cliff onset. Default: disabled."
        ),
    )
    parser.add_argument(
        "--wired-limit",
        metavar="auto|none|BYTES",
        type=parse_memory_limit,
        default="auto",
        help=(
            "MLX wired memory limit. Default: auto. "
            "Use none to skip mx.set_wired_limit; BYTES accepts suffixes like 48GB."
        ),
    )
    parser.add_argument(
        "--cache-limit",
        metavar="auto|none|BYTES",
        type=parse_memory_limit,
        default=None,
        help=(
            "MLX cache memory limit. Default: 4GB for longctx, auto otherwise. "
            "Use none to skip mx.set_cache_limit; BYTES accepts suffixes like 4GB."
        ),
    )
    parser.add_argument(
        "--model",
        metavar="HF_REF_OR_PATH",
        default=None,
        help="Target model. Required.",
    )
    parser.add_argument(
        "--draft",
        metavar="HF_REF_OR_PATH",
        default=None,
        help="DFlash draft model. Default: auto-resolved from target.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable tokenizer chat template. Default: chat template enabled.",
    )
    parser.add_argument(
        "--draft-quant",
        metavar="SPEC",
        default=None,
        help="Draft quantization override, e.g. w4:gs64; use 'none' to disable model defaults.",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Suppress EOS so generation reaches --max-tokens. Default: EOS enabled.",
    )
    parser.add_argument(
        "--only-dflash",
        action="store_true",
        help="Skip the baseline MLX run and measure DFlash only. Default: baseline then DFlash.",
    )
    add_offline_runtime_arguments(parser, BENCHMARK_RUNTIME_FIELDS)
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help=(
            "Artifact output directory. Default: "
            ".artifacts/dflash/benchmarks/<timestamp>-<suite>-<model>."
        ),
    )
    return parser

def _controlled_flag_values(
    args: argparse.Namespace,
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    ctx_tokens = int(config.get("ctx_tokens", _ctx_tokens(args)))
    return {
        "suite": config.get("suite", args.suite),
        "limit": int(config.get("limit", args.limit if args.limit is not None else _default_limit_for_suite(args.suite))),
        "ctx_tokens": ctx_tokens,
        "prompt_file": config.get("prompt_file", args.prompt_file),
        "shuffle": bool(config.get("shuffle", args.shuffle)),
        "seed": int(config.get("seed", args.seed)),
        "prompt": args.prompt,
        "max_tokens": int(args.max_tokens),
        "block_tokens": int(args.block_tokens),
        "include_memory": not bool(args.no_memory),
        "no_memory": bool(args.no_memory),
        "repeat": int(args.repeat),
        "cooldown": int(args.cooldown),
        "wired_limit": config.get("wired_limit", args.wired_limit),
        "cache_limit": config.get("cache_limit", args.cache_limit),
        "model": config.get("model", args.model),
        "draft": config.get("draft", args.draft),
        "use_chat_template": not bool(args.no_chat_template),
        "draft_quant": config.get("draft_quant", args.draft_quant),
        "no_eos": bool(args.no_eos),
        "prefill_step_size": int(args.prefill_step_size),
        "target_fa_window": int(args.target_fa_window),
        "draft_sink_size": int(args.draft_sink_size),
        "draft_window_size": int(args.draft_window_size),
        "verify_len_cap": int(args.verify_len_cap),
        "verify_mode": str(args.verify_mode),
        "copyspec_mode": str(args.copyspec_mode),
        "only_dflash": bool(args.only_dflash),
        "out": str(output_path),
    }

def _explicit_flag_values(
    args: argparse.Namespace,
    argv: list[str],
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    option_to_name = {
        "--suite": "suite",
        "--limit": "limit",
        "--ctx-tokens": "ctx_tokens",
        "--prompt-file": "prompt_file",
        "--shuffle": "shuffle",
        "--seed": "seed",
        "--prompt": "prompt",
        "--max-tokens": "max_tokens",
        "--block-tokens": "block_tokens",
        "--no-memory": "no_memory",
        "--repeat": "repeat",
        "--cooldown": "cooldown",
        "--wired-limit": "wired_limit",
        "--cache-limit": "cache_limit",
        "--model": "model",
        "--draft": "draft",
        "--no-chat-template": "use_chat_template",
        "--draft-quant": "draft_quant",
        "--no-eos": "no_eos",
        "--prefill-step-size": "prefill_step_size",
        "--target-fa-window": "target_fa_window",
        "--draft-sink-size": "draft_sink_size",
        "--draft-window-size": "draft_window_size",
        "--verify-len-cap": "verify_len_cap",
        "--verify-mode": "verify_mode",
        "--copyspec-mode": "copyspec_mode",
        "--only-dflash": "only_dflash",
        "--out": "out",
    }
    seen = {option_to_name[token] for token in argv if token in option_to_name}
    values = _controlled_flag_values(args, output_path, config)
    if "draft_quant" in seen:
        values["draft_quant"] = args.draft_quant
    return {name: values[name] for name in CONTROLLED_FLAG_NAMES if name in seen}

def _build_invocation(
    args: argparse.Namespace,
    output_path: Path,
    argv: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "command": _format_command(argv),
        "argv": argv,
        "output_path": str(output_path),
        "output_dir": str(output_path),
        "explicit_flags": _explicit_flag_values(args, argv[1:], output_path, config),
        "effective": _controlled_flag_values(args, output_path, config),
        "protocol_order": ["dflash"] if bool(args.only_dflash) else ["baseline", "dflash"],
        "same_prompt_token_ids": None if bool(args.only_dflash) else True,
        "primary_metric": "post_prefill_generation_tps",
    }

def _format_command(argv: list[str]) -> str:
    if not argv:
        return ""
    if " " in argv[0]:
        tail = shlex.join(argv[1:])
        return argv[0] if not tail else f"{argv[0]} {tail}"
    return shlex.join(argv)

def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = build_parser(prog=prog)
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    try:
        args = _finalize_benchmark_args(parser.parse_args(argv_list), argv_list)
    except ValueError as exc:
        parser.error(offline_runtime_error_message(str(exc)))
    if args.model is None:
        parser.error("--model is required")
    args.metal_limits = apply_metal_limits(
        wired_request=args.wired_limit,
        cache_request=args.cache_limit,
    )
    prompts = resolve_benchmark_prompts(args)
    output_path = create_run_dir(
        "benchmark",
        _benchmark_label(args),
        explicit_path=args.out,
    )
    if args.sustained_minutes is not None:
        try:
            result = benchmark_sustained(prompt=prompts[0].prompt, args=args)
        except ValueError as exc:
            parser.error(str(exc))
        command_argv = (
            [prog or "dflash benchmark", *argv_list] if argv is not None else list(sys.argv)
        )
        manifest = write_manifest(
            output_path,
            kind="benchmark",
            label=_benchmark_label(args),
            argv=command_argv,
            model=result["config"].get("model"),
            draft=result["config"].get("draft"),
            effective_config=result["config"],
        )
        manifest["benchmark_summary"] = result.get("sustained")
        write_json(output_path / "manifest.json", manifest)
        write_json(output_path / "results.json", result)
        write_jsonl(output_path / "runs.jsonl", list(result.get("runs", [])))
        (output_path / "summary.md").write_text(_sustained_summary_markdown(result))
        print(json.dumps(result.get("sustained", {}), indent=2))
        print(f"\nArtifacts: {output_path}")
        return
    try:
        result = benchmark_suite(prompts=prompts, args=args)
    except ValueError as exc:
        parser.error(str(exc))
    command_argv = [prog or "dflash benchmark", *argv_list] if argv is not None else list(sys.argv)
    result["invocation"] = _build_invocation(args, output_path, command_argv, result["config"])
    manifest = write_manifest(
        output_path,
        kind="benchmark",
        label=_benchmark_label(args),
        argv=command_argv,
        model=result["config"].get("model"),
        draft=result["config"].get("draft"),
        effective_config=result["config"],
    )
    manifest.update(
        {
            "suite": result["config"].get("suite"),
            "prompt_ids": result["config"].get("prompt_ids"),
            "prompt_count": result["config"].get("prompt_count"),
            "ctx_tokens": result["config"].get("ctx_tokens"),
            "prompt_file": result["config"].get("prompt_file"),
            "prompt_source": result["config"].get("prompt_source"),
            "hf_dataset_name": result["config"].get("hf_dataset_name"),
            "hf_dataset_config": result["config"].get("hf_dataset_config"),
            "hf_dataset_split": result["config"].get("hf_dataset_split"),
            "hf_shuffle_seed": result["config"].get("hf_shuffle_seed"),
            "shuffle": result["config"].get("shuffle"),
            "seed": result["config"].get("seed"),
            "selected_row_indices": result["config"].get("selected_row_indices"),
            "prompt_tokenization_mode": result["config"].get("prompt_tokenization_mode"),
            "machine_summary": result.get("hardware"),
            "benchmark_summary": result.get("summary"),
        }
    )
    write_json(output_path / "manifest.json", manifest)
    write_json(output_path / "invocation.json", result["invocation"])
    write_json(output_path / "results.json", result)
    write_json(output_path / "summary.json", result)
    write_jsonl(output_path / "runs.jsonl", list(result.get("runs", [])))
    (output_path / "summary.md").write_text(summary_markdown(result))
    print_summary(result, output_path)

if __name__ == "__main__":
    main()

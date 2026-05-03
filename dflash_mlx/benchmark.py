# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import argparse
import gc
import os
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
)
from dflash_mlx.metal_limits import apply_metal_limits
from dflash_mlx.runtime import (
    get_stop_token_ids,
    load_draft_bundle,
    load_target_bundle,
    resolve_model_ref,
    stream_dflash_generate,
)
from dflash_mlx.runtime_context import (
    RuntimeContext,
    build_offline_runtime_context,
)

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
    "ctx",
    "include_memory",
    "no_memory",
    "repeat",
    "cooldown",
    "model",
    "draft",
    "use_chat_template",
    "draft_quant",
    "no_eos",
    "split_sdpa",
    "target_fa_window",
    "draft_sink_size",
    "draft_window_size",
    "verify_len_cap",
    "out",
)

class _BenchmarkHelpFormatter(argparse.RawTextHelpFormatter):
    def __init__(self, prog: str):
        super().__init__(prog, max_help_position=28, width=100)

def _git_hash_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
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
    except Exception:
        pass
    return "unknown"

def _warn_if_throttled(thermal_pressure: str) -> None:
    if thermal_pressure == "nominal":
        return
    print(
        f"WARNING: thermal pressure is '{thermal_pressure}' — results may be throttled. "
        "Increase --cooldown or wait for chip to cool.",
        file=sys.stderr,
    )

def _slugify_model_ref(model_ref: str | None) -> str:
    resolved = resolve_model_ref(model_ref, kind="target")
    label = Path(str(resolved)).name or str(resolved)
    label = re.sub(r"[^a-z0-9]+", "-", label.lower())
    label = re.sub(r"-+", "-", label).strip("-")
    return label or "model"

def _slugify_prompt_id(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt[:60].lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "prompt"


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
    if args.repeat is None:
        args.repeat = 1
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.ctx < 0:
        raise ValueError("--ctx must be >= 0")
    if args.ctx_tokens is not None and args.ctx_tokens < 0:
        raise ValueError("--ctx-tokens must be >= 0")
    if args.ctx_tokens is None and args.ctx > 0:
        args.ctx_tokens = int(args.ctx)
    if args.ctx_tokens is not None and args.suite == "smoke" and not suite_explicit:
        args.suite = "longctx"
    if args.suite == "longctx" and (args.ctx_tokens is None or args.ctx_tokens == 0):
        args.ctx_tokens = DEFAULT_CTX_TOKENS
    if args.ctx_tokens is None:
        args.ctx_tokens = 0
    args.ctx = int(args.ctx_tokens)
    args.seed = int(args.seed)
    if args.limit is None:
        args.limit = _default_limit_for_suite(args.suite)
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.target_fa_window < 0:
        raise ValueError("--target-fa-window must be >= 0")
    if args.draft_sink_size < 0:
        raise ValueError("--draft-sink-size must be >= 0")
    if args.draft_window_size <= 0:
        raise ValueError("--draft-window-size must be > 0")
    if args.verify_len_cap < 0:
        raise ValueError("--verify-len-cap must be >= 0")
    return args

def _strip_generation_payload(result: dict[str, Any], *, drop_phase_timings: bool = False) -> dict[str, Any]:
    cleaned = dict(result)
    cleaned.pop("generated_token_ids", None)
    if drop_phase_timings:
        phase_timings = dict(cleaned.pop("phase_timings_us", {}) or {})
        if "prefill" in phase_timings and "prefill_us" not in cleaned:
            cleaned["prefill_us"] = float(phase_timings["prefill"])
    return cleaned

def _format_run_entry(run: dict[str, Any]) -> dict[str, Any]:
    baseline = dict(run["baseline"])
    dflash = dict(run["dflash"])
    return {
        "run": int(run["run_index"]),
        "thermal_pressure": str(run.get("thermal_pressure", "unknown")),
        "baseline": {
            "ttft_ms": float(run["baseline_ttft_ms"]),
            "generation_tps": float(run["baseline_generation_tps"]),
            "peak_memory_gb": baseline.get("peak_memory_gb"),
        },
        "dflash": {
            "ttft_ms": float(run["dflash_ttft_ms"]),
            "generation_tps": float(run["dflash_generation_tps"]),
            "tokens_per_cycle": float(dflash.get("tokens_per_cycle", 0.0)),
            "cycles": int(dflash.get("cycles_completed", 0)),
            "acceptance_ratio": float(dflash.get("acceptance_ratio", 0.0)),
            "acceptance_first_20_avg": float(dflash.get("acceptance_first_20_avg", 0.0)),
            "acceptance_last_20_avg": float(dflash.get("acceptance_last_20_avg", 0.0)),
            "peak_memory_gb": dflash.get("peak_memory_gb"),
        },
        "speedup": float(run["generation_speedup_vs_baseline"]) if run["generation_speedup_vs_baseline"] is not None else None,
    }

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
    no_eos: bool,
    split_sdpa: bool,
    target_fa_window: int,
    draft_sink_size: int,
    draft_window_size: int,
    verify_len_cap: int,
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
        "no_eos": bool(no_eos),
        "split_sdpa": bool(split_sdpa),
        "target_fa_window": int(target_fa_window),
        "draft_sink_size": int(draft_sink_size),
        "draft_window_size": int(draft_window_size),
        "verify_len_cap": int(verify_len_cap),
        "prompt": prompt,
        "prompt_tokens": int(prompt_tokens),
        "prompt_id": _slugify_prompt_id(prompt),
        "git_hash": _git_hash_short(),
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
    no_eos: bool,
    split_sdpa: bool,
    target_fa_window: int,
    draft_sink_size: int,
    draft_window_size: int,
    verify_len_cap: int,
) -> dict[str, Any]:
    run_entries = [_format_run_entry(run) for run in runs]
    baseline_tps_values = [float(run["baseline_generation_tps"]) for run in runs]
    dflash_tps_values = [float(run["dflash_generation_tps"]) for run in runs]
    speedup_values = [float(run["generation_speedup_vs_baseline"]) for run in runs if run["generation_speedup_vs_baseline"] is not None]
    acceptance_ratio_values = [float(run["dflash"]["acceptance_ratio"]) for run in runs]
    prompt_tokens = int(runs[0]["baseline"]["prompt_token_count"]) if runs else 0
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
            no_eos=no_eos,
            split_sdpa=split_sdpa,
            target_fa_window=target_fa_window,
            draft_sink_size=draft_sink_size,
            draft_window_size=draft_window_size,
            verify_len_cap=verify_len_cap,
        ),
        "runs": run_entries,
        "summary": {
            "baseline_tps_median": statistics.median(baseline_tps_values) if baseline_tps_values else None,
            "dflash_tps_median": statistics.median(dflash_tps_values) if dflash_tps_values else None,
            "dflash_tps_min": min(dflash_tps_values) if dflash_tps_values else None,
            "dflash_tps_max": max(dflash_tps_values) if dflash_tps_values else None,
            "speedup_median": statistics.median(speedup_values) if speedup_values else None,
            "acceptance_ratio_median": statistics.median(acceptance_ratio_values) if acceptance_ratio_values else None,
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
    result.setdefault("summary", {}).update(
        {
            "baseline_peak_memory_gb_median": statistics.median(baseline_peaks)
            if baseline_peaks
            else None,
            "dflash_peak_memory_gb_median": statistics.median(dflash_peaks)
            if dflash_peaks
            else None,
        }
    )

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

def _generate_stock_baseline_once(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    no_eos: bool,
    use_chat_template: bool = True,
    prompt_tokens_override: list[int] | None = None,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

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
        except Exception:
            tokenizer.eos_token_ids = []
        try:
            tokenizer.eos_token_id = None
        except Exception:
            pass

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
            "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
        }

    prompt_tokens = int(final_response.prompt_tokens)
    prompt_tps = float(final_response.prompt_tps)
    generation_tokens = int(final_response.generation_tokens)
    generation_tps = float(final_response.generation_tps)
    prefill_us = (prompt_tokens / prompt_tps) * 1e6 if prompt_tps > 0.0 else 0.0
    generation_us = (generation_tokens / generation_tps) * 1e6 if generation_tps > 0.0 else 0.0
    return {
        "elapsed_us": elapsed_us,
        "prefill_us": prefill_us,
        "prompt_token_count": prompt_tokens,
        "generated_token_ids": generated_token_ids,
        "generation_tokens": generation_tokens,
        "generation_tps": generation_tps,
        "peak_memory_gb": float(final_response.peak_memory),
    }

def _generate_dflash_stream_once(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool,
    block_tokens: int | None,
    stop_token_ids: list[int] | None,
    suppress_token_ids: list[int] | None,
    runtime_context: RuntimeContext,
    prompt_tokens_override: list[int] | None = None,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass

    start_ns = time.perf_counter_ns()
    first_token_us: float | None = None
    summary: dict[str, Any] | None = None
    stream = stream_dflash_generate(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
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
            event_type = event.get("event")
            if event_type == "token" and first_token_us is None:
                first_token_us = (time.perf_counter_ns() - start_ns) / 1_000.0
            elif event_type == "summary":
                summary = dict(event)
    finally:
        stream.close()

    if summary is None:
        raise RuntimeError("DFlash stream did not yield a summary event")

    summary["ttft_us"] = (
        first_token_us
        if first_token_us is not None
        else float(dict(summary.get("phase_timings_us", {})).get("prefill", 0.0))
    )
    return summary

def _release_loaded_models() -> None:
    gc.collect()
    if hasattr(mx, "clear_cache"):
        try:
            mx.clear_cache()
            return
        except Exception:
            pass
    if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

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
    split_sdpa: bool,
    target_fa_window: int = 0,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
    verify_len_cap: int = 0,
) -> dict[str, Any]:
    pristine_target_model, pristine_tokenizer, pristine_meta = _load_pristine_target_bundle(
        target_model_ref
    )

    if use_chat_template and hasattr(pristine_tokenizer, "apply_chat_template"):
        prompt_tokens = list(
            pristine_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    else:
        prompt_tokens = list(pristine_tokenizer.encode(prompt))
    try:
        baseline = _generate_stock_baseline_once(
            target_model=pristine_target_model,
            tokenizer=pristine_tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            no_eos=no_eos,
            use_chat_template=use_chat_template,
            prompt_tokens_override=prompt_tokens,
        )
    finally:
        del pristine_target_model
        del pristine_tokenizer
        _release_loaded_models()

    target_model = None
    tokenizer = None
    draft_model = None
    try:
        runtime_context = build_offline_runtime_context(
            target_fa_window=target_fa_window,
            draft_sink_size=draft_sink_size,
            draft_window_size=draft_window_size,
            verify_len_cap=verify_len_cap,
        )
        target_model, tokenizer, target_meta = load_target_bundle(
            target_model_ref,
            lazy=True,
            split_full_attention_sdpa=split_sdpa,
            verify_config=runtime_context.verify,
        )
        draft_model, draft_meta = load_draft_bundle(
            draft_model_ref,
            lazy=True,
            draft_quant=draft_quant,
        )

        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            dflash_prompt_tokens = list(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        else:
            dflash_prompt_tokens = list(tokenizer.encode(prompt))
        assert prompt_tokens == dflash_prompt_tokens, (
            f"Tokenizer drift between pristine and DFlash bundles: "
            f"{len(prompt_tokens)} vs {len(dflash_prompt_tokens)} tokens"
        )
        dflash_eos_token_ids = get_stop_token_ids(tokenizer)
        dflash_stop_token_ids = [] if no_eos else dflash_eos_token_ids
        dflash_suppress_token_ids = dflash_eos_token_ids if no_eos else None
        dflash = _generate_dflash_stream_once(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            block_tokens=block_tokens,
            stop_token_ids=dflash_stop_token_ids,
            suppress_token_ids=dflash_suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            runtime_context=runtime_context,
        )
    finally:
        if target_model is not None:
            del target_model
        if tokenizer is not None:
            del tokenizer
        if draft_model is not None:
            del draft_model
        _release_loaded_models()

    baseline_elapsed = float(baseline["elapsed_us"])
    dflash_elapsed = float(dflash["elapsed_us"])
    baseline_generation_tps = _generation_tps_from_baseline(baseline)
    dflash_generation_tps = _generation_tps_from_dflash(dflash)
    return {
        "baseline": _strip_generation_payload(baseline),
        "dflash": _strip_generation_payload(dflash, drop_phase_timings=True),
        "speedup_vs_baseline": _speedup(baseline_elapsed, dflash_elapsed),
        "baseline_ttft_ms": _ttft_ms_from_baseline(baseline),
        "dflash_ttft_ms": _ttft_ms_from_dflash(dflash),
        "baseline_generation_tps": baseline_generation_tps,
        "dflash_generation_tps": dflash_generation_tps,
        "generation_speedup_vs_baseline": _generation_speedup(
            baseline_generation_tps,
            dflash_generation_tps,
        ),
        "token_match": baseline["generated_token_ids"] == dflash["generated_token_ids"],
        "target_meta": target_meta,
        "draft_meta": draft_meta,
        "pristine_target_meta": pristine_meta,
    }

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
    split_sdpa: bool = True,
    target_fa_window: int = 0,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
    verify_len_cap: int = 0,
    cooldown: int = 10,
) -> dict[str, Any]:
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
        split_sdpa=split_sdpa,
        target_fa_window=target_fa_window,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
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
        draft_quant=draft_quant,
        no_eos=no_eos,
        split_sdpa=split_sdpa,
        target_fa_window=target_fa_window,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
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
    split_sdpa: bool = True,
    target_fa_window: int = 0,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
    verify_len_cap: int = 0,
    cooldown: int = 10,
) -> dict[str, Any]:
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
            split_sdpa=split_sdpa,
            target_fa_window=target_fa_window,
            draft_sink_size=draft_sink_size,
            draft_window_size=draft_window_size,
            verify_len_cap=verify_len_cap,
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
        draft_quant=draft_quant,
        no_eos=no_eos,
        split_sdpa=split_sdpa,
        target_fa_window=target_fa_window,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
    )

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
        "split_sdpa": args.split_sdpa,
        "target_fa_window": args.target_fa_window,
        "draft_sink_size": args.draft_sink_size,
        "draft_window_size": args.draft_window_size,
        "verify_len_cap": args.verify_len_cap,
        "cooldown": args.cooldown,
    }
    for prompt in prompts:
        if args.repeat > 1:
            report = benchmark_repeated(
                prompt=prompt.prompt,
                max_new_tokens=args.max_tokens,
                repeat=args.repeat,
                **common_kwargs,
            )
        else:
            report = benchmark_once(
                prompt=prompt.prompt,
                max_new_tokens=args.max_tokens,
                **common_kwargs,
            )
        report["config"]["prompt_id"] = prompt.id
        report["config"]["prompt_suite"] = prompt.suite
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
            "Named runtime prompt suite. Default: smoke.\n"
            "humaneval/gsm8k/math500 load HF datasets for runtime measurement, "
            "not official accuracy scores."
        ),
    )
    parser.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Number of prompts to run from the selected suite. Default: 1 for smoke/longctx, 10 otherwise.",
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
        default=64,
        help="Number of tokens to generate. Default: 64.",
    )
    parser.add_argument(
        "--block-tokens",
        metavar="INT",
        type=int,
        default=16,
        help="DFlash speculative verify block size. Default: 16.",
    )
    parser.add_argument(
        "--ctx",
        metavar="INT",
        type=int,
        default=0,
        help="Existing shorthand for --ctx-tokens. Default: 0.",
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
        help="Sleep between measured runs. Default: 10.",
    )
    parser.add_argument(
        "--model",
        metavar="HF_REF_OR_PATH",
        default=None,
        help="Target model. Default: auto-resolved default target.",
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
        help="Optional in-memory draft quantization, e.g. w4:gs64.",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        help="Suppress EOS so generation reaches --max-tokens. Default: EOS enabled.",
    )
    parser.add_argument(
        "--split-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable split full-attention SDPA on target. Default: enabled.",
    )
    parser.add_argument(
        "--target-fa-window",
        metavar="INT",
        type=int,
        default=0,
        help="Experimental verifier FA KV window. Default: 0 = full KV.",
    )
    parser.add_argument(
        "--draft-sink-size",
        metavar="INT",
        type=int,
        default=64,
        help="Draft context cache sink tokens. Default: 64.",
    )
    parser.add_argument(
        "--draft-window-size",
        metavar="INT",
        type=int,
        default=1024,
        help="Draft context cache rolling window tokens. Default: 1024.",
    )
    parser.add_argument(
        "--verify-len-cap",
        metavar="INT",
        type=int,
        default=0,
        help="Max tokens verified per target forward. Default: 0 = block size.",
    )
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
        "ctx": ctx_tokens,
        "include_memory": not bool(args.no_memory),
        "no_memory": bool(args.no_memory),
        "repeat": int(args.repeat),
        "cooldown": int(args.cooldown),
        "model": config.get("model", args.model),
        "draft": config.get("draft", args.draft),
        "use_chat_template": not bool(args.no_chat_template),
        "draft_quant": args.draft_quant,
        "no_eos": bool(args.no_eos),
        "split_sdpa": bool(args.split_sdpa),
        "target_fa_window": int(args.target_fa_window),
        "draft_sink_size": int(args.draft_sink_size),
        "draft_window_size": int(args.draft_window_size),
        "verify_len_cap": int(args.verify_len_cap),
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
        "--ctx": "ctx",
        "--no-memory": "no_memory",
        "--repeat": "repeat",
        "--cooldown": "cooldown",
        "--model": "model",
        "--draft": "draft",
        "--no-chat-template": "use_chat_template",
        "--draft-quant": "draft_quant",
        "--no-eos": "no_eos",
        "--split-sdpa": "split_sdpa",
        "--no-split-sdpa": "split_sdpa",
        "--target-fa-window": "target_fa_window",
        "--draft-sink-size": "draft_sink_size",
        "--draft-window-size": "draft_window_size",
        "--verify-len-cap": "verify_len_cap",
        "--out": "out",
    }
    seen = {option_to_name[token] for token in argv if token in option_to_name}
    values = _controlled_flag_values(args, output_path, config)
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
        "protocol_order": ["baseline", "dflash"],
        "same_prompt_token_ids": True,
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
    apply_metal_limits()
    parser = build_parser(prog=prog)
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = _finalize_benchmark_args(parser.parse_args(argv_list), argv_list)
    prompts = resolve_benchmark_prompts(args)
    output_path = create_run_dir(
        "benchmark",
        _benchmark_label(args),
        explicit_path=args.out,
    )
    result = benchmark_suite(prompts=prompts, args=args)
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

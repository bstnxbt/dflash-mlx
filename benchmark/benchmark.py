# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)


import argparse
import gc
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import stream_generate as mlx_stream_generate
from mlx_lm.utils import load as load_pristine_target

from dflash_mlx.runtime import (
    load_draft_bundle,
    load_target_bundle,
    resolve_model_ref,
    stream_dflash_generate,
)

DEFAULT_SCHEDULES: tuple[int, ...] = (8, 16, 32)
DEFAULT_REPEAT = 3


def _git_hash_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
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


def _slugify_prompt_id(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug[:48] or "prompt"


def _slugify_model_ref(model_ref: str | None) -> str:
    resolved = resolve_model_ref(model_ref, kind="target")
    label = Path(str(resolved)).name or str(resolved)
    label = re.sub(r"[^a-z0-9]+", "-", label.lower())
    label = re.sub(r"-+", "-", label).strip("-")
    return label or "model"


def _default_results_path(*, target_model_ref: str | None, max_new_tokens: int) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"{_slugify_model_ref(target_model_ref)}-{int(max_new_tokens)}-{ts}"
    return Path("benchmark/results") / f"{name}.json"


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
    target_model: str,
    draft_model: str,
) -> dict[str, Any]:
    return {
        "target_model": target_model,
        "draft_model": draft_model,
        "max_new_tokens": int(max_new_tokens),
        "block_tokens": int(block_tokens),
        "cooldown": int(cooldown),
        "prompt": prompt,
        "prompt_tokens": int(prompt_tokens),
        "prompt_id": _slugify_prompt_id(prompt),
        "repeat": int(repeat),
        "git_hash": _git_hash_short(),
    }


def _build_single_case_report(
    *,
    prompt: str,
    max_new_tokens: int,
    repeat: int,
    cooldown: int,
    runs: list[dict[str, Any]],
    target_model: str,
    draft_model: str,
) -> dict[str, Any]:
    run_entries = [_format_run_entry(run) for run in runs]
    baseline_tps_values = [float(run["baseline_generation_tps"]) for run in runs]
    dflash_tps_values = [float(run["dflash_generation_tps"]) for run in runs]
    speedup_values = [float(run["generation_speedup_vs_baseline"]) for run in runs if run["generation_speedup_vs_baseline"] is not None]
    acceptance_ratio_values = [float(run["dflash"]["acceptance_ratio"]) for run in runs]
    prompt_tokens = int(runs[0]["baseline"]["prompt_token_count"]) if runs else 0
    effective_block_tokens = (
        int(runs[0]["dflash"].get("block_tokens"))
        if runs and runs[0].get("dflash", {}).get("block_tokens") is not None
        else None
    )
    return {
        "hardware": _hardware_info(),
        "config": _build_config(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            block_tokens=effective_block_tokens,
            repeat=repeat,
            cooldown=cooldown,
            target_model=target_model,
            draft_model=draft_model,
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


def get_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_token_ids = list(getattr(tokenizer, "eos_token_ids", None) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in eos_token_ids:
        eos_token_ids.append(int(eos_token_id))
    return eos_token_ids


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

    # Pre-tokenized input: feed token IDs directly (mlx_stream_generate accepts List[int]).
    # This guarantees baseline + DFlash see byte-identical prompt tokens.
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
    quantize_draft: bool,
    no_eos: bool,
    split_sdpa: bool,
) -> dict[str, Any]:
    pristine_target_model, pristine_tokenizer, pristine_meta = _load_pristine_target_bundle(
        target_model_ref
    )
    # Pre-tokenize ONCE with pristine tokenizer. Same token IDs feed both paths —
    # no reliance on str→encode round-trip equality across tokenizer versions.
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

    target_model, tokenizer, target_meta = load_target_bundle(
        target_model_ref,
        lazy=True,
        split_full_attention_sdpa=split_sdpa,
    )
    draft_model, draft_meta = load_draft_bundle(
        draft_model_ref,
        lazy=True,
        quantize_draft=quantize_draft,
    )
    # Cross-check: DFlash tokenizer is a different instance; tokens must match.
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
    try:
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
        )
    finally:
        del target_model
        del tokenizer
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
    quantize_draft: bool = False,
    no_eos: bool = False,
    split_sdpa: bool = True,
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
        quantize_draft=quantize_draft,
        no_eos=no_eos,
        split_sdpa=split_sdpa,
    )
    target_meta = result.pop("target_meta")
    draft_meta = result.pop("draft_meta")
    result.pop("pristine_target_meta", None)
    result["run_index"] = 1
    result["thermal_pressure"] = thermal_pressure
    return _build_single_case_report(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        repeat=1,
        cooldown=cooldown,
        runs=[result],
        target_model=target_meta["resolved_model_ref"],
        draft_model=draft_meta["resolved_model_ref"],
    )


def benchmark_matrix(
    *,
    prompts: tuple[str, ...] = (),
    schedules: tuple[int, ...] = DEFAULT_SCHEDULES,
    repeat: int = DEFAULT_REPEAT,
    block_tokens: int | None = None,
    use_chat_template: bool = False,
    target_model_ref: str | None = None,
    draft_model_ref: str | None = None,
    quantize_draft: bool = False,
    no_eos: bool = False,
    split_sdpa: bool = True,
    cooldown: int = 10,
) -> dict[str, Any]:
    target_meta: dict[str, Any] | None = None
    draft_meta: dict[str, Any] | None = None
    if len(prompts) != 1 or len(schedules) != 1:
        raise ValueError("benchmark_matrix currently expects exactly one prompt and one schedule.")
    prompt = prompts[0]
    max_new_tokens = schedules[0]
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
            quantize_draft=quantize_draft,
            no_eos=no_eos,
            split_sdpa=split_sdpa,
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
        repeat=repeat,
        cooldown=cooldown,
        runs=runs,
        target_model=target_meta["resolved_model_ref"] if target_meta is not None else "",
        draft_model=draft_meta["resolved_model_ref"] if draft_meta is not None else "",
    )


def main() -> None:
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_cache_limit(wired_limit // 4)
    parser = argparse.ArgumentParser(description="Benchmark baseline MLX vs DFlash MLX runtime.")
    parser.add_argument("--prompt", required=True, help="Prompt to benchmark.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Number of measured runs. Uses matrix mode automatically when > 1.",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=10,
        help="Seconds between runs for thermal stabilization.",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--draft", default=None)
    parser.add_argument("--quantize-draft", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument(
        "--split-sdpa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable split_full_attention_sdpa when loading the target model (default: enabled).",
    )
    args = parser.parse_args()
    repeat = args.repeat if args.repeat is not None else (DEFAULT_REPEAT if args.matrix else 1)
    if repeat < 1:
        raise ValueError("--repeat must be >= 1")

    common_kwargs = {
        "block_tokens": args.block_tokens,
        "use_chat_template": not args.no_chat_template,
        "target_model_ref": args.model,
        "draft_model_ref": args.draft,
        "quantize_draft": args.quantize_draft,
        "no_eos": args.no_eos,
        "split_sdpa": args.split_sdpa,
        "cooldown": args.cooldown,
    }
    if args.matrix or repeat > 1:
        result = benchmark_matrix(
            prompts=(args.prompt,),
            schedules=(args.max_tokens,),
            repeat=repeat,
            **common_kwargs,
        )
    else:
        result = benchmark_once(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            **common_kwargs,
        )

    output_path = _default_results_path(
        target_model_ref=args.model,
        max_new_tokens=args.max_tokens,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

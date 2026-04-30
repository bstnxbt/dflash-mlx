# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from dflash_mlx.artifacts import create_run_dir, write_json, write_manifest
from dflash_mlx.diagnostics import (
    DiagnosticsConfig,
    TraceConfig,
)
from dflash_mlx.metal_limits import (
    MetalLimitConfig,
    apply_metal_limits,
    parse_memory_limit,
)
from dflash_mlx.runtime_profiles import (
    format_profiles,
    profile_names,
    resolve_runtime_config,
)
from dflash_mlx.runtime_context import build_runtime_context

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DFlash Server.")
    parser.add_argument(
        "--profile",
        choices=profile_names(),
        default=None,
        help="Runtime preset. Explicit CLI flags override profile values.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List runtime profiles and exit.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Target model repo ID or local path.",
    )
    parser.add_argument("--adapter-path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allowed-origins",
        type=lambda x: x.split(","),
        default="*",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--draft-model",
        "--draft",
        dest="draft_model",
        type=str,
        default=None,
        help="Draft model repo ID or local path. Omit to use the DFlash registry.",
    )
    parser.add_argument(
        "--dflash-max-ctx",
        type=int,
        default=None,
        help="Hard cap on runtime context length.",
    )
    parser.add_argument(
        "--target-fa-window",
        type=int,
        default=None,
        help=(
            "Experimental target verifier full-attention KV window. "
            "0 keeps full KV cache; N>0 uses a rotating KV cache of N tokens "
            "for target full-attention layers only."
        ),
    )
    parser.add_argument(
        "--draft-sink-size",
        type=int,
        default=None,
        help="Draft context cache sink tokens kept before the rolling window.",
    )
    parser.add_argument(
        "--draft-window-size",
        type=int,
        default=None,
        help="Draft context cache rolling window tokens.",
    )
    parser.add_argument(
        "--verify-len-cap",
        type=int,
        default=None,
        help="Max tokens verified per target forward; 0 uses the block size.",
    )
    parser.add_argument(
        "--diagnostics",
        choices=("off", "basic", "full"),
        default="off",
        help=(
            "Product diagnostics mode. basic writes structured request/cache logs; "
            "full also enables memory waterfall and per-cycle profiling."
        ),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=str,
        default=None,
        help="Diagnostics output directory. Default: .artifacts/dflash/diagnostics/<timestamp>-serve-<mode>.",
    )
    parser.add_argument(
        "--wired-limit",
        metavar="auto|none|BYTES",
        type=parse_memory_limit,
        default="auto",
        help=(
            "MLX wired memory limit. auto keeps the current device recommended "
            "limit; none skips mx.set_wired_limit; BYTES accepts suffixes like 48GB."
        ),
    )
    parser.add_argument(
        "--cache-limit",
        metavar="auto|none|BYTES",
        type=parse_memory_limit,
        default="auto",
        help=(
            "MLX cache memory limit. auto keeps the current wired-limit/4 policy; "
            "none skips mx.set_cache_limit; BYTES accepts suffixes like 8GB."
        ),
    )
    parser.add_argument("--num-draft-tokens", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument(
        "--draft-quant",
        default=None,
        metavar="SPEC",
        help="Optional in-memory draft quantization, e.g. w4:gs64.",
    )
    parser.add_argument("--quantize-draft", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trust-remote-code", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Inline chat template override.",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Force the tokenizer default chat template.",
    )
    parser.add_argument("--temp", type=float, default=0.0, help="Default request temperature.")
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling cutoff.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k; 0 disables top-k filtering.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default minimum probability filter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default max generated tokens per request.",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        default="{}",
        help="JSON object passed to the chat-template renderer.",
    )
    parser.add_argument("--decode-concurrency", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--prompt-concurrency", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="Prompt prefill chunk size.",
    )
    parser.add_argument(
        "--clear-cache-boundaries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Clear the MLX cache at safe request boundaries.",
    )
    parser.add_argument(
        "--memory-waterfall",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit memory bucket snapshots in runtime events.",
    )
    parser.add_argument(
        "--bench-log-dir",
        type=str,
        default=None,
        help="Directory for structured JSONL runtime events.",
    )
    parser.add_argument(
        "--verify-mode",
        choices=("auto", "off"),
        default=None,
        help="Verify path mode. Use off only for debug/parity.",
    )
    parser.add_argument(
        "--max-snapshot-tokens",
        type=int,
        default=None,
        help="Skip prefix-cache snapshot inserts above this token count; 0 disables the cap.",
    )
    parser.add_argument(
        "--prefix-cache-l2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable SSD L2 for evicted prefix snapshots.",
    )
    parser.add_argument(
        "--prefix-cache-l2-dir",
        type=str,
        default=None,
        help="Directory for prefix-cache L2 files.",
    )
    parser.add_argument(
        "--prefix-cache-l2-max-bytes",
        type=int,
        default=None,
        help="Byte budget for prefix-cache L2.",
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=10,
        help="mlx_lm prompt-cache entry count.",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=int,
        help="mlx_lm prompt-cache byte cap.",
    )
    parser.add_argument("--pipeline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the DFlash prefix cache that reuses cross-turn KV state. Default: enabled. "
             "Big win on multi-turn agentic workloads, ~neutral on single-turn.",
    )
    parser.add_argument(
        "--prefix-cache-max-entries",
        type=int,
        default=None,
        help="Maximum number of cached prefix snapshots (default: 4).",
    )
    parser.add_argument(
        "--prefix-cache-max-bytes",
        type=int,
        default=None,
        help="Maximum total bytes the prefix cache may hold (default: 8 GiB).",
    )
    return parser

def normalize_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.list_profiles:
        print(format_profiles())
        raise SystemExit(0)
    diagnostics_dir = _configure_diagnostics_args(args)
    if args.bench_log_dir is not None and not args.bench_log_dir.strip():
        raise SystemExit("--bench-log-dir must not be empty")
    try:
        runtime_config = resolve_runtime_config(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if runtime_config.target_fa_window > 0 and (
        args.prefix_cache is not False
        or args.prefix_cache_l2 is True
        or runtime_config.profile == "long-session"
    ):
        sys.stderr.write(
            "[dflash] warning: target FA window disables prefix cache and L2 snapshots\n"
        )
        sys.stderr.flush()
    if diagnostics_dir is not None:
        _write_diagnostics_bootstrap(args, runtime_config, diagnostics_dir)
    diagnostics_config = _build_diagnostics_config(args, runtime_config, diagnostics_dir)
    args.diagnostics_config = diagnostics_config
    args.runtime_config = runtime_config
    args.runtime_context = build_runtime_context(runtime_config, diagnostics_config)
    args.prefill_step_size = runtime_config.prefill_step_size
    return args

def _configure_diagnostics_args(args: argparse.Namespace) -> os.PathLike[str] | None:
    mode = str(getattr(args, "diagnostics", "off") or "off")
    diagnostics_dir = getattr(args, "diagnostics_dir", None)
    if mode == "off":
        if diagnostics_dir is not None:
            raise SystemExit("--diagnostics-dir requires --diagnostics basic or full")
        return None
    if diagnostics_dir is not None and getattr(args, "bench_log_dir", None) is not None:
        raise SystemExit("--diagnostics-dir and --bench-log-dir cannot be used together")
    if getattr(args, "bench_log_dir", None) is not None:
        raise SystemExit("--diagnostics and --bench-log-dir cannot be used together")

    run_dir = create_run_dir(
        "diagnostics",
        f"serve-{mode}",
        explicit_path=diagnostics_dir,
    )
    args.bench_log_dir = str(run_dir)
    args.diagnostics_dir_resolved = str(run_dir)
    if mode == "full":
        args.memory_waterfall = True
    (run_dir / "post_events.jsonl").touch()
    (run_dir / "cache_events.jsonl").touch()
    if mode == "full":
        (run_dir / "cycle_events.jsonl").touch()
    return run_dir

def _build_diagnostics_config(
    args: argparse.Namespace,
    runtime_config,
    diagnostics_dir: os.PathLike[str] | None,
) -> DiagnosticsConfig:
    trace_dir: Path | None = None
    cycle_events = False
    if diagnostics_dir is not None:
        trace_dir = Path(diagnostics_dir)
        cycle_events = getattr(args, "diagnostics", "off") == "full"
    elif runtime_config.bench_log_dir:
        trace_dir = Path(runtime_config.bench_log_dir)
    return DiagnosticsConfig(
        mode=getattr(args, "diagnostics", "off"),
        run_dir=Path(diagnostics_dir) if diagnostics_dir is not None else None,
        memory_waterfall=bool(runtime_config.memory_waterfall),
        trace=TraceConfig(log_dir=trace_dir, cycle_events=cycle_events),
    )

def _write_diagnostics_bootstrap(
    args: argparse.Namespace,
    runtime_config,
    run_dir: os.PathLike[str],
) -> None:
    path = os.fspath(run_dir)
    run_path = os.path.abspath(path)
    effective = asdict(runtime_config)
    effective.update(
        {
            "diagnostics": args.diagnostics,
            "diagnostics_dir": path,
            "profile_cycles": args.diagnostics == "full",
        }
    )
    write_manifest(
        Path(path),
        kind="diagnostics",
        label=f"serve-{args.diagnostics}",
        argv=list(sys.argv),
        model=getattr(args, "model", None),
        draft=getattr(args, "draft_model", None),
        profile=runtime_config.profile,
        effective_config=effective,
    )
    write_json(
        Path(path) / "invocation.json",
        {
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "diagnostics": args.diagnostics,
            "diagnostics_dir": path,
        },
    )
    write_json(Path(path) / "effective_config.json", effective)
    summary = [
        "# DFlash Diagnostics",
        "",
        f"- mode: {args.diagnostics}",
        f"- directory: {run_path}",
        f"- model: {getattr(args, 'model', None) or 'unset'}",
        f"- draft: {getattr(args, 'draft_model', None) or 'auto'}",
        f"- profile: {runtime_config.profile}",
        "",
        "Files:",
        "- `post_events.jsonl`: request timing, throughput, cache hit tokens, and memory peak when enabled.",
        "- `cache_events.jsonl`: prefix-cache lookup/insert/prune events.",
    ]
    if args.diagnostics == "full":
        summary.extend(
            [
                "- `cycle_events.jsonl`: per-cycle timings and memory-waterfall events.",
                "",
                "Full diagnostics has overhead because cycle profiling synchronizes more aggressively.",
            ]
        )
    (Path(path) / "summary.md").write_text("\n".join(summary) + "\n")

def configure_metal_limits(args: argparse.Namespace) -> MetalLimitConfig:
    limits = apply_metal_limits(
        wired_request=getattr(args, "wired_limit", "auto"),
        cache_request=getattr(args, "cache_limit", "auto"),
    )
    args.metal_limits = limits
    return limits

def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

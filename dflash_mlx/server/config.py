# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
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
    MemoryLimit,
    MetalLimitConfig,
    apply_metal_limits,
    parse_memory_limit,
)
from dflash_mlx.runtime.config import (
    GiB,
    add_runtime_config_arguments,
    resolve_runtime_config,
)
from dflash_mlx.runtime.context import build_runtime_context

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DFlash Server.")
    add_runtime_config_arguments(parser)
    parser.add_argument(
        "--model",
        type=str,
        help="Target model repo ID or local path.",
    )
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
        default=None,
        help=(
            "MLX cache memory limit. Default: 4GB for the serve runtime; auto uses wired-limit/4; "
            "none skips mx.set_cache_limit; BYTES accepts suffixes like 8GB."
        ),
    )
    parser.add_argument(
        "--draft-quant",
        default=None,
        metavar="SPEC",
        help="Draft quantization override, e.g. w4:gs64; use 'none' to disable model defaults.",
    )
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
        "--fastpath-max-tokens",
        type=int,
        default=0,
        help=(
            "Use target-only AR for requests with max_tokens <= this threshold. "
            "Default: 0 (disabled); set a positive value to enable."
        ),
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        default="{}",
        help="JSON object passed to the chat-template renderer.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Set chat-template arg enable_thinking=true.",
    )
    parser.add_argument(
        "--no-eos",
        action="store_true",
        default=False,
        help="Suppress EOS so generation reaches the requested max tokens.",
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=10,
        help="mlx_lm prompt-cache entry count.",
    )
    return parser

def normalize_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    _normalize_chat_template_args(args)
    if args.fastpath_max_tokens < 0:
        raise SystemExit("--fastpath-max-tokens must be >= 0")
    diagnostics_dir = _configure_diagnostics_args(args)
    try:
        runtime_config = resolve_runtime_config(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if runtime_config.target_fa_window > 0 and (
        args.prefix_cache is not False
        or args.prefix_cache_l2 is True
    ):
        sys.stderr.write(
            "[dflash] warning: target FA window disables prefix cache and L2 snapshots\n"
        )
        sys.stderr.flush()
    if diagnostics_dir is not None:
        _write_diagnostics_bootstrap(args, runtime_config, diagnostics_dir)
    diagnostics_config = _build_diagnostics_config(args, diagnostics_dir)
    args.diagnostics_config = diagnostics_config
    args.runtime_config = runtime_config
    args.runtime_context = build_runtime_context(runtime_config, diagnostics_config)
    args.prefill_step_size = runtime_config.prefill_step_size
    return args

def _normalize_chat_template_args(args: argparse.Namespace) -> None:
    raw_args = getattr(args, "chat_template_args", None)
    if raw_args is None:
        chat_args = {}
    elif isinstance(raw_args, str):
        try:
            chat_args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise SystemExit("--chat-template-args must be a JSON object") from exc
    else:
        chat_args = raw_args
    if not isinstance(chat_args, dict):
        raise SystemExit("--chat-template-args must be a JSON object")
    args.chat_template_args = dict(chat_args)
    if getattr(args, "enable_thinking", False):
        args.chat_template_args["enable_thinking"] = True

def _configure_diagnostics_args(args: argparse.Namespace) -> os.PathLike[str] | None:
    mode = str(getattr(args, "diagnostics", "off") or "off")
    diagnostics_dir = getattr(args, "diagnostics_dir", None)
    if mode == "off":
        if diagnostics_dir is not None:
            raise SystemExit("--diagnostics-dir requires --diagnostics basic or full")
        return None

    run_dir = create_run_dir(
        "diagnostics",
        f"serve-{mode}",
        explicit_path=diagnostics_dir,
    )
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
    diagnostics_dir: os.PathLike[str] | None,
) -> DiagnosticsConfig:
    trace_dir: Path | None = None
    cycle_events = False
    if diagnostics_dir is not None:
        trace_dir = Path(diagnostics_dir)
        cycle_events = getattr(args, "diagnostics", "off") == "full"
    return DiagnosticsConfig(
        mode=getattr(args, "diagnostics", "off"),
        run_dir=Path(diagnostics_dir) if diagnostics_dir is not None else None,
        memory_waterfall=bool(getattr(args, "memory_waterfall", None)),
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
    diagnostics_meta = {
        "mode": args.diagnostics,
        "dir": path,
        "memory_waterfall": bool(getattr(args, "memory_waterfall", None)),
        "profile_cycles": args.diagnostics == "full",
    }
    write_manifest(
        Path(path),
        kind="diagnostics",
        label=f"serve-{args.diagnostics}",
        argv=list(sys.argv),
        model=getattr(args, "model", None),
        draft=getattr(args, "draft_model", None),
        effective_config=effective,
    )
    write_json(
        Path(path) / "invocation.json",
        {
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
        "diagnostics": diagnostics_meta,
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
        "",
        "Files:",
        "- `post_events.jsonl`: request timing, throughput, cache hit tokens, prefill accounting, and memory peak when enabled.",
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
        cache_request=_resolve_default_cache_limit(args),
    )
    args.metal_limits = limits
    return limits

def _resolve_default_cache_limit(args: argparse.Namespace) -> MemoryLimit:
    explicit = getattr(args, "cache_limit", None)
    if explicit is not None:
        return explicit
    return 4 * GiB

def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

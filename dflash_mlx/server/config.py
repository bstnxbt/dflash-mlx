# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import json
import logging
import os

import mlx.core as mx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DFlash Server.")
    parser.add_argument("--model", type=str)
    parser.add_argument("--adapter-path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allowed-origins",
        type=lambda x: x.split(","),
        default="*",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--draft-model", "--draft", dest="draft_model", type=str, default=None)
    parser.add_argument("--dflash-max-ctx", type=int, default=None)
    parser.add_argument("--num-draft-tokens", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--draft-quant", default=None, metavar="SPEC")
    parser.add_argument("--quantize-draft", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trust-remote-code", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--chat-template", type=str, default="")
    parser.add_argument("--use-default-chat-template", action="store_true")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chat-template-args", type=json.loads, default="{}")
    parser.add_argument("--decode-concurrency", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--prompt-concurrency", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--prefill-step-size", type=int, default=2048, help=argparse.SUPPRESS)
    parser.add_argument("--prompt-cache-size", type=int, default=10)
    parser.add_argument("--prompt-cache-bytes", type=int)
    parser.add_argument("--pipeline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--prefix-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the DFlash prefix cache that reuses cross-turn KV state. Default: enabled. "
             "Big win on multi-turn agentic workloads, ~neutral on single-turn.",
    )
    parser.add_argument(
        "--prefix-cache-max-entries",
        type=int,
        default=4,
        help="Maximum number of cached prefix snapshots (default: 4).",
    )
    parser.add_argument(
        "--prefix-cache-max-bytes",
        type=int,
        default=8 * 1024 * 1024 * 1024,
        help="Maximum total bytes the prefix cache may hold (default: 8 GiB).",
    )
    return parser


def normalize_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.dflash_max_ctx is not None:
        if args.dflash_max_ctx <= 0:
            raise SystemExit("--dflash-max-ctx must be > 0")
        os.environ["DFLASH_MAX_CTX"] = str(args.dflash_max_ctx)
    if args.prefix_cache_max_entries <= 0:
        raise SystemExit("--prefix-cache-max-entries must be > 0")
    if args.prefix_cache_max_bytes < 0:
        raise SystemExit("--prefix-cache-max-bytes must be >= 0")
    os.environ["DFLASH_PREFIX_CACHE"] = "1" if args.prefix_cache else "0"
    os.environ["DFLASH_PREFIX_CACHE_MAX_ENTRIES"] = str(args.prefix_cache_max_entries)
    os.environ["DFLASH_PREFIX_CACHE_MAX_BYTES"] = str(args.prefix_cache_max_bytes)
    return args


def configure_metal_limits() -> None:
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)
        mx.set_cache_limit(wired_limit // 4)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

_COMMANDS = ("serve", "generate", "benchmark", "doctor", "profiles", "models")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dflash",
        description="DFlash runtime CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="start the OpenAI-compatible server")
    subparsers.add_parser("generate", help="generate one prompt")
    subparsers.add_parser("benchmark", help="run the public baseline vs DFlash benchmark")
    subparsers.add_parser("doctor", help="run local runtime checks")
    subparsers.add_parser("profiles", help="list runtime profiles")
    subparsers.add_parser("models", help="list supported draft mappings")
    return parser

def run(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        build_parser().print_help()
        return 0

    command = args[0]
    if command in ("-h", "--help"):
        build_parser().print_help()
        return 0
    if command == "serve":
        return _run_module_main("dflash_mlx.serve", "dflash serve", args[1:])
    if command == "generate":
        return _run_module_main("dflash_mlx.generate", "dflash generate", args[1:])
    if command == "benchmark":
        return _run_module_main("dflash_mlx.benchmark", "dflash benchmark", args[1:])
    if command == "doctor":
        return _run_module_main("dflash_mlx.doctor", "dflash doctor", args[1:])
    if command == "profiles":
        return _print_profiles()
    if command == "models":
        return _print_models()

    if args and args[0].startswith("-") and ("--model" in args or "--prompt" in args):
        print("error: use `dflash generate ...` for one-shot generation", file=sys.stderr)
        return 2

    build_parser().parse_args(args)
    return 2

def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(run(argv))

def _run_module_main(module_name: str, prog: str, args: Sequence[str]) -> int:
    module = importlib.import_module(module_name)
    try:
        module.main(list(args), prog=prog)
    except SystemExit as exc:
        return _exit_code(exc)
    return 0

def _exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    print(exc.code, file=sys.stderr)
    return 1

def _print_profiles() -> int:
    from dflash_mlx.runtime_profiles import format_profiles

    print(format_profiles())
    return 0

def _print_models() -> int:
    from dflash_mlx.generate import DRAFT_REGISTRY

    for target, draft in DRAFT_REGISTRY.items():
        print(f"{target:22} {draft}")
    return 0

if __name__ == "__main__":
    main()

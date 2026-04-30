# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

COMMANDS = {
    "multiturn": "in-process prefix-cache on/off multi-turn probe",
    "l2-long-session": "real-model L2 prefix-cache long-session probe",
    "l2-synthetic": "synthetic L2 snapshot-size probe without model load",
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run internal DFlash prefix-cache probes.",
    )
    subparsers = parser.add_subparsers(dest="command")
    for name, help_text in COMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    command, rest = args[0], args[1:]
    if command == "multiturn":
        from tools.benchmarks import _prefix_cache_multiturn

        return int(_prefix_cache_multiturn.main(rest))
    if command == "l2-long-session":
        from tools.benchmarks import _prefix_l2_long_session

        return int(_prefix_l2_long_session.main(rest))
    if command == "l2-synthetic":
        from tools.benchmarks import _prefix_l2_synthetic

        result = _prefix_l2_synthetic.main(rest)
        return 0 if result is None else int(result)

    build_parser().parse_args(args)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())

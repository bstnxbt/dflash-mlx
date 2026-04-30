# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

COMMANDS = {
    "run": "start server, proxy, agentic client, and post-process the trace",
    "session": "run an agentic client against an already running server",
    "proxy": "run the HTTP/SSE recording proxy",
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run internal DFlash agentic trace tools.")
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
    if command == "run":
        from tools.benchmarks import _agentic_trace

        return int(_agentic_trace.main(rest))
    if command == "session":
        from tools.benchmarks import _agentic_session

        return int(_agentic_session.main(rest))
    if command == "proxy":
        from tools.benchmarks import _agentic_proxy

        return int(_agentic_proxy.main(rest))

    build_parser().parse_args(args)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())

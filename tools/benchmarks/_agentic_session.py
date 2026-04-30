# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from dflash_mlx.artifacts import create_run_dir, write_manifest

DEFAULT_TASK = (
    "Create a small brick breaker game in Python in main.py, then run "
    "python -m py_compile main.py and fix any syntax error."
)

PI_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def _run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def _task_from_args(args: argparse.Namespace) -> str:
    if args.task_file:
        return Path(args.task_file).read_text()
    return args.task

def _build_command_opencode(args: argparse.Namespace, workspace: Path, task: str) -> list[str]:
    cmd = [
        args.client_bin,
        "run",
        "--model", args.model,
        "--dir", str(workspace),
        "--format", "json",
        "--title", args.label,
    ]
    if args.opencode_thinking:
        cmd.append("--thinking")
    if args.dangerously_skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    cmd.append(task)
    return cmd

def _build_command_pi(args: argparse.Namespace, task: str) -> list[str]:
    cmd = [args.client_bin, "-p", "--model", args.model, "--mode", "json", "--no-session"]
    if not args.pi_extensions:
        cmd.append("--no-extensions")
    if not args.pi_skills:
        cmd.append("--no-skills")
    if not args.pi_prompt_templates:
        cmd.append("--no-prompt-templates")
    if not args.pi_themes:
        cmd.append("--no-themes")
    if not args.pi_context_files:
        cmd.append("--no-context-files")
    if args.pi_thinking != "off":
        cmd += ["--thinking", args.pi_thinking]
    cmd.append(task)
    return cmd

def _summarize_stdout(stdout: str) -> dict[str, Any]:
    event_counts: dict[str, int] = {}
    json_lines = 0
    text_bytes = len(stdout.encode("utf-8", errors="ignore"))
    for line in stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        json_lines += 1
        event = obj.get("type") or obj.get("event") or obj.get("kind") or "json"
        event_counts[str(event)] = event_counts.get(str(event), 0) + 1
    return {
        "stdout_bytes": text_bytes,
        "stdout_lines": len(stdout.splitlines()),
        "json_lines": json_lines,
        "event_counts": event_counts,
    }

def _workspace_manifest(workspace: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if not workspace.exists():
        return files
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace)
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        files.append({"path": str(rel), "bytes": size})
    return files

def _status_snippet(summary: dict[str, Any]) -> str:
    cmd = shlex.join(summary["command"])
    extra = ""
    if summary["client"] == "pi":
        extra = f"- Thinking: `{summary.get('pi_thinking', 'off')}`\n"
    return f"""### {summary["started_at"]} - {summary["client"]} agentic bench: {summary["label"]}

Scope:

- Client: `{summary["client"]}`
- Model: `{summary["model"]}`
{extra}- Worktree: `{summary["cwd"]}`
- Branch: `{summary["git"]["branch"]}`
- Commit: `{summary["git"]["commit"]}`

Command:

```bash
{cmd}
```

Outputs:

- Run directory: `{summary["run_dir"]}`
- Workspace: `{summary["workspace"]}`
- stdout: `{summary["stdout_log"]}`
- stderr: `{summary["stderr_log"]}`
- summary: `{summary["summary_json"]}`

Observed:

- Exit code: `{summary["exit_code"]}`
- Wall time: `{summary["wall_s"]:.2f}s`
- stdout bytes: `{summary["stdout"]["stdout_bytes"]}`
- stderr bytes: `{summary["stderr_bytes"]}`
- Workspace files: `{len(summary["workspace_files"])}`

Verdict:

- TODO: fill after reading stdout/stderr and generated files.
"""

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--client", choices=("opencode", "pi"), default="opencode",
                   help="Which agentic CLI to drive.")
    p.add_argument("--label", default=None,
                   help="Run label (default: <client>_<timestamp>).")
    p.add_argument("--model", required=True,
                   help='Model spec, e.g. "dflash/mlx-community/Qwen3.6-27B-4bit".')
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="Inline task string (used when --task-file is omitted).")
    p.add_argument("--task-file", default=None,
                   help="Path to a file holding the task prompt.")
    p.add_argument("--out-root", default=None,
                   help="Output root directory (default: .artifacts/dflash/traces).")
    p.add_argument("--workspace", default=None,
                   help="Working dir the agent edits in (default: <run_dir>/workspace).")
    p.add_argument("--timeout-s", type=float, default=1800.0,
                   help="Subprocess wall timeout (default: 1800).")
    p.add_argument("--client-bin", default=None,
                   help="Path to the CLI binary (default: which(<client>)).")
    p.add_argument("--append-status", action="store_true",
                   help="Append the run's STATUS snippet to ./STATUS.md.")

    g_oc = p.add_argument_group("opencode-specific")
    g_oc.add_argument("--opencode-thinking", action=argparse.BooleanOptionalAction, default=True,
                      help="Pass opencode --thinking. Default: on.")
    g_oc.add_argument("--dangerously-skip-permissions",
                      action=argparse.BooleanOptionalAction, default=True,
                      help="Pass opencode --dangerously-skip-permissions. Default: on.")

    g_pi = p.add_argument_group("pi-specific")
    g_pi.add_argument("--pi-thinking", default="high", choices=PI_THINKING_LEVELS,
                      help="pi --thinking level (use 'off' to omit). Default: high.")
    g_pi.add_argument("--pi-extensions", action=argparse.BooleanOptionalAction, default=False,
                      help="Enable pi extension discovery. Default: off.")
    g_pi.add_argument("--pi-skills", action=argparse.BooleanOptionalAction, default=False,
                      help="Enable pi skills discovery. Default: off.")
    g_pi.add_argument("--pi-prompt-templates", action=argparse.BooleanOptionalAction, default=False,
                      help="Enable pi prompt-template discovery. Default: off.")
    g_pi.add_argument("--pi-themes", action=argparse.BooleanOptionalAction, default=False,
                      help="Enable pi theme discovery. Default: off.")
    g_pi.add_argument("--pi-context-files", action=argparse.BooleanOptionalAction, default=False,
                      help="Enable pi auto-loading of AGENTS.md / CLAUDE.md. Default: off.")

    return p.parse_args(list(argv) if argv is not None else None)

def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.label is None:
        args.label = f"{args.client}_{_now_stamp()}"
    if args.client_bin is None:
        args.client_bin = shutil.which(args.client) or args.client

    if args.out_root is None:
        run_dir = create_run_dir("trace", f"{args.client}-{args.label}")
    else:
        run_dir = Path(args.out_root) / f"{_now_stamp()}-{args.client}-{args.label}"
        run_dir.mkdir(parents=True, exist_ok=False)
    workspace = Path(args.workspace) if args.workspace else run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    task = _task_from_args(args)
    if args.client == "opencode":
        command = _build_command_opencode(args, workspace, task)
    else:
        command = _build_command_pi(args, task)
    started_at = _iso_now()

    metadata: dict[str, Any] = {
        "started_at": started_at,
        "label": args.label,
        "client": args.client,
        "model": args.model,
        "cwd": os.getcwd(),
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "task": task,
        "command": command,
        "git": {
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _run_git(["rev-parse", "HEAD"]),
            "status_short": _run_git(["status", "--short"]),
        },
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "PATH": os.environ.get("PATH", ""),
        },
    }
    if args.client == "opencode":
        metadata["opencode_thinking"] = bool(args.opencode_thinking)
        metadata["dangerously_skip_permissions"] = bool(args.dangerously_skip_permissions)
    else:
        metadata["pi_thinking"] = args.pi_thinking
        metadata["pi_extensions"] = bool(args.pi_extensions)
        metadata["pi_skills"] = bool(args.pi_skills)
        metadata["pi_prompt_templates"] = bool(args.pi_prompt_templates)
        metadata["pi_themes"] = bool(args.pi_themes)
        metadata["pi_context_files"] = bool(args.pi_context_files)

    write_manifest(
        run_dir,
        kind="trace",
        label=f"{args.client}-{args.label}",
        argv=list(sys.argv),
        model=args.model,
        effective_config=metadata,
    )
    _write_json(run_dir / "metadata.json", metadata)
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n")
    (run_dir / "task.txt").write_text(task)

    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=args.timeout_s,
        check=False,
    )
    wall_s = time.perf_counter() - start

    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    stdout_log.write_text(proc.stdout)
    stderr_log.write_text(proc.stderr)

    summary = dict(metadata)
    summary.update(
        {
            "finished_at": _iso_now(),
            "exit_code": proc.returncode,
            "wall_s": wall_s,
            "stdout": _summarize_stdout(proc.stdout),
            "stderr_bytes": len(proc.stderr.encode("utf-8", errors="ignore")),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "summary_json": str(run_dir / "summary.json"),
            "workspace_files": _workspace_manifest(workspace),
        }
    )
    _write_json(run_dir / "summary.json", summary)

    snippet = _status_snippet(summary)
    (run_dir / "STATUS_SNIPPET.md").write_text(snippet)
    if args.append_status:
        with Path("STATUS.md").open("a") as f:
            f.write("\n" + snippet)

    print(f"Run directory: {run_dir}")
    print(f"Exit code    : {proc.returncode}")
    print(f"Wall         : {wall_s:.2f}s")
    print(f"Summary      : {run_dir / 'summary.json'}")
    print(f"Status entry : {run_dir / 'STATUS_SNIPPET.md'}")
    return proc.returncode

if __name__ == "__main__":
    raise SystemExit(main())

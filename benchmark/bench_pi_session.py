"""Uniform pi agentic benchmark runner.

Sister of bench_opencode_session.py for the `pi` CLI (https://pi.dev).

This runner does not start model servers. Start exactly one server yourself,
then run this script against the configured pi provider/model. It records
the command, metadata, stdout/stderr, summary, workspace, and a STATUS.md
snippet — same shape as the opencode bench so the two are directly comparable.

The pi provider/model is passed as a single string in pi's "<provider>/<id>"
form — e.g. `dflash/mlx-community/Qwen3.6-27B-4bit` or
`mlxlm/mlx-community/Qwen3.6-35B-A3B-4bit`. Custom providers must already be
declared in `~/.pi/agent/models.json`.
"""

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
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TASK = (
    "Create a small brick breaker game in Python in main.py, then run "
    "python -m py_compile main.py and fix any syntax error."
)

THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")


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


def _build_command(args: argparse.Namespace, task: str) -> list[str]:
    cmd = [args.pi_bin, "-p", "--model", args.model, "--mode", "json", "--no-session"]
    if not args.extensions:
        cmd.append("--no-extensions")
    if not args.skills:
        cmd.append("--no-skills")
    if not args.prompt_templates:
        cmd.append("--no-prompt-templates")
    if not args.themes:
        cmd.append("--no-themes")
    if not args.context_files:
        cmd.append("--no-context-files")
    if args.thinking != "off":
        cmd += ["--thinking", args.thinking]
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
    return f"""### {summary["started_at"]} - pi agentic bench: {summary["label"]}

Scope:

- Client: `pi`
- Model: `{summary["model"]}`
- Thinking: `{summary["thinking"]}`
- Worktree: `{summary["cwd"]}`
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=f"pi_{_now_stamp()}")
    parser.add_argument(
        "--model",
        required=True,
        help='pi model spec, e.g. "dflash/mlx-community/Qwen3.6-27B-4bit".',
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--task-file")
    parser.add_argument("--out-root", default="benchmark/pi_runs")
    parser.add_argument("--workspace")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--pi-bin", default=shutil.which("pi") or "pi")
    parser.add_argument(
        "--thinking",
        default="high",
        choices=THINKING_LEVELS,
        help="Thinking level passed to pi (default: high). Use 'off' to omit the flag.",
    )
    parser.add_argument(
        "--extensions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable pi extension discovery (default: off for reproducible runs).",
    )
    parser.add_argument(
        "--skills",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable pi skills discovery (default: off).",
    )
    parser.add_argument(
        "--prompt-templates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable pi prompt-template discovery (default: off).",
    )
    parser.add_argument(
        "--themes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable pi theme discovery (default: off).",
    )
    parser.add_argument(
        "--context-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable AGENTS.md / CLAUDE.md auto-loading by pi (default: off).",
    )
    parser.add_argument("--append-status", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.out_root) / f"{_now_stamp()}_{args.label}"
    workspace = Path(args.workspace) if args.workspace else run_dir / "workspace"
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace.mkdir(parents=True, exist_ok=True)

    task = _task_from_args(args)
    command = _build_command(args, task)
    started_at = _iso_now()

    metadata = {
        "started_at": started_at,
        "label": args.label,
        "client": "pi",
        "model": args.model,
        "thinking": args.thinking,
        "cwd": os.getcwd(),
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "task": task,
        "command": command,
        "extensions": bool(args.extensions),
        "skills": bool(args.skills),
        "prompt_templates": bool(args.prompt_templates),
        "themes": bool(args.themes),
        "context_files": bool(args.context_files),
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
        status_path = Path("STATUS.md")
        with status_path.open("a") as f:
            f.write("\n" + snippet)

    print(f"Run directory: {run_dir}")
    print(f"Exit code    : {proc.returncode}")
    print(f"Wall         : {wall_s:.2f}s")
    print(f"Summary      : {run_dir / 'summary.json'}")
    print(f"Status entry : {run_dir / 'STATUS_SNIPPET.md'}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

"""End-to-end agentic trace bench: server + proxy + opencode + post-process.

Per-run output layout:

    benchmark/results/agentic-trace-<stamp>-<label>/
        requests/NNN.json           — full request body (from proxy)
        sse/NNN.jsonl               — per-chunk SSE log with t_ms
        server/
            cmd.txt
            stderr.log
            stdout.log
            metrics.jsonl           — parsed per-POST DFlash/mlx_lm metrics
        opencode/
            cmd.txt
            stdout.jsonl
            stderr.log
        proxy/
            cmd.txt
            proxy.log
        workspace/                  — opencode workspace
        config_snapshot.json
        summary.json                — derived landmarks per POST + global
        compare.md                  — human-readable

Answers per POST:
    prompt_tokens, completion_tokens
    first_byte_ms, first_token_ms, first_reasoning_ms
    first_tool_call_sent_ms, tool_call_complete_ms
    server_tps, acceptance, tokens_per_cycle, cache_hit_tokens
    finish_reason
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROXY_PORT = 9788
DEFAULT_DFLASH_PORT = 8090
DEFAULT_MLXLM_PORT = 8091
OPENCODE_CONFIG = Path.home() / ".config/opencode/opencode.jsonc"
TRACE_PROVIDER_ID = "trace"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _wait_health(url: str, timeout_s: float, label: str) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(2)
    sys.stderr.write(f"[orch] {label} health timeout on {url}\n")
    return False


def _patch_opencode_config(target: str, proxy_port: int) -> dict[str, Any]:
    """Add a `trace` provider entry pointing at the proxy. Returns the original
    parsed config so we can restore."""
    raw = OPENCODE_CONFIG.read_text()
    # opencode.jsonc may have // comments; strip them for parsing
    no_comments = re.sub(r"^\s*//.*$", "", raw, flags=re.MULTILINE)
    config = json.loads(no_comments)
    config.setdefault("provider", {})
    config["provider"][TRACE_PROVIDER_ID] = {
        "name": "Trace",
        "npm": "@ai-sdk/openai-compatible",
        "models": {
            target: {
                "name": target,
                "limit": {"context": 131072, "output": 40000},
            }
        },
        "options": {"baseURL": f"http://127.0.0.1:{proxy_port}/v1"},
    }
    OPENCODE_CONFIG.write_text(json.dumps(config, indent=2))
    return config


def _restore_opencode_config(snapshot_text: str) -> None:
    OPENCODE_CONFIG.write_text(snapshot_text)


def _spawn(cmd: list[str], stdout_path: Path, stderr_path: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    stdout_f = stdout_path.open("w")
    stderr_f = stderr_path.open("w")
    return subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=stderr_f,
        env={**os.environ, **(env or {})},
        cwd=REPO_ROOT,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )


def _terminate(proc: subprocess.Popen, label: str, term_grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except Exception as e:
        sys.stderr.write(f"[orch] {label} term err: {e!r}\n")
    try:
        proc.wait(timeout=term_grace_s)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        proc.wait(timeout=5)


# --------- bench_logger event readers (preferred over stderr) ---------

def read_dflash_events(events_dir: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Read post_events.jsonl + cycle_events.jsonl + cache_events.jsonl emitted
    by dflash_mlx.bench_logger when DFLASH_BENCH_LOG_DIR is set. Returns
    (post_events, cycles_by_request_id, cache_events). Missing files / bad
    lines are silently skipped so pre-instrumentation runs still parse."""
    posts: list[dict[str, Any]] = []
    cycles_by_req: dict[int, list[dict[str, Any]]] = {}
    cache: list[dict[str, Any]] = []

    pe = events_dir / "post_events.jsonl"
    if pe.exists():
        for line in pe.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                posts.append(json.loads(line))
            except Exception:
                pass

    ce = events_dir / "cycle_events.jsonl"
    if ce.exists():
        for line in ce.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                rid = ev.get("request_id")
                if rid is not None:
                    cycles_by_req.setdefault(rid, []).append(ev)
            except Exception:
                pass

    xe = events_dir / "cache_events.jsonl"
    if xe.exists():
        for line in xe.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cache.append(json.loads(line))
            except Exception:
                pass

    return posts, cycles_by_req, cache


def summarize_cycles(cycles: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cycles:
        return None
    n = len(cycles)
    sorted_verify = sorted(c.get("verify_us", 0.0) for c in cycles)
    sorted_block = sorted(c.get("block_len", 0) for c in cycles)
    sorted_commit = sorted(c.get("commit_count", 0) for c in cycles)
    sorted_accept = sorted(c.get("acceptance_len", 0) for c in cycles)
    return {
        "n_cycles": n,
        "total_commits": sum(c.get("commit_count", 0) for c in cycles),
        "mean_acceptance_len": (sum(sorted_accept) / n) if n else None,
        "mean_block_len": (sum(sorted_block) / n) if n else None,
        "mean_commit_count": (sum(sorted_commit) / n) if n else None,
        "verify_us_p50": sorted_verify[n // 2] if n else None,
        "verify_us_p99": sorted_verify[min(n - 1, max(0, int(n * 0.99) - 1))] if n else None,
    }


def post_event_to_legacy_metric(pe: dict[str, Any], cycles_summary: dict[str, Any] | None) -> dict[str, Any]:
    """Project a post_event into the same shape `_post_view` already consumes
    so the rest of the pipeline (aggregate, compare.md) works unchanged."""
    wall_ms = pe.get("wall_ms") or 0.0
    gen = pe.get("generated_tokens") or 0
    tps = (gen / (wall_ms / 1000.0)) if wall_ms > 0 else None
    return {
        "tps": tps,
        "accept": pe.get("acceptance_ratio"),
        "tokens": gen,
        "wall_s": wall_ms / 1000.0 if wall_ms else None,
        "prompt_tokens": pe.get("prompt_tokens"),
        "cache_hit_tokens": pe.get("cache_hit_tokens"),
        # extra fields available to compare.md / summary.json
        "ttft_ms_server": pe.get("ttft_ms"),
        "prefill_ms_server": pe.get("prefill_ms"),
        "decode_ms_server": pe.get("decode_ms"),
        "cycles_completed": pe.get("cycles_completed"),
        "finish_reason_server": pe.get("finish_reason"),
        "cache_lookup_ms": pe.get("cache_lookup_ms"),
        "cache_insert_ms": pe.get("cache_insert_ms"),
        "mode_used": pe.get("mode_used"),
        "request_id": pe.get("request_id"),
        "cycles_summary": cycles_summary,
        "_source": "events",
    }


def summarize_cache_events(cache: list[dict[str, Any]]) -> dict[str, Any]:
    lookups = [e for e in cache if e.get("op") == "lookup"]
    # bench_logger emits result in {"miss", "prefix_hit", "exact_hit", ...} — anything not "miss" is a hit
    hits = [e for e in lookups if e.get("result") and e.get("result") != "miss"]
    inserts = [e for e in cache if e.get("op") == "insert"]
    fingerprint_reject = sum(1 for e in lookups if e.get("fingerprint_reject"))
    return {
        "n_lookups": len(lookups),
        "n_hits": len(hits),
        "hit_rate": (len(hits) / len(lookups)) if lookups else None,
        "n_inserts": len(inserts),
        "fingerprint_rejects": fingerprint_reject,
        "total_matched_tokens": sum(e.get("matched_len", 0) for e in hits),
    }


# --------- server-stderr metric parsers (legacy fallback) ---------

_DFLASH_TPS_RE = re.compile(
    r"\[dflash\]\s+([\d.]+)\s+tok/s\s+\|\s+([\d.]+)%\s+accepted\s+\|\s+(\d+)\s+tokens\s+\|\s+([\d.]+)s\s+\|\s+prompt:\s+(\d+)\s+tokens"
)
_DFLASH_HIT_RE = re.compile(
    r"\[dflash\]\s+prefix\s+cache\s+hit\s+(\d+)/(\d+)\s+tokens"
)
_DFLASH_STATS_RE = re.compile(
    r"\[dflash\]\s+prefix-cache-stats.*?prefill_tokens_saved=(\d+)"
)


def parse_dflash_stderr(text: str) -> list[dict[str, Any]]:
    """Walk the stderr top-down. Each `tok/s` line is a per-call metric.
    Hit lines preceding it (within ~same second) attach to that call."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _DFLASH_TPS_RE.search(line)
        if m:
            events.append({
                "kind": "tps",
                "raw": line,
                "tps": float(m.group(1)),
                "accept": float(m.group(2)) / 100.0,
                "tokens": int(m.group(3)),
                "wall_s": float(m.group(4)),
                "prompt_tokens": int(m.group(5)),
            })
            continue
        m = _DFLASH_HIT_RE.search(line)
        if m:
            events.append({
                "kind": "hit",
                "raw": line,
                "hit": int(m.group(1)),
                "stable": int(m.group(2)),
            })
            continue
        m = _DFLASH_STATS_RE.search(line)
        if m:
            events.append({
                "kind": "stats",
                "raw": line,
                "prefill_tokens_saved": int(m.group(1)),
            })
    return events


def attach_dflash_metrics_to_posts(events: list[dict[str, Any]], n_posts: int) -> list[dict[str, Any]]:
    """Group events into per-POST buckets. Each `tps` event closes a POST.
    `hit` and `stats` events that came BEFORE the next tps event belong to the
    POST that produced that tps."""
    buckets: list[dict[str, Any]] = []
    pending: dict[str, Any] = {"hit": None, "stats": None, "all": []}
    for ev in events:
        pending["all"].append(ev)
        if ev["kind"] == "hit":
            pending["hit"] = ev
        elif ev["kind"] == "stats":
            pending["stats"] = ev
        elif ev["kind"] == "tps":
            buckets.append({
                "tps": ev["tps"],
                "accept": ev["accept"],
                "tokens": ev["tokens"],
                "wall_s": ev["wall_s"],
                "prompt_tokens": ev["prompt_tokens"],
                "cache_hit_tokens": pending["hit"]["hit"] if pending["hit"] else None,
                "stable_prefix": pending["hit"]["stable"] if pending["hit"] else None,
                "prefill_tokens_saved_cumulative": pending["stats"]["prefill_tokens_saved"] if pending["stats"] else None,
            })
            pending = {"hit": None, "stats": None, "all": []}
    # DFlash logs the same big turn twice (mid + final summary) → merge: keep
    # the LAST entry per logical POST. Heuristic: if two consecutive entries
    # share prompt_tokens and the second's tokens > first's tokens, drop the
    # earlier one.
    merged: list[dict[str, Any]] = []
    for b in buckets:
        if merged and merged[-1]["prompt_tokens"] == b["prompt_tokens"] and b["tokens"] >= merged[-1]["tokens"]:
            merged[-1] = b
        else:
            merged.append(b)
    return merged


# --------- SSE post-processing ---------

def _delta_text(delta: dict[str, Any]) -> str:
    if not isinstance(delta, dict):
        return ""
    out = ""
    if isinstance(delta.get("content"), str):
        out += delta["content"]
    return out


def _delta_reasoning(delta: dict[str, Any]) -> str:
    if not isinstance(delta, dict):
        return ""
    for k in ("reasoning_content", "reasoning"):
        v = delta.get(k)
        if isinstance(v, str):
            return v
    return ""


def _delta_tool_calls(delta: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not isinstance(delta, dict):
        return None
    tc = delta.get("tool_calls")
    if isinstance(tc, list) and tc:
        return tc
    return None


def derive_post_landmarks(sse_path: Path) -> dict[str, Any]:
    """Read sse/NNN.jsonl and compute landmark t_ms values."""
    out: dict[str, Any] = {
        "first_byte_ms": None,
        "first_content_token_ms": None,
        "first_reasoning_ms": None,
        "first_tool_call_sent_ms": None,
        "tool_call_complete_ms": None,
        "finish_reason": None,
        "n_chunks": 0,
        "total_content_chars": 0,
        "total_reasoning_chars": 0,
        "saw_think_open_ms": None,
        "saw_think_close_ms": None,
        "end_t_ms": None,
        "tool_calls": [],
    }
    accumulated_tool_call_args: list[str] = []
    in_think = False

    with sse_path.open() as f:
        for raw_line in f:
            try:
                ev = json.loads(raw_line)
            except Exception:
                continue
            if ev.get("type") == "first_byte":
                out["first_byte_ms"] = ev["t_ms"]
                continue
            if ev.get("type") == "end":
                out["end_t_ms"] = ev["t_ms"]
                continue
            if ev.get("type") not in ("event", "event_tail"):
                continue
            payload = ev.get("payload") or {}
            data = payload.get("data")
            if data is None:
                # raw "[DONE]" or non-JSON
                continue
            t_ms = ev["t_ms"]
            out["n_chunks"] += 1
            choices = data.get("choices") or []
            for ch in choices:
                delta = ch.get("delta") or {}
                fr = ch.get("finish_reason")
                if fr and out["finish_reason"] is None:
                    out["finish_reason"] = fr
                txt = _delta_text(delta)
                rsn = _delta_reasoning(delta)
                tcs = _delta_tool_calls(delta)
                if rsn:
                    out["total_reasoning_chars"] += len(rsn)
                    if out["first_reasoning_ms"] is None:
                        out["first_reasoning_ms"] = t_ms
                if txt:
                    if not in_think and "<think>" in txt and out["saw_think_open_ms"] is None:
                        out["saw_think_open_ms"] = t_ms
                        in_think = True
                    if in_think and "</think>" in txt and out["saw_think_close_ms"] is None:
                        out["saw_think_close_ms"] = t_ms
                        in_think = False
                    out["total_content_chars"] += len(txt)
                    if out["first_content_token_ms"] is None:
                        out["first_content_token_ms"] = t_ms
                if tcs:
                    if out["first_tool_call_sent_ms"] is None:
                        out["first_tool_call_sent_ms"] = t_ms
                    for tc in tcs:
                        out["tool_calls"].append({"t_ms": t_ms, "delta": tc})
                    if fr in ("tool_calls", "function_call") and out["tool_call_complete_ms"] is None:
                        out["tool_call_complete_ms"] = t_ms
            if data.get("usage"):
                out["usage"] = data["usage"]
    if out["tool_calls"] and out["tool_call_complete_ms"] is None:
        out["tool_call_complete_ms"] = out["tool_calls"][-1]["t_ms"]
    return out


def derive_request_summary(req_path: Path) -> dict[str, Any]:
    obj = json.loads(req_path.read_text())
    body = obj.get("body") or {}
    msgs = body.get("messages") or []
    return {
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
        "stream": body.get("stream"),
        "n_messages": len(msgs),
        "last_role": msgs[-1].get("role") if msgs else None,
        "tools_count": len(body.get("tools") or []),
        "has_tool_choice": "tool_choice" in body,
        "first_message_chars": sum(len(str(m.get("content", ""))) for m in msgs[:1]),
        "total_message_chars": sum(len(str(m.get("content", ""))) for m in msgs),
    }


# --------- main runner ---------

def _build_server_cmd(args) -> tuple[list[str], int, str]:
    if args.backend == "dflash":
        port = args.dflash_port
        cmd = [
            f"{REPO_ROOT}/.venv/bin/python",
            "-m",
            "dflash_mlx.serve",
            "--model",
            args.target,
            "--draft",
            args.draft,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--chat-template-args",
            '{"enable_thinking":true}',
        ]
        return cmd, port, f"http://127.0.0.1:{port}"
    if args.backend == "mlxlm":
        port = args.mlxlm_port
        cmd = [
            f"{REPO_ROOT}/.venv/bin/python",
            "-m",
            "mlx_lm.server",
            "--model",
            args.target,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--chat-template-args",
            '{"enable_thinking":true}',
        ]
        return cmd, port, f"http://127.0.0.1:{port}"
    raise SystemExit(f"unknown backend {args.backend}")


def _build_proxy_cmd(args, run_dir: Path, upstream_url: str) -> list[str]:
    return [
        f"{REPO_ROOT}/.venv/bin/python",
        "-m",
        "benchmark.agentic_proxy",
        "--listen-host",
        "127.0.0.1",
        "--listen-port",
        str(args.proxy_port),
        "--upstream-url",
        upstream_url,
        "--out-dir",
        str(run_dir),
    ]


def _build_opencode_cmd(args, workspace: Path, task: str, label: str) -> list[str]:
    cmd = [
        args.opencode_bin,
        "run",
        "--model",
        f"{TRACE_PROVIDER_ID}/{args.target}",
        "--dir",
        str(workspace.resolve()),
        "--format",
        "json",
        "--title",
        label,
        "--print-logs",
        "--log-level",
        "INFO",
    ]
    if args.thinking:
        cmd.append("--thinking")
    if args.dangerously_skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    cmd.append(task)
    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["dflash", "mlxlm"], required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--draft", default=None, help="required for --backend dflash")
    p.add_argument("--task-file", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--out-root", default="benchmark/results")
    p.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    p.add_argument("--dflash-port", type=int, default=DEFAULT_DFLASH_PORT)
    p.add_argument("--mlxlm-port", type=int, default=DEFAULT_MLXLM_PORT)
    p.add_argument("--server-ready-timeout-s", type=float, default=300.0)
    p.add_argument("--proxy-ready-timeout-s", type=float, default=30.0)
    p.add_argument("--opencode-timeout-s", type=float, default=1800.0)
    p.add_argument("--opencode-bin", default=shutil.which("opencode") or "opencode")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--dangerously-skip-permissions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--enable-prefix-cache", action=argparse.BooleanOptionalAction, default=True,
                   help="dflash only: set DFLASH_PREFIX_CACHE=1 + sane defaults")
    p.add_argument("--compare-to", default=None,
                   help="path to a prior agentic-trace run dir; emits a tool_call_latency_gap verdict in compare.md")
    args = p.parse_args()

    if args.backend == "dflash" and not args.draft:
        raise SystemExit("--draft is required when --backend=dflash")

    label = args.label or f"{args.backend}_{Path(args.target).name}"
    stamp = _now_stamp()
    run_dir = Path(args.out_root) / f"agentic-trace-{stamp}-{label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "server").mkdir()
    (run_dir / "proxy").mkdir()
    (run_dir / "opencode").mkdir()
    workspace = run_dir / "workspace"
    workspace.mkdir()

    task = Path(args.task_file).read_text()

    server_cmd, server_port, upstream_url = _build_server_cmd(args)
    server_health_url = f"{upstream_url}/v1/models"
    proxy_cmd = _build_proxy_cmd(args, run_dir, upstream_url)
    proxy_health_url = f"http://127.0.0.1:{args.proxy_port}/v1/models"
    opencode_cmd = _build_opencode_cmd(args, workspace, task, label)

    (run_dir / "server" / "cmd.txt").write_text(shlex.join(server_cmd) + "\n")
    (run_dir / "proxy" / "cmd.txt").write_text(shlex.join(proxy_cmd) + "\n")
    (run_dir / "opencode" / "cmd.txt").write_text(shlex.join(opencode_cmd) + "\n")
    (run_dir / "task.txt").write_text(task)

    server_env = {}
    if args.backend == "dflash":
        events_dir = run_dir / "events"
        events_dir.mkdir(exist_ok=True)
        server_env["DFLASH_BENCH_LOG_DIR"] = str(events_dir)
        if args.enable_prefix_cache:
            server_env["DFLASH_PREFIX_CACHE"] = "1"
            server_env["DFLASH_PREFIX_CACHE_MAX_ENTRIES"] = "8"
            server_env["DFLASH_PREFIX_CACHE_MAX_BYTES"] = "10737418240"

    config_text_before = OPENCODE_CONFIG.read_text()
    (run_dir / "config_snapshot.json").write_text(config_text_before)

    metadata = {
        "started_at": _iso_now(),
        "label": label,
        "backend": args.backend,
        "target": args.target,
        "draft": args.draft,
        "git": {
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _git(["rev-parse", "HEAD"]),
        },
        "host": platform.platform(),
        "python": sys.version,
        "server_env": server_env,
        "proxy_port": args.proxy_port,
        "server_port": server_port,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    server_proc = None
    proxy_proc = None
    oc_proc = None
    oc_returncode = None
    oc_wall_s = None

    try:
        # 1. server
        sys.stderr.write(f"[orch] starting server: {' '.join(server_cmd)}\n")
        server_proc = _spawn(server_cmd, run_dir / "server" / "stdout.log", run_dir / "server" / "stderr.log", env=server_env)
        if not _wait_health(server_health_url, args.server_ready_timeout_s, "server"):
            raise SystemExit("server not ready")

        # 2. proxy
        sys.stderr.write(f"[orch] starting proxy: {' '.join(proxy_cmd)}\n")
        proxy_proc = _spawn(proxy_cmd, run_dir / "proxy" / "stdout.log", run_dir / "proxy" / "stderr.log")
        if not _wait_health(proxy_health_url, args.proxy_ready_timeout_s, "proxy"):
            raise SystemExit("proxy not ready")

        # 3. patch config (after proxy is up so opencode finds the URL alive)
        _patch_opencode_config(args.target, args.proxy_port)

        # 4. run opencode
        sys.stderr.write(f"[orch] starting opencode: {' '.join(opencode_cmd)}\n")
        oc_t0 = time.perf_counter()
        oc_proc = subprocess.Popen(
            opencode_cmd,
            cwd=workspace,
            stdout=(run_dir / "opencode" / "stdout.jsonl").open("w"),
            stderr=(run_dir / "opencode" / "stderr.log").open("w"),
            env=os.environ.copy(),
        )
        try:
            oc_returncode = oc_proc.wait(timeout=args.opencode_timeout_s)
        except subprocess.TimeoutExpired:
            sys.stderr.write("[orch] opencode timeout, killing\n")
            oc_proc.kill()
            oc_returncode = -9
        oc_wall_s = time.perf_counter() - oc_t0
    finally:
        # restore config first so a crash here doesn't leave stale entries
        try:
            _restore_opencode_config(config_text_before)
        except Exception as e:
            sys.stderr.write(f"[orch] restore config err: {e!r}\n")
        if proxy_proc is not None:
            _terminate(proxy_proc, "proxy")
        if server_proc is not None:
            _terminate(server_proc, "server")

    # ------- post-process -------

    server_stderr_text = (run_dir / "server" / "stderr.log").read_text()
    cache_summary: dict[str, Any] = {}
    per_post_metrics: list[dict[str, Any]] = []
    if args.backend == "dflash":
        events_dir = run_dir / "events"
        post_evts, cycles_by_req, cache_evts = read_dflash_events(events_dir)
        if post_evts:
            # use events as source of truth — sort by request_id for positional join
            for pe in sorted(post_evts, key=lambda e: e.get("request_id", 0)):
                rid = pe.get("request_id")
                cycles_summary = summarize_cycles(cycles_by_req.get(rid, []))
                per_post_metrics.append(post_event_to_legacy_metric(pe, cycles_summary))
            cache_summary = summarize_cache_events(cache_evts)
        else:
            # legacy fallback when DFLASH_BENCH_LOG_DIR was not set
            events = parse_dflash_stderr(server_stderr_text)
            per_post_metrics = attach_dflash_metrics_to_posts(events, n_posts=0)
        (run_dir / "server" / "metrics.jsonl").write_text(
            "\n".join(json.dumps(m) for m in per_post_metrics) + ("\n" if per_post_metrics else "")
        )
    else:
        (run_dir / "server" / "metrics.jsonl").write_text("")

    request_files = sorted((run_dir / "requests").glob("*.json"))
    sse_files = sorted((run_dir / "sse").glob("*.jsonl"))

    posts: list[dict[str, Any]] = []
    for i, req_path in enumerate(request_files, start=1):
        sse_path = run_dir / "sse" / req_path.name.replace(".json", ".jsonl")
        req_summary = derive_request_summary(req_path)
        landmarks = derive_post_landmarks(sse_path) if sse_path.exists() else {}
        server_metric = per_post_metrics[i - 1] if (i - 1) < len(per_post_metrics) else None
        posts.append({
            "idx": i,
            "request": req_summary,
            "landmarks": landmarks,
            "server_metric": server_metric,
        })

    workspace_files = sorted([
        {"path": str(p.relative_to(workspace)), "bytes": p.stat().st_size}
        for p in workspace.rglob("*") if p.is_file() and ".ruff_cache" not in p.parts
    ], key=lambda d: d["path"])

    summary = {
        "metadata": metadata,
        "finished_at": _iso_now(),
        "opencode_exit_code": oc_returncode,
        "opencode_wall_s": oc_wall_s,
        "post_count": len(posts),
        "posts": posts,
        "workspace_files": workspace_files,
        "totals": _aggregate(posts, oc_wall_s),
        "cache_summary": cache_summary if args.backend == "dflash" else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    peer_summary = None
    if args.compare_to:
        peer_path = Path(args.compare_to)
        peer_json = peer_path / "summary.json" if peer_path.is_dir() else peer_path
        try:
            peer_summary = json.loads(peer_json.read_text())
        except Exception as e:
            sys.stderr.write(f"[orch] could not load peer summary {peer_json}: {e!r}\n")
    (run_dir / "compare.md").write_text(_render_compare(summary, peer=peer_summary))

    print(f"Run directory: {run_dir}")
    print(f"OpenCode exit: {oc_returncode}")
    print(f"Wall         : {oc_wall_s:.2f}s")
    print(f"POSTs        : {len(posts)}")
    print(f"Summary      : {run_dir / 'summary.json'}")
    print(f"Compare      : {run_dir / 'compare.md'}")
    return 0 if oc_returncode == 0 else 1


def _post_view(p: dict[str, Any]) -> dict[str, Any]:
    """Unified per-POST view that falls back from server_metric to SSE usage.
    Server metrics (DFlash stderr) win when present; otherwise we use the
    `usage` chunk that mlx_lm.server emits at end-of-stream so the row isn't
    blank on baseline runs."""
    sm = p.get("server_metric") or {}
    lm = p.get("landmarks") or {}
    usage = lm.get("usage") or {}
    fb = lm.get("first_byte_ms")
    end = lm.get("end_t_ms")
    decode_wall_s_est = ((end - fb) / 1000.0) if (fb is not None and end is not None and end > fb) else None
    prompt_tokens = sm.get("prompt_tokens") if sm.get("prompt_tokens") is not None else usage.get("prompt_tokens")
    decode_tokens = sm.get("tokens") if sm.get("tokens") is not None else usage.get("completion_tokens")
    wall_s = sm.get("wall_s") if sm.get("wall_s") is not None else decode_wall_s_est
    tps = sm.get("tps")
    if tps is None and decode_tokens and wall_s and wall_s > 0:
        tps = decode_tokens / wall_s
    return {
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "wall_s": wall_s,
        "tps": tps,
        "accept": sm.get("accept"),
        "cache_hit_tokens": sm.get("cache_hit_tokens"),
        "source": "server" if sm else ("usage" if usage else "none"),
    }


def _aggregate(posts: list[dict[str, Any]], wall_s: float | None) -> dict[str, Any]:
    total_decode_tokens = 0
    total_decode_wall_s = 0.0
    total_prompt_tokens = 0
    total_cache_hit = 0
    accept_weighted_num = 0.0
    accept_weighted_den = 0.0
    first_tool_call_ms_list: list[float] = []
    for p in posts:
        v = _post_view(p)
        if v["decode_tokens"]:
            total_decode_tokens += v["decode_tokens"]
        if v["wall_s"]:
            total_decode_wall_s += v["wall_s"]
        if v["prompt_tokens"]:
            total_prompt_tokens += v["prompt_tokens"]
        if v["cache_hit_tokens"]:
            total_cache_hit += v["cache_hit_tokens"]
        if v["accept"] is not None and v["decode_tokens"]:
            accept_weighted_num += v["accept"] * v["decode_tokens"]
            accept_weighted_den += v["decode_tokens"]
        ftc = (p.get("landmarks") or {}).get("first_tool_call_sent_ms")
        if isinstance(ftc, (int, float)):
            first_tool_call_ms_list.append(float(ftc))
    return {
        "wall_s": wall_s,
        "post_count": len(posts),
        "total_prompt_tokens": total_prompt_tokens,
        "total_decode_tokens": total_decode_tokens,
        "total_decode_wall_s": total_decode_wall_s,
        "decode_tps_avg": (total_decode_tokens / total_decode_wall_s) if total_decode_wall_s > 0 else None,
        "total_cache_hit_tokens": total_cache_hit,
        "weighted_acceptance": (accept_weighted_num / accept_weighted_den) if accept_weighted_den else None,
        "first_tool_call_ms_per_post": first_tool_call_ms_list,
        "first_tool_call_ms_sum": sum(first_tool_call_ms_list) if first_tool_call_ms_list else None,
        "first_tool_call_ms_avg": (sum(first_tool_call_ms_list) / len(first_tool_call_ms_list)) if first_tool_call_ms_list else None,
    }


def _ms(v):
    return f"{v:.0f}" if isinstance(v, (int, float)) else "—"


def _render_compare(summary: dict[str, Any], peer: dict[str, Any] | None = None) -> str:
    md = []
    meta = summary["metadata"]
    tot = summary["totals"]
    md.append(f"# Agentic trace — {meta['label']}")
    md.append("")
    md.append(f"- backend: `{meta['backend']}`")
    md.append(f"- target: `{meta['target']}`")
    md.append(f"- draft: `{meta.get('draft')}`")
    md.append(f"- commit: `{meta['git']['commit']}`")
    md.append(f"- env: `{meta['server_env']}`")
    md.append(f"- wall_s: **{tot['wall_s']:.2f}**" if tot["wall_s"] else "- wall_s: —")
    md.append(f"- POSTs: **{summary['post_count']}**")
    md.append(f"- total prompt tokens (sum across POSTs): {tot['total_prompt_tokens']}")
    md.append(f"- total decode tokens: {tot['total_decode_tokens']}")
    md.append(f"- decode tps avg: {tot['decode_tps_avg']:.2f}" if tot["decode_tps_avg"] else "- decode tps avg: —")
    md.append(f"- weighted acceptance: {tot['weighted_acceptance']}")
    md.append(f"- total cache hit tokens: {tot['total_cache_hit_tokens']}")
    if tot.get("first_tool_call_ms_avg") is not None:
        md.append(f"- first_tool_call_ms (avg over POSTs that emit a tool call): {tot['first_tool_call_ms_avg']:.0f}")
    md.append("")
    cs = summary.get("cache_summary")
    if cs:
        md.append(
            f"- prefix-cache: lookups={cs.get('n_lookups')} hits={cs.get('n_hits')} "
            f"hit_rate={cs.get('hit_rate'):.2%}".rstrip()
            if isinstance(cs.get("hit_rate"), float)
            else f"- prefix-cache: lookups={cs.get('n_lookups')} hits={cs.get('n_hits')}"
        )
    md.append("")
    md.append("## Per-POST")
    md.append("")
    md.append("| # | prompt | decode | wall_s | tps | accept | cache_hit | ttft_srv | prefill_srv | decode_srv | cycles | first_byte | first_content | first_tool | finish | src |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in summary["posts"]:
        v = _post_view(p)
        lm = p.get("landmarks") or {}
        sm = p.get("server_metric") or {}
        md.append(
            "| {idx} | {prompt} | {decode} | {wall} | {tps} | {accept} | {hit} | {ttft_s} | {pf_s} | {dc_s} | {cyc} | {fb} | {fc} | {ftc} | {fin} | {src} |".format(
                idx=p["idx"],
                prompt=v["prompt_tokens"] if v["prompt_tokens"] is not None else "—",
                decode=v["decode_tokens"] if v["decode_tokens"] is not None else "—",
                wall=f"{v['wall_s']:.2f}" if v["wall_s"] is not None else "—",
                tps=f"{v['tps']:.1f}" if v["tps"] is not None else "—",
                accept=f"{v['accept']*100:.1f}%" if v["accept"] is not None else "—",
                hit=v["cache_hit_tokens"] if v["cache_hit_tokens"] is not None else "—",
                ttft_s=_ms(sm.get("ttft_ms_server")),
                pf_s=_ms(sm.get("prefill_ms_server")),
                dc_s=_ms(sm.get("decode_ms_server")),
                cyc=sm.get("cycles_completed", "—") if sm.get("cycles_completed") is not None else "—",
                fb=_ms(lm.get("first_byte_ms")),
                fc=_ms(lm.get("first_content_token_ms")),
                ftc=_ms(lm.get("first_tool_call_sent_ms")),
                fin=lm.get("finish_reason") or "—",
                src=v["source"],
            )
        )
    md.append("")
    # cycle summary section (DFlash only, when events were captured)
    cycle_lines = []
    for p in summary["posts"]:
        sm = p.get("server_metric") or {}
        cyc = sm.get("cycles_summary")
        if cyc:
            cycle_lines.append(
                f"| {p['idx']} | {cyc['n_cycles']} | {cyc['total_commits']} | "
                f"{cyc['mean_acceptance_len']:.2f} | {cyc['mean_block_len']:.2f} | "
                f"{cyc['verify_us_p50']/1000:.1f} | {cyc['verify_us_p99']/1000:.1f} |"
            )
    if cycle_lines:
        md.append("## Cycle stats (per-POST)")
        md.append("")
        md.append("| # | cycles | commits | avg_accept_len | avg_block_len | verify_p50_ms | verify_p99_ms |")
        md.append("|---|---|---|---|---|---|---|")
        md.extend(cycle_lines)
        md.append("")
    if peer is not None:
        md.append(_render_peer_comparison(summary, peer))
        md.append("")
    md.append("## Workspace files")
    md.append("")
    for f in summary["workspace_files"]:
        md.append(f"- `{f['path']}` ({f['bytes']} bytes)")
    return "\n".join(md) + "\n"


def _render_peer_comparison(this: dict[str, Any], peer: dict[str, Any]) -> str:
    """Compute tool_call_latency_gap = this - peer (ms). Negative = this is faster."""
    md: list[str] = []
    this_label = this["metadata"]["label"]
    peer_label = peer["metadata"]["label"]
    md.append(f"## Verdict — {this_label} vs {peer_label}")
    md.append("")
    md.append("Per-POST `first_tool_call_ms` gap (this − peer; negative = this is faster):")
    md.append("")
    md.append("| # | this | peer | gap_ms |")
    md.append("|---|---|---|---|")
    posts_this = this.get("posts") or []
    posts_peer = peer.get("posts") or []
    n = min(len(posts_this), len(posts_peer))
    gaps: list[float] = []
    for i in range(n):
        a = (posts_this[i].get("landmarks") or {}).get("first_tool_call_sent_ms")
        b = (posts_peer[i].get("landmarks") or {}).get("first_tool_call_sent_ms")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            gap = float(a) - float(b)
            gaps.append(gap)
            md.append(f"| {i+1} | {a:.0f} | {b:.0f} | {gap:+.0f} |")
        else:
            md.append(f"| {i+1} | {_ms(a)} | {_ms(b)} | — |")
    md.append("")
    a_tot = (this.get("totals") or {}).get("first_tool_call_ms_sum")
    b_tot = (peer.get("totals") or {}).get("first_tool_call_ms_sum")
    if isinstance(a_tot, (int, float)) and isinstance(b_tot, (int, float)):
        gap_sum = a_tot - b_tot
        md.append(f"- **tool_call_latency_gap (sum)**: {gap_sum:+.0f} ms ({this_label} − {peer_label})")
    if gaps:
        avg = sum(gaps) / len(gaps)
        md.append(f"- **tool_call_latency_gap (avg per POST)**: {avg:+.0f} ms")
    a_wall = (this.get("totals") or {}).get("wall_s")
    b_wall = (peer.get("totals") or {}).get("wall_s")
    if isinstance(a_wall, (int, float)) and isinstance(b_wall, (int, float)) and b_wall > 0:
        md.append(f"- wall ratio (this / peer): {a_wall / b_wall:.3f} ({a_wall:.2f}s vs {b_wall:.2f}s)")
    return "\n".join(md)


if __name__ == "__main__":
    raise SystemExit(main())

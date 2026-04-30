# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

BUCKETS = (
    "rss",
    "mlx_active",
    "mlx_cache",
    "untracked",
    "system_wired",
    "target_fa_kv",
    "target_gdn_state",
    "rollback_tape",
    "draft_kv",
    "target_hidden_active",
    "gen_hidden_chunks",
    "l1_snapshot",
    "l2_disk",
)

def load_memory_events(path: Path) -> list[dict[str, Any]]:
    files: list[Path]
    if path.is_dir():
        files = [
            path / "cycle_events.jsonl",
            path / "post_events.jsonl",
            path / "cache_events.jsonl",
        ]
    else:
        files = [path]
    events: list[dict[str, Any]] = []
    for file in files:
        if not file.exists():
            continue
        with file.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "memory_phase" in row:
                    events.append(row)
                peak = row.get("memory_waterfall_peak")
                if isinstance(peak, dict) and peak:
                    clone = dict(peak)
                    clone.setdefault("memory_phase", "post_peak")
                    events.append(clone)
    return events

def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, dict[str, float]] = {}
    peaks: dict[str, float] = {bucket: 0.0 for bucket in BUCKETS}
    for event in events:
        phase = str(event.get("memory_phase", "unknown"))
        phase_row = phases.setdefault(phase, {})
        for bucket in BUCKETS:
            value = _bucket_gb(event, bucket)
            if value > peaks[bucket]:
                peaks[bucket] = value
            if value > phase_row.get(bucket, 0.0):
                phase_row[bucket] = value
    top = sorted(peaks.items(), key=lambda item: item[1], reverse=True)
    return {
        "event_count": len(events),
        "phases": phases,
        "deltas": compute_phase_deltas(events),
        "peaks_gb": peaks,
        "top_buckets": top,
        "verdict": classify_verdict(peaks),
    }

def compute_phase_deltas(events: list[dict[str, Any]]) -> dict[str, Any]:
    phase_deltas: dict[str, dict[str, float]] = {}
    top_deltas: list[dict[str, Any]] = []
    previous: dict[str, float] | None = None
    previous_phase = "start"
    for event in events:
        phase = str(event.get("memory_phase", "unknown"))
        if phase == "post_peak":
            continue
        current = {bucket: _bucket_gb(event, bucket) for bucket in BUCKETS}
        if previous is None:
            previous = current
            previous_phase = phase
            continue
        row = phase_deltas.setdefault(phase, {})
        for bucket in BUCKETS:
            delta = current[bucket] - previous.get(bucket, 0.0)
            if delta <= 0.0005:
                continue
            if delta > row.get(bucket, 0.0):
                row[bucket] = delta
            top_deltas.append(
                {
                    "from": previous_phase,
                    "to": phase,
                    "bucket": bucket,
                    "delta_gb": delta,
                }
            )
        previous = current
        previous_phase = phase
    top_deltas.sort(key=lambda item: float(item["delta_gb"]), reverse=True)
    return {
        "by_phase": phase_deltas,
        "top_positive": top_deltas[:20],
    }

def classify_verdict(peaks_gb: dict[str, float]) -> str:
    fa_kv = float(peaks_gb.get("target_fa_kv", 0.0) or 0.0)
    target_hidden = float(peaks_gb.get("target_hidden_active", 0.0) or 0.0)
    l1 = float(peaks_gb.get("l1_snapshot", 0.0) or 0.0)
    mlx_cache = float(peaks_gb.get("mlx_cache", 0.0) or 0.0)
    untracked = float(peaks_gb.get("untracked", 0.0) or 0.0)
    top_name, top_value = max(peaks_gb.items(), key=lambda item: item[1])
    if top_value <= 0.0:
        return "no memory waterfall data"
    if top_name == "l1_snapshot" or l1 >= max(target_hidden, fa_kv, mlx_cache, untracked):
        return "optimize L1/L2 pressure"
    if top_name == "target_fa_kv" or fa_kv >= max(target_hidden, l1, mlx_cache, untracked):
        return "paged/quantized KV worth investigating"
    if top_name in ("mlx_cache", "untracked") or max(mlx_cache, untracked) >= max(target_hidden, fa_kv, l1):
        return "allocator/scratch/cache policy first"
    if target_hidden > fa_kv:
        return "optimize target_hidden first"
    return f"inspect {top_name}"

def render_summary(summary: dict[str, Any], *, include_delta: bool = False) -> str:
    lines: list[str] = []
    lines.append(f"events: {summary['event_count']}")
    lines.append("")
    lines.append("phase | top buckets")
    lines.append("--- | ---")
    for phase, row in sorted(summary["phases"].items()):
        top = sorted(row.items(), key=lambda item: item[1], reverse=True)[:5]
        rendered = ", ".join(f"{name}={value:.3f}GB" for name, value in top if value > 0.0)
        lines.append(f"{phase} | {rendered or 'none'}")
    lines.append("")
    lines.append("peak bucket | GB")
    lines.append("--- | ---")
    for name, value in summary["top_buckets"]:
        if value > 0.0:
            lines.append(f"{name} | {value:.3f}")
    if include_delta:
        lines.append("")
        lines.append("delta phase | top positive deltas")
        lines.append("--- | ---")
        by_phase = summary.get("deltas", {}).get("by_phase", {})
        for phase, row in sorted(by_phase.items()):
            top = sorted(row.items(), key=lambda item: item[1], reverse=True)[:5]
            rendered = ", ".join(
                f"{name}=+{value:.3f}GB" for name, value in top if value > 0.0
            )
            lines.append(f"{phase} | {rendered or 'none'}")
        lines.append("")
        lines.append("top positive transitions")
        lines.append("---")
        for row in summary.get("deltas", {}).get("top_positive", [])[:10]:
            lines.append(
                f"{row['from']} -> {row['to']}: {row['bucket']} +{row['delta_gb']:.3f}GB"
            )
    lines.append("")
    lines.append(f"verdict: {summary['verdict']}")
    return "\n".join(lines)

def _est_tokens_chars(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

class _Tokenizer:
    def __init__(self, ref: Optional[str]):
        self.ref = ref
        self._impl = None
        if ref:
            try:
                from mlx_lm import load as _load

                _, self._impl = _load(ref, lazy=True)
            except Exception as exc:
                sys.stderr.write(
                    f"warn: tokenizer load failed ({exc}); falling back to char estimate\n"
                )
                self._impl = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._impl is None:
            return _est_tokens_chars(text)
        try:
            ids = self._impl.encode(text)
        except Exception:
            return _est_tokens_chars(text)
        return len(ids)

def _segment_message(msg: dict[str, Any]) -> list[tuple[str, str]]:
    role = str(msg.get("role") or "unknown")
    out: list[tuple[str, str]] = []
    content = msg.get("content")
    if isinstance(content, str):
        if content:
            out.append((f"msg.{role}.content", content))
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str) and text:
                    out.append((f"msg.{role}.content_block", text))
    tool_calls = msg.get("tool_calls") or []
    for tool_call in tool_calls:
        fn = tool_call.get("function") or {}
        name = fn.get("name") or "unknown"
        args = fn.get("arguments")
        if isinstance(args, str) and args:
            out.append((f"msg.{role}.tool_call.{name}.args", args))
        elif isinstance(args, dict):
            out.append((f"msg.{role}.tool_call.{name}.args", json.dumps(args)))
    if role == "tool":
        tool_call_id = msg.get("tool_call_id") or "?"
        if isinstance(content, str) and content:
            out.append((f"msg.tool.result.{tool_call_id}", content))
    return out

def _segment_tools(tools: list[dict[str, Any]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for tool in tools or []:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name") or "unknown"
        out.append((f"tool_def.{name}", json.dumps(fn, sort_keys=True)))
    return out

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def analyze_request(body: dict[str, Any], tok: _Tokenizer) -> dict[str, Any]:
    messages = body.get("messages") or []
    tools = body.get("tools") or []

    segments: list[dict[str, Any]] = []

    for label, text in _segment_tools(tools):
        segments.append(
            {
                "category": "tool_def",
                "label": label,
                "tokens": tok.count(text),
                "chars": len(text),
                "hash": _content_hash(text),
            }
        )

    for message in messages:
        for label, text in _segment_message(message):
            cat_root = label.split(".")[0]
            sub = label.split(".")[1] if "." in label else ""
            if cat_root == "msg":
                if sub == "system":
                    category = "system"
                elif "tool_call" in label:
                    category = "tool_call_args"
                elif sub == "tool":
                    category = "tool_result"
                else:
                    category = f"msg_{sub}"
            else:
                category = cat_root
            segments.append(
                {
                    "category": category,
                    "label": label,
                    "tokens": tok.count(text),
                    "chars": len(text),
                    "hash": _content_hash(text),
                }
            )

    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tokens": 0, "chars": 0, "n": 0}
    )
    total_tokens = 0
    total_chars = 0
    for segment in segments:
        category = segment["category"]
        by_category[category]["tokens"] += segment["tokens"]
        by_category[category]["chars"] += segment["chars"]
        by_category[category]["n"] += 1
        total_tokens += segment["tokens"]
        total_chars += segment["chars"]

    return {
        "n_messages": len(messages),
        "n_tools": len(tools),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "by_category": {key: dict(value) for key, value in by_category.items()},
        "segments": segments,
    }

def analyze_run_dir(run_dir: Path, tok: _Tokenizer) -> dict[str, Any]:
    requests_dir = run_dir / "requests"
    if not requests_dir.is_dir():
        raise FileNotFoundError(f"no requests/ dir in {run_dir}")
    request_files = sorted(requests_dir.glob("*.json"))

    per_request: list[dict[str, Any]] = []
    segment_hash_count: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "tokens": 0, "label": None, "category": None}
    )

    for request_file in request_files:
        try:
            obj = json.loads(request_file.read_text())
        except Exception as exc:
            sys.stderr.write(f"warn: failed to read {request_file}: {exc}\n")
            continue
        body = obj.get("body") or {}
        analysis = analyze_request(body, tok)
        analysis["request_file"] = request_file.name
        analysis["idx"] = obj.get("idx")
        analysis["model"] = body.get("model")
        per_request.append(analysis)
        for segment in analysis["segments"]:
            digest = segment["hash"]
            segment_hash_count[digest]["count"] += 1
            segment_hash_count[digest]["tokens"] = segment["tokens"]
            segment_hash_count[digest]["label"] = segment["label"]
            segment_hash_count[digest]["category"] = segment["category"]

    dedupe_summary: dict[str, Any] = {
        "unique_segments": len(segment_hash_count),
        "total_segment_instances": sum(value["count"] for value in segment_hash_count.values()),
        "redundant_token_instances": sum(
            value["tokens"] * (value["count"] - 1)
            for value in segment_hash_count.values()
            if value["count"] > 1
        ),
        "top_repeated_segments": sorted(
            [
                {
                    "label": value["label"],
                    "category": value["category"],
                    "tokens": value["tokens"],
                    "count": value["count"],
                    "redundant_tokens": value["tokens"] * (value["count"] - 1),
                }
                for value in segment_hash_count.values()
                if value["count"] > 1
            ],
            key=lambda row: row["redundant_tokens"],
            reverse=True,
        )[:10],
    }

    return {
        "run_dir": str(run_dir),
        "n_requests": len(per_request),
        "per_request": per_request,
        "dedupe": dedupe_summary,
    }

def render_prompt_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    run_dir = report["run_dir"]
    lines.append(f"# Prompt composition analysis - {run_dir}\n")
    lines.append(f"Requests analyzed: **{report['n_requests']}**\n")
    if not report["per_request"]:
        lines.append("(no requests parsed)\n")
        return "\n".join(lines)

    categories: dict[str, dict[str, int]] = defaultdict(lambda: {"tokens": 0, "n": 0})
    for row in report["per_request"]:
        for category, value in row["by_category"].items():
            categories[category]["tokens"] += value["tokens"]
            categories[category]["n"] += value["n"]
    total_tokens = sum(category["tokens"] for category in categories.values())
    lines.append("## Aggregate token breakdown across all POSTs\n")
    lines.append("| category | total tokens | share | n segments |")
    lines.append("|---|---:|---:|---:|")
    for category, value in sorted(categories.items(), key=lambda item: -item[1]["tokens"]):
        share = (value["tokens"] / total_tokens * 100) if total_tokens else 0
        lines.append(f"| {category} | {value['tokens']} | {share:.1f}% | {value['n']} |")
    lines.append(f"| **total** | **{total_tokens}** | 100.0% | |\n")

    lines.append("## Per-request totals (tokens)\n")
    category_keys = sorted({category for row in report["per_request"] for category in row["by_category"]})
    header = ["#", "model", "n_msg", "n_tools", "total"] + category_keys
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in report["per_request"]:
        rendered_row = [
            str(row.get("idx") if row.get("idx") is not None else row["request_file"]),
            (row.get("model") or "?")[:40],
            str(row["n_messages"]),
            str(row["n_tools"]),
            str(row["total_tokens"]),
        ]
        for category in category_keys:
            rendered_row.append(str((row["by_category"].get(category) or {}).get("tokens", 0)))
        lines.append("| " + " | ".join(rendered_row) + " |")

    lines.append("\n## Cross-request dedupe\n")
    dedupe = report["dedupe"]
    lines.append(f"- Unique segments: **{dedupe['unique_segments']}**")
    lines.append(f"- Total segment instances: **{dedupe['total_segment_instances']}**")
    lines.append(f"- Redundant tokens if deduped: **{dedupe['redundant_token_instances']}**\n")
    if dedupe["top_repeated_segments"]:
        lines.append("Top 10 repeated segments by redundant token cost:\n")
        lines.append("| label | category | tokens/instance | count | redundant tokens |")
        lines.append("|---|---|---:|---:|---:|")
        for row in dedupe["top_repeated_segments"]:
            lines.append(
                f"| {row['label'][:60]} | {row['category']} | {row['tokens']} | "
                f"{row['count']} | {row['redundant_tokens']} |"
            )

    if len(report["per_request"]) > 1:
        lines.append("\n## Per-turn growth\n")
        lines.append("| from->to | delta total | delta tool_def | delta system | delta msg_user | delta msg_assistant | delta tool_result |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        previous = None
        for row in report["per_request"]:
            if previous is not None:

                def _diff(category: str) -> int:
                    return (
                        (row["by_category"].get(category) or {}).get("tokens", 0)
                        - (previous["by_category"].get(category) or {}).get("tokens", 0)
                    )

                lines.append(
                    f"| {previous.get('idx', previous['request_file'])}->{row.get('idx', row['request_file'])} "
                    f"| {row['total_tokens'] - previous['total_tokens']} "
                    f"| {_diff('tool_def')} "
                    f"| {_diff('system')} "
                    f"| {_diff('msg_user')} "
                    f"| {_diff('msg_assistant')} "
                    f"| {_diff('tool_result')} |"
                )
            previous = row

    return "\n".join(lines) + "\n"

def _bucket_gb(event: dict[str, Any], bucket: str) -> float:
    gb_key = f"{bucket}_gb"
    if gb_key in event:
        return _float(event.get(gb_key))
    bytes_key = f"{bucket}_bytes"
    if bytes_key in event:
        return _float(event.get(bytes_key)) / 1_000_000_000.0
    return 0.0

def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0

def run_memory(args: argparse.Namespace) -> int:
    events = load_memory_events(args.path)
    summary = summarize_events(events)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(render_summary(summary, include_delta=args.delta))
    return 0

def run_prompt(args: argparse.Namespace) -> int:
    tok = _Tokenizer(args.tokenizer)
    reports: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        path = Path(run_dir)
        if not path.is_dir():
            sys.stderr.write(f"skip: {run_dir} not a dir\n")
            continue
        reports.append(analyze_run_dir(path, tok))

    bundle = {"reports": reports, "tokenizer_ref": args.tokenizer}
    if args.out:
        Path(args.out).write_text(json.dumps(bundle, indent=2))
        print(f"wrote {args.out}")
    markdown = "\n---\n".join(render_prompt_markdown(report) for report in reports)
    if args.md:
        Path(args.md).write_text(markdown)
        print(f"wrote {args.md}")
    else:
        print(markdown)
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze DFlash trace artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    memory = subparsers.add_parser("memory", help="summarize memory-waterfall events")
    memory.add_argument("path", type=Path)
    memory.add_argument("--json", action="store_true")
    memory.add_argument("--delta", action="store_true")
    memory.set_defaults(func=run_memory)

    prompt = subparsers.add_parser("prompt", help="summarize prompt composition from trace requests")
    prompt.add_argument("--run-dir", action="append", required=True, help="agentic trace run dir; can repeat")
    prompt.add_argument("--tokenizer", default=None, help="HF model ref for tokenizer; optional")
    prompt.add_argument("--out", default=None, help="JSON output path")
    prompt.add_argument("--md", default=None, help="Markdown output path")
    prompt.set_defaults(func=run_prompt)

    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())

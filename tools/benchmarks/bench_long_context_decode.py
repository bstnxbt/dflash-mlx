# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from dflash_mlx.artifacts import create_run_dir

REPO_ROOT = Path(__file__).resolve().parents[2]

def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def _make_prompt_text(target_tokens: int, tokenizer) -> tuple[str, int]:
    src = (REPO_ROOT / "dflash_mlx" / "runtime.py").read_text()
    text = src
    ids = tokenizer.encode(text)
    while len(ids) < target_tokens:
        text = text + "\n# ----\n" + src
        ids = tokenizer.encode(text)
    ids = ids[:target_tokens]
    text = tokenizer.decode(ids)
    return text, len(ids)

def _post_stream(url: str, body: dict, timeout_s: float) -> dict:
    req_t0 = time.perf_counter()
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer bench"},
        method="POST",
    )
    first_byte_t = None
    end_t = None
    last_chunk_text = ""
    usage = None
    n_chunks = 0
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            if first_byte_t is None:
                first_byte_t = time.perf_counter()
            buf += chunk
            while b"\n\n" in buf:
                event_block, _, buf = buf.partition(b"\n\n")
                lines = event_block.split(b"\n")
                for ln in lines:
                    if not ln.startswith(b"data:"):
                        continue
                    payload = ln[len(b"data:"):].strip()
                    if payload == b"[DONE]":
                        end_t = time.perf_counter()
                        continue
                    try:
                        ev = json.loads(payload)
                    except Exception:
                        continue
                    n_chunks += 1
                    if isinstance(ev.get("usage"), dict):
                        usage = ev["usage"]
                    for ch in ev.get("choices") or []:
                        delta = ch.get("delta") or {}
                        c = delta.get("content")
                        if isinstance(c, str):
                            last_chunk_text = c
                        if ch.get("finish_reason"):
                            finish_reason = ch["finish_reason"]
        if end_t is None:
            end_t = time.perf_counter()
    return {
        "request_wall_s": end_t - req_t0,
        "ttft_s": (first_byte_t - req_t0) if first_byte_t else None,
        "decode_wall_s": (end_t - first_byte_t) if first_byte_t else None,
        "n_chunks": n_chunks,
        "usage": usage,
        "finish_reason": finish_reason,
    }

def _read_dflash_events(events_dir: Path) -> dict[str, Any] | None:
    pe_path = events_dir / "post_events.jsonl"
    ce_path = events_dir / "cycle_events.jsonl"
    if not pe_path.exists():
        return None
    posts = []
    for ln in pe_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            posts.append(json.loads(ln))
        except Exception:
            pass
    if not posts:
        return None
    pe = posts[-1]
    rid = pe.get("request_id")
    cycles = []
    if ce_path.exists():
        for ln in ce_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                ev = json.loads(ln)
                if ev.get("request_id") == rid:
                    cycles.append(ev)
            except Exception:
                pass
    cyc_summary = None
    if cycles:
        n = len(cycles)
        commits = sum(c.get("commit_count", 0) for c in cycles)
        verify_us = sorted(c.get("verify_us", 0.0) for c in cycles)
        accept = sorted(c.get("acceptance_len", 0) for c in cycles)
        cyc_summary = {
            "n_cycles": n,
            "total_commits": commits,
            "tokens_per_cycle": commits / n,
            "mean_acceptance_len": sum(accept) / n,
            "verify_us_p50": verify_us[n // 2],
            "verify_us_p99": verify_us[min(n - 1, max(0, int(n * 0.99) - 1))],
        }
    return {
        "post": pe,
        "cycles_summary": cyc_summary,
    }

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["dflash", "mlxlm"], required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ctx-tokens", type=int, required=True,
                   help="approximate prompt tokens (filler size; chat template adds ~30 tok)")
    p.add_argument("--decode-tokens", type=int, default=512)
    p.add_argument("--label", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--events-dir", default=None,
                   help="dflash events dir (post_events.jsonl); only used when backend=dflash")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else create_run_dir("benchmark", "long-context-decode")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stderr.write(f"Output: {out_dir}\n")

    sys.stderr.write("[bench] loading tokenizer...\n")
    from mlx_lm import load as _load
    _, tokenizer = _load(args.target, lazy=True)

    sys.stderr.write(f"[bench] building {args.ctx_tokens}-tok prompt...\n")
    text, exact_tok = _make_prompt_text(args.ctx_tokens, tokenizer)

    user_message = (
        text
        + "\n\n---\n\nNow write a single Python function that returns the integer 42, "
        "with no explanation. Just the function definition."
    )

    body = {
        "model": args.target,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": args.decode_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"

    runs = []
    for i in range(args.runs):
        sys.stderr.write(f"[bench] run {i+1}/{args.runs} ctx={args.ctx_tokens} backend={args.backend}\n")

        events_summary = None
        events_dir = Path(args.events_dir) if args.events_dir else None
        if events_dir and events_dir.exists():
            for f in ("post_events.jsonl", "cycle_events.jsonl", "cache_events.jsonl"):
                p_ = events_dir / f
                if p_.exists():
                    p_.write_text("")
        t0 = time.perf_counter()
        try:
            r = _post_stream(url, body, timeout_s=args.timeout_s)
        except Exception as e:
            sys.stderr.write(f"[bench] run {i+1} FAILED: {e!r}\n")
            r = {"error": repr(e), "request_wall_s": time.perf_counter() - t0}

        if events_dir and args.backend == "dflash":
            events_summary = _read_dflash_events(events_dir)
        usage = (r.get("usage") or {}) if r else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        wall_s = r.get("decode_wall_s")
        decode_tps = (completion_tokens / wall_s) if (completion_tokens and wall_s and wall_s > 0) else None
        runs.append({
            "run_idx": i + 1,
            "prompt_filler_tokens_target": args.ctx_tokens,
            "prompt_tokens_server": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_s": r.get("ttft_s"),
            "decode_wall_s": wall_s,
            "request_wall_s": r.get("request_wall_s"),
            "decode_tps": decode_tps,
            "finish_reason": r.get("finish_reason"),
            "n_chunks": r.get("n_chunks"),
            "error": r.get("error"),
            "events": events_summary,
        })

    summary = {
        "label": args.label,
        "backend": args.backend,
        "target": args.target,
        "ctx_target": args.ctx_tokens,
        "decode_tokens_target": args.decode_tokens,
        "runs": runs,
        "started_at": _now(),
        "host_port": args.port,
    }
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    sys.stderr.write(f"[bench] wrote {out_path}\n")

    print(f"label={args.label} ctx={args.ctx_tokens} backend={args.backend}")
    for r in runs:
        print(
            f"  run {r['run_idx']}: prompt={r['prompt_tokens_server']} "
            f"decode={r['completion_tokens']} ttft={r['ttft_s']} "
            f"tps={r['decode_tps']:.1f}" if r.get("decode_tps") else
            f"  run {r['run_idx']}: ERROR {r.get('error')}"
        )
        if r.get("events") and r["events"].get("cycles_summary"):
            cs = r["events"]["cycles_summary"]
            print(
                f"    cycles={cs['n_cycles']} tpc={cs['tokens_per_cycle']:.2f} "
                f"accept_len={cs['mean_acceptance_len']:.2f} "
                f"verify_p50={cs['verify_us_p50']/1000:.1f}ms"
            )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

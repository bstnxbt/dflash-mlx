# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import tempfile
import time
from collections.abc import Sequence
from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.cache.prefix_l2 import DFlashPrefixL2Cache
from dflash_mlx.generate import load_runtime_components
from dflash_mlx.cache.codecs import (
    build_snapshot,
    target_cache_is_serializable,
)
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.runtime import stream_dflash_generate
from dflash_mlx.runtime_context import build_runtime_context, runtime_config_from_profile

def rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)

def mlx_active_mb() -> float:
    try:
        return mx.get_active_memory() / (1024 * 1024)
    except Exception:
        return 0.0

def _tokenize(tok: Any, text: str) -> list[int]:
    if hasattr(tok, "encode"):
        return [int(t) for t in tok.encode(text)]
    return [int(t) for t in tok(text)]

def _build_key(
    target_ref: str,
    draft_ref: str,
    draft_model: Any,
    runtime_context: Any,
) -> DFlashPrefixKey:
    capture = tuple(int(x) for x in getattr(draft_model, "target_layer_ids", ()) or ())
    runtime_config = runtime_context.runtime
    return DFlashPrefixKey(
        target_model_id=target_ref,
        draft_model_id=draft_ref,
        capture_layer_ids=capture,
        draft_sink_size=int(runtime_config.draft_sink_size),
        draft_window_size=int(runtime_config.draft_window_size),
    )

def _make_long_prompt(tok: Any, n_tokens: int, label: str) -> list[int]:
    seed = (
        f"Session {label}. Engineering log entry. We are reasoning about "
        "speculative decoding on Apple Silicon and the dflash runtime. "
    )
    filler = (
        f"[{label}] runtime note: bf16 KV cache, GQA heads, FA + GDN hybrid layers, "
        "verify kernel rollback semantics, prefill cost dominates short replies. "
    )
    out = _tokenize(tok, seed)
    while len(out) < n_tokens:
        out.extend(_tokenize(tok, filler))
    out = out[:n_tokens]
    out.extend(_tokenize(tok, "\nAssistant:"))
    return out

def _run_turn(
    *,
    target_model,
    tokenizer,
    draft_model,
    prompt_tokens: list[int],
    max_tokens: int,
    cache: DFlashPrefixCache,
    key: DFlashPrefixKey,
    runtime_context: Any,
) -> dict[str, Any]:
    matched, prefix_snap = cache.lookup(prompt_tokens, key)
    start = time.perf_counter_ns()
    first_us: Optional[float] = None
    n_tokens = 0
    from_snap = False
    breakdown: dict[str, Any] = {}
    stream = stream_dflash_generate(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt="",
        max_new_tokens=max_tokens,
        use_chat_template=False,
        prompt_tokens_override=prompt_tokens,
        prefix_snapshot=prefix_snap,
        runtime_context=runtime_context,
    )
    try:
        for ev in stream:
            kind = ev.get("event")
            if kind == "prefill":
                breakdown = {
                    "prefill_us": float(ev.get("prefill_us") or 0.0),
                    "rebuild_us": float(ev.get("phase_rebuild_us") or 0.0),
                    "cold_us": float(ev.get("phase_cold_us") or 0.0),
                }
                continue
            if kind == "prefill_snapshot_ready":
                from_snap = bool(ev.get("from_snapshot", False))
                tc = ev.get("target_cache")
                th = ev.get("target_hidden")
                ll = ev.get("last_logits")
                tk = ev.get("token_ids") or []
                if (
                    tc is not None
                    and th is not None
                    and ll is not None
                    and target_cache_is_serializable(tc)
                ):
                    snap = build_snapshot(
                        token_ids=list(tk),
                        target_cache=tc,
                        target_hidden=th,
                        last_logits=ll,
                        key=key,
                    )
                    cache.insert(snap)
                continue
            if kind == "token":
                if first_us is None:
                    first_us = (time.perf_counter_ns() - start) / 1_000.0
                n_tokens += 1
                if n_tokens >= max_tokens:
                    break
    finally:
        stream.close()
    total_us = (time.perf_counter_ns() - start) / 1_000.0
    return {
        "ttft_us": first_us,
        "total_us": total_us,
        "n_tokens": n_tokens,
        "prompt_len": len(prompt_tokens),
        "matched_lookup": matched,
        "from_snapshot": from_snap,
        **breakdown,
    }

def _run_session(
    *,
    target_model,
    tokenizer,
    draft_model,
    prompts: list[list[int]],
    max_tokens: int,
    cache: DFlashPrefixCache,
    key: DFlashPrefixKey,
    runtime_context: Any,
    label: str,
    revisit_indices: list[int],
) -> dict[str, Any]:
    print(f"\n=== Session: {label} ===")
    baseline_mlx = mlx_active_mb()
    baseline_rss = rss_mb()
    print(f"  baseline   MLX={baseline_mlx:7.0f} MB  RSS={baseline_rss:7.0f} MB")

    turns: list[dict[str, Any]] = []
    mlx_trajectory: list[float] = []
    rss_trajectory: list[float] = []

    for i, prompt in enumerate(prompts, 1):
        r = _run_turn(
            target_model=target_model, tokenizer=tokenizer, draft_model=draft_model,
            prompt_tokens=prompt, max_tokens=max_tokens, cache=cache, key=key,
            runtime_context=runtime_context,
        )

        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass
        mlx_now = mlx_active_mb()
        rss_now = rss_mb()
        l1_stats = cache.stats()
        l2_stats = l1_stats.get("l2") or {}
        r["mlx_after_mb"] = mlx_now
        r["rss_after_mb"] = rss_now
        r["mlx_delta_mb"] = mlx_now - baseline_mlx
        r["l1_entries"] = l1_stats["current_entries"]
        r["l1_evictions"] = l1_stats["evictions"]
        r["l2_writes"] = l2_stats.get("writes", 0)
        r["l2_hits"] = l1_stats.get("l2_hits", 0)
        turns.append(r)
        mlx_trajectory.append(mlx_now)
        rss_trajectory.append(rss_now)
        print(
            f"  turn {i:02d}  ttft={r['ttft_us']/1000:7.1f}ms "
            f"prefill={r.get('prefill_us', 0)/1000:7.1f}ms "
            f"from_snap={str(r['from_snapshot']):>5s}  "
            f"L1 entries={l1_stats['current_entries']}/{l1_stats['max_entries']}  "
            f"evict={l1_stats['evictions']}  L2 writes={l2_stats.get('writes', 0)}  "
            f"MLX={mlx_now:7.0f} MB (Δ {mlx_now - baseline_mlx:+7.0f})  "
            f"RSS={rss_now:7.0f} MB"
        )

    print(f"\n  -- Revisits (turn indices, 1-based): {revisit_indices}")
    revisits: list[dict[str, Any]] = []
    for idx in revisit_indices:
        r = _run_turn(
            target_model=target_model, tokenizer=tokenizer, draft_model=draft_model,
            prompt_tokens=prompts[idx - 1], max_tokens=max_tokens, cache=cache, key=key,
            runtime_context=runtime_context,
        )
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass
        revisits.append({"turn": idx, **r})
        l1_stats = cache.stats()
        l2_stats = l1_stats.get("l2") or {}
        print(
            f"  revisit turn {idx:02d}  ttft={r['ttft_us']/1000:7.1f}ms "
            f"prefill={r.get('prefill_us', 0)/1000:7.1f}ms "
            f"from_snap={str(r['from_snapshot']):>5s} "
            f"L1 evict={l1_stats['evictions']} L2 hits={l1_stats.get('l2_hits', 0)}"
        )

    mlx_max = max(mlx_trajectory)
    mlx_last = mlx_trajectory[-1]
    mlx_growth_per_turn = (mlx_trajectory[-1] - mlx_trajectory[max(0, len(mlx_trajectory) // 2)]) / max(
        1, len(mlx_trajectory) - len(mlx_trajectory) // 2
    )
    return {
        "label": label,
        "baseline_mlx_mb": baseline_mlx,
        "baseline_rss_mb": baseline_rss,
        "turns": turns,
        "revisits": revisits,
        "l1_stats_final": cache.stats(),
        "mlx_max_mb": mlx_max,
        "mlx_last_mb": mlx_last,
        "mlx_growth_per_turn_second_half": mlx_growth_per_turn,
        "mlx_trajectory": mlx_trajectory,
        "rss_trajectory": rss_trajectory,
    }

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target", default="mlx-community/Qwen3.6-27B-4bit",
                   help="HF target model ref")
    p.add_argument("--draft", default="z-lab/Qwen3.6-27B-DFlash",
                   help="HF draft model ref")
    p.add_argument("--turns", type=int, default=20,
                   help="number of distinct prompts in the build phase")
    p.add_argument("--system-tokens", type=int, default=4000,
                   help="approximate prompt size in tokens")
    p.add_argument("--max-tokens", type=int, default=16,
                   help="tokens generated per turn")
    p.add_argument("--revisits", default="1,5,10",
                   help="comma-separated 1-based indices to revisit "
                        "after the build phase (empty = no revisits)")
    p.add_argument("--l1-max-entries", type=int, default=3)
    p.add_argument("--l1-max-bytes", type=int, default=2 * 1024**3,
                   help="L1 byte budget (default 2 GiB)")
    p.add_argument("--l2-max-bytes", type=int, default=40 * 1024**3,
                   help="L2 byte budget on tempfs (default 40 GiB)")
    p.add_argument("--control", action=argparse.BooleanOptionalAction, default=True,
                   help="run a no-L2 control pass first to prove L2 value")
    p.add_argument("--out", default=None,
                   help="JSON output path (auto-generated if omitted)")
    return p.parse_args(list(argv) if argv is not None else None)

def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    target_ref = args.target
    draft_ref = args.draft
    n_turns = args.turns
    sys_tokens = args.system_tokens
    max_tokens = args.max_tokens
    revisits = [int(x) for x in args.revisits.split(",") if x.strip()]
    l1_max_entries = args.l1_max_entries
    l1_max_bytes = args.l1_max_bytes
    l2_max_bytes = args.l2_max_bytes
    run_control = args.control

    print(f"Target={target_ref}")
    print(f"Draft ={draft_ref}")
    print(f"N turns={n_turns}  system_tokens={sys_tokens}  max_tokens={max_tokens}  revisits={revisits}")
    print(f"L1 max_entries={l1_max_entries}  max_bytes={l1_max_bytes/(1024**3):.1f} GB  "
          f"L2 max_bytes={l2_max_bytes/(1024**3):.0f} GB  control_pass={run_control}")

    target_model, tokenizer, draft_model, resolved_draft = load_runtime_components(
        model_ref=target_ref,
        draft_ref=draft_ref,
    )
    runtime_context = build_runtime_context(runtime_config_from_profile(profile="balanced"))
    print(f"Loaded. resolved_draft={resolved_draft}")

    prompts = [_make_long_prompt(tokenizer, sys_tokens, f"P{i:02d}") for i in range(1, n_turns + 1)]
    print(f"Built {len(prompts)} distinct prompts, lens={[len(p) for p in prompts][:3]} ...")

    print("Warmup (8 tokens, no L2)...")
    warm_cache = DFlashPrefixCache(max_entries=1, max_bytes=4 * 1024 * 1024 * 1024)
    warm_key = _build_key(target_ref, resolved_draft or draft_ref, draft_model, runtime_context)
    _run_turn(
        target_model=target_model, tokenizer=tokenizer, draft_model=draft_model,
        prompt_tokens=prompts[0][:64], max_tokens=8, cache=warm_cache, key=warm_key,
        runtime_context=runtime_context,
    )
    del warm_cache
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    ctl: Optional[dict[str, Any]] = None
    if run_control:

        print("\n#### PASS 1: control (no L2) ####")
        cache_ctl = DFlashPrefixCache(max_entries=l1_max_entries, max_bytes=l1_max_bytes)
        key_ctl = _build_key(target_ref, resolved_draft or draft_ref, draft_model, runtime_context)
        ctl = _run_session(
            target_model=target_model, tokenizer=tokenizer, draft_model=draft_model,
            prompts=prompts, max_tokens=max_tokens, cache=cache_ctl, key=key_ctl,
            runtime_context=runtime_context,
            label="control_no_L2", revisit_indices=revisits,
        )
        del cache_ctl
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    label = "PASS 2" if run_control else "PASS"
    print(f"\n#### {label}: with L2 (L1 + SSD tier) ####")
    with tempfile.TemporaryDirectory(prefix="dflash_l2_long_") as l2_dir:
        l2 = DFlashPrefixL2Cache(cache_dir=l2_dir, max_bytes=l2_max_bytes)
        cache_l2 = DFlashPrefixCache(
            max_entries=l1_max_entries, max_bytes=l1_max_bytes, l2=l2,
        )
        key_l2 = _build_key(target_ref, resolved_draft or draft_ref, draft_model, runtime_context)
        with_l2 = _run_session(
            target_model=target_model, tokenizer=tokenizer, draft_model=draft_model,
            prompts=prompts, max_tokens=max_tokens, cache=cache_l2, key=key_l2,
            runtime_context=runtime_context,
            label="with_L2", revisit_indices=revisits,
        )
        with_l2["l2_stats_final"] = l2.stats()
        l2.shutdown()

    print("\n#### Verdict ####")

    def _summary(s: dict[str, Any]) -> None:
        print(f"  [{s['label']}]")
        print(f"    baseline MLX           : {s['baseline_mlx_mb']:7.0f} MB")
        print(f"    MLX max during turns   : {s['mlx_max_mb']:7.0f} MB  "
              f"(Δ {s['mlx_max_mb'] - s['baseline_mlx_mb']:+7.0f})")
        print(f"    MLX last               : {s['mlx_last_mb']:7.0f} MB  "
              f"(Δ {s['mlx_last_mb'] - s['baseline_mlx_mb']:+7.0f})")
        print(f"    growth/turn (2nd half) : {s['mlx_growth_per_turn_second_half']:+7.2f} MB/turn")
        l1f = s["l1_stats_final"]
        print(f"    L1 final entries={l1f['current_entries']}  evictions={l1f['evictions']}  "
              f"l2_hits={l1f.get('l2_hits', 0)}  l2_misses={l1f.get('l2_misses', 0)}")
        revs = s["revisits"]
        from_snap = sum(1 for r in revs if r["from_snapshot"])
        avg_ttft = sum(r["ttft_us"] for r in revs) / max(1, len(revs)) / 1000.0
        print(f"    Revisits: {from_snap}/{len(revs)} from_snapshot, avg TTFT {avg_ttft:.1f} ms")

    if ctl is not None:
        _summary(ctl)
    _summary(with_l2)
    if "l2_stats_final" in with_l2:
        l2f = with_l2["l2_stats_final"]
        print(f"  L2 final stats: writes={l2f['writes']} hits={l2f['hits']} "
              f"write_drops={l2f['write_drops_queue_full']} write_errors={l2f['write_errors']} "
              f"load_errors={l2f['load_errors']} disk_MB={l2f['current_bytes']/(1024*1024):.0f}")

    ram_bounded = with_l2["mlx_growth_per_turn_second_half"] < 100.0
    all_revisits_l2hit = all(r["from_snapshot"] for r in with_l2["revisits"]) if with_l2["revisits"] else True
    all_revisits_ctl_cold = (
        all(not r["from_snapshot"] for r in ctl["revisits"]) if (ctl and ctl["revisits"]) else None
    )
    no_l2_errors = (
        with_l2.get("l2_stats_final", {}).get("write_errors", 0) == 0
        and with_l2.get("l2_stats_final", {}).get("load_errors", 0) == 0
        and with_l2.get("l2_stats_final", {}).get("materialize_errors", 0) == 0
        and with_l2.get("l2_stats_final", {}).get("write_drops_queue_full", 0) == 0
    )
    print()
    print(f"  RAM bounded (with_L2 2nd-half growth < 100 MB/turn) : {ram_bounded}  "
          f"({with_l2['mlx_growth_per_turn_second_half']:+.2f} MB/turn)")
    print(f"  All revisits L2-hit on with_L2                      : {all_revisits_l2hit}")
    if all_revisits_ctl_cold is not None:
        print(f"  All revisits COLD on control (proves L2 value)      : {all_revisits_ctl_cold}")
    print(f"  No L2 errors / drops                                : {no_l2_errors}")
    gates = [ram_bounded, all_revisits_l2hit, no_l2_errors]
    if all_revisits_ctl_cold is not None:
        gates.append(all_revisits_ctl_cold)
    verdict = all(gates)
    print(f"\n  VERDICT: {'PASS' if verdict else 'FAIL'}")

    out_path = args.out or (
        ".artifacts/dflash/benchmarks/"
        f"{time.strftime('%Y%m%d-%H%M%S')}-prefix-l2-long-session/summary.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "target": target_ref, "draft": draft_ref,
            "n_turns": n_turns, "system_tokens": sys_tokens, "max_tokens": max_tokens,
            "revisits": revisits,
            "control": ctl, "with_l2": with_l2,
            "ram_bounded": ram_bounded,
            "all_revisits_l2hit": all_revisits_l2hit,
            "all_revisits_ctl_cold": all_revisits_ctl_cold,
            "no_l2_errors": no_l2_errors,
            "verdict": verdict,
        }, f, indent=2, default=str)
    print(f"Results written to {out_path}")
    return 0 if verdict else 1

if __name__ == "__main__":
    sys.exit(main())

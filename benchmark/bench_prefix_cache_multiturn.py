"""Multi-turn bench comparing the DFlash runtime WITH vs WITHOUT the prefix cache.

Simulates an agentic session of N turns where each turn's prompt extends
the previous turn's prompt by a small user message (prompt grows monotonically).

Measures per turn:
  - TTFT (first token latency)
  - total wall (prefill + generation)
  - prompt_len

Prints a side-by-side table. Uses an in-process runtime — same model loaded
once, two runs (off then on). Same seed, same prompts, same max_tokens.

Usage:
    PYTHONPATH=$PWD .venv/bin/python -m benchmark.bench_prefix_cache_multiturn

Env knobs:
    BENCH_TARGET   HF ref for target (default mlx-community/Qwen3.6-27B-4bit)
    BENCH_DRAFT    HF ref for draft  (default z-lab/Qwen3.6-27B-DFlash)
    BENCH_TURNS    number of turns (default 10)
    BENCH_SYSTEM_TOKENS   approximate system prompt size (default 800)
    BENCH_MAX_TOKENS      tokens to generate per turn (default 48)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

from mlx_lm import stream_generate as mlxlm_stream_generate

from dflash_mlx.generate import load_runtime_components
from dflash_mlx.prefix_cache import (
    DFlashPrefixCache,
    DFlashPrefixKey,
    build_snapshot,
    target_cache_is_serializable,
)
from dflash_mlx.runtime import _resolve_draft_window, stream_dflash_generate


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return [int(t) for t in tokenizer.encode(text)]
    return [int(t) for t in tokenizer(text)]


def _build_key(target_ref: str, draft_ref: str, draft_model: Any) -> DFlashPrefixKey:
    capture = tuple(int(x) for x in getattr(draft_model, "target_layer_ids", ()) or ())
    sink, window = _resolve_draft_window()
    return DFlashPrefixKey(
        target_model_id=target_ref,
        draft_model_id=draft_ref,
        capture_layer_ids=capture,
        draft_sink_size=int(sink),
        draft_window_size=int(window),
    )


def _run_one_turn(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    prefix_snapshot=None,
    cache_to_populate: Optional[DFlashPrefixCache] = None,
    cache_key: Optional[DFlashPrefixKey] = None,
) -> dict[str, Any]:
    start = time.perf_counter_ns()
    first_token_us: Optional[float] = None
    n_tokens = 0
    from_snapshot = False
    snap_len_used = 0
    inserted = False
    stream = stream_dflash_generate(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt="",
        max_new_tokens=max_tokens,
        use_chat_template=False,
        prompt_tokens_override=prompt_tokens,
        prefix_snapshot=prefix_snapshot,
    )
    try:
        for event in stream:
            ev = event.get("event")
            if ev == "prefill_snapshot_ready":
                from_snapshot = bool(event.get("from_snapshot", False))
                snap_len_used = int(event.get("snap_prefix_len", 0))
                if cache_to_populate is not None and cache_key is not None:
                    tc = event.get("target_cache")
                    th = event.get("target_hidden")
                    ll = event.get("last_logits")
                    tk = event.get("token_ids") or []
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
                            key=cache_key,
                        )
                        cache_to_populate.insert(snap)
                        inserted = True
                continue
            if ev == "token":
                if first_token_us is None:
                    first_token_us = (time.perf_counter_ns() - start) / 1_000.0
                n_tokens += 1
                if n_tokens >= max_tokens:
                    break
    finally:
        stream.close()

    total_us = (time.perf_counter_ns() - start) / 1_000.0
    return {
        "ttft_us": first_token_us,
        "total_us": total_us,
        "n_tokens": n_tokens,
        "prompt_len": len(prompt_tokens),
        "from_snapshot": from_snapshot,
        "snap_prefix_len": snap_len_used,
        "inserted": inserted,
    }


def _build_turn_prompts(tokenizer: Any, n_turns: int, system_target_tokens: int) -> list[list[int]]:
    """Agentic-style prompts: shared system preamble, then growing user turns."""
    system_lorem = (
        "You are an expert software engineer in a long-running chat with a power user. "
        "The user is working on GPU programming, CUDA, Metal, and MLX runtime internals. "
        "Be precise, show code when useful, and cite specific Apple Silicon characteristics. "
        "If the user asks about speculative decoding, explain draft models, verify kernels, "
        "GQA, KV cache, and rollback semantics clearly. "
    ) * 6
    system_tokens = _tokenize(tokenizer, system_lorem)
    # Pad/trim to target size. The bench is meant to model long agentic
    # system/tool prefixes, so keep extending the stable prefix when requested.
    filler = (
        "Follow the repository's coding style. Preserve scope. Prefer small diffs. "
        "Explain runtime tradeoffs with concrete measurements. Avoid generated artifacts. "
    )
    while len(system_tokens) < system_target_tokens:
        system_tokens.extend(_tokenize(tokenizer, filler))
    if len(system_tokens) > system_target_tokens:
        system_tokens = system_tokens[:system_target_tokens]
    tool_preamble = _tokenize(
        tokenizer,
        "\n\nTools available: shell, python, search. Respond concisely.\n\n",
    )
    base = system_tokens + tool_preamble

    user_turns = [
        "User: What is a kernel in GPU programming?",
        "User: How does kernel fusion reduce memory bandwidth pressure?",
        "User: Explain speculative decoding in one paragraph.",
        "User: What's the difference between prefill and decode phases?",
        "User: Show a simple Metal kernel for a vector add.",
        "User: Why do we use GQA instead of full multi-head attention?",
        "User: Describe how a KV cache grows during long-context generation.",
        "User: What is rollback semantics when a draft token gets rejected?",
        "User: How does Apple Silicon's unified memory change runtime design?",
        "User: Summarize everything we've discussed in two lines.",
    ]
    assistant_filler = (
        " Yes. That's a good question. Let me explain briefly. "
    ) * 20
    assistant_cue = _tokenize(tokenizer, "\nAssistant:")

    prompts: list[list[int]] = []
    running = list(base)
    for i in range(n_turns):
        user_text = user_turns[i % len(user_turns)] + "\n"
        running = running + _tokenize(tokenizer, user_text)
        prompts.append(list(running + assistant_cue))
        # Simulate a reply that the caller appended before turn i+1.
        running = running + assistant_cue + _tokenize(
            tokenizer,
            assistant_filler + f" (reply #{i+1})\n",
        )
    return prompts


def _run_mlxlm_turn(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    """Run one turn through mlx_lm.stream_generate — honest AR baseline.

    Fresh implicit cache every turn (no session reuse), matching the typical
    naive mlx_lm caller pattern. This is what DFlash competes against
    head-to-head.
    """
    start = time.perf_counter_ns()
    first_token_us: Optional[float] = None
    n_tokens = 0
    for _response in mlxlm_stream_generate(
        model=target_model,
        tokenizer=tokenizer,
        prompt=prompt_tokens,
        max_tokens=max_tokens,
    ):
        if first_token_us is None:
            first_token_us = (time.perf_counter_ns() - start) / 1_000.0
        n_tokens += 1
        if n_tokens >= max_tokens:
            break
    total_us = (time.perf_counter_ns() - start) / 1_000.0
    return {
        "ttft_us": first_token_us,
        "total_us": total_us,
        "n_tokens": n_tokens,
        "prompt_len": len(prompt_tokens),
        "from_snapshot": False,
        "snap_prefix_len": 0,
        "inserted": False,
    }


def _session_mlxlm(
    *,
    target_model: Any,
    tokenizer: Any,
    prompts: list[list[int]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """mlx_lm baseline AR session, fresh cache each turn."""
    results: list[dict[str, Any]] = []
    for i, prompt_tokens in enumerate(prompts):
        res = _run_mlxlm_turn(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )
        res["turn"] = i + 1
        res["matched_lookup"] = 0
        results.append(res)
        print(
            f"  turn {i+1:02d} "
            f"prompt={res['prompt_len']:5d} "
            f"ttft={res['ttft_us']/1000:7.1f}ms "
            f"total={res['total_us']/1000:7.1f}ms "
            f"tokens={res['n_tokens']:3d}"
        )
    return results


def _session(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: Any,
    prompts: list[list[int]],
    max_tokens: int,
    use_cache: bool,
    cache: Optional[DFlashPrefixCache],
    key: Optional[DFlashPrefixKey],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i, prompt_tokens in enumerate(prompts):
        prefix_snapshot = None
        matched = 0
        if use_cache and cache is not None and key is not None:
            matched, prefix_snapshot = cache.lookup(prompt_tokens, key)
        res = _run_one_turn(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            prefix_snapshot=prefix_snapshot,
            cache_to_populate=cache if use_cache else None,
            cache_key=key if use_cache else None,
        )
        res["turn"] = i + 1
        res["matched_lookup"] = matched
        results.append(res)
        print(
            f"  turn {i+1:02d} "
            f"prompt={res['prompt_len']:5d} "
            f"ttft={res['ttft_us']/1000:7.1f}ms "
            f"total={res['total_us']/1000:7.1f}ms "
            f"tokens={res['n_tokens']:3d} "
            f"match={matched:5d} "
            f"from_snap={str(res['from_snapshot']):>5s}"
        )
    return results


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Multi-turn prefix-cache bench (DFlash with/without cache vs mlx_lm AR)."
    )
    p.add_argument("--target", default=os.environ.get("BENCH_TARGET", "mlx-community/Qwen3.6-27B-4bit"))
    p.add_argument("--draft", default=os.environ.get("BENCH_DRAFT", "z-lab/Qwen3.6-27B-DFlash"))
    p.add_argument("--turns", type=int, default=int(os.environ.get("BENCH_TURNS", "10")))
    p.add_argument("--system-tokens", type=int, default=int(os.environ.get("BENCH_SYSTEM_TOKENS", "800")))
    p.add_argument("--max-tokens", type=int, default=int(os.environ.get("BENCH_MAX_TOKENS", "48")))
    args = p.parse_args()
    target_ref = args.target
    draft_ref = args.draft
    n_turns = args.turns
    system_target_tokens = args.system_tokens
    max_tokens = args.max_tokens

    print(f"Target={target_ref}")
    print(f"Draft ={draft_ref}")
    print(f"Turns={n_turns}  system_tokens~={system_target_tokens}  max_tokens={max_tokens}")
    target_model, tokenizer, draft_model, resolved_draft = load_runtime_components(
        model_ref=target_ref,
        draft_ref=draft_ref,
    )
    print(f"Loaded. resolved_draft={resolved_draft}")

    prompts = _build_turn_prompts(tokenizer, n_turns, system_target_tokens)
    print(f"Prompt lengths: {[len(p) for p in prompts]}")

    # Warmup the runtime once (unmeasured) to pay MLX JIT cost.
    print("Warmup (8 tokens, no cache)...")
    _run_one_turn(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt_tokens=prompts[0][:64],
        max_tokens=8,
    )

    print("\n=== mlx_lm AR (pure baseline, fresh cache per turn) ===")
    mlxlm = _session_mlxlm(
        target_model=target_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_tokens=max_tokens,
    )

    print("\n=== DFlash (cache OFF — pre-feature behavior) ===")
    baseline = _session(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompts=prompts,
        max_tokens=max_tokens,
        use_cache=False,
        cache=None,
        key=None,
    )

    print("\n=== DFlash + PREFIX CACHE (new feature) ===")
    cache = DFlashPrefixCache(max_entries=16)
    key = _build_key(target_ref, resolved_draft or draft_ref, draft_model)
    cached = _session(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompts=prompts,
        max_tokens=max_tokens,
        use_cache=True,
        cache=cache,
        key=key,
    )

    print("\n=== SUMMARY: mlx_lm AR  vs  DFlash off  vs  DFlash+cache ===")
    header = (
        f"{'turn':>4s} | {'prompt':>6s} | "
        f"{'mlx_lm TTFT':>12s} {'mlx_lm tot':>12s} | "
        f"{'dflash TTFT':>12s} {'dflash tot':>12s} | "
        f"{'+cache TTFT':>12s} {'+cache tot':>12s} | "
        f"{'speedup vs mlx_lm (tot)':>24s}"
    )
    print(header)
    mlxlm_total_ttft = 0.0
    dflash_total_ttft = 0.0
    cached_total_ttft = 0.0
    mlxlm_total_wall = 0.0
    dflash_total_wall = 0.0
    cached_total_wall = 0.0
    mlxlm_total_tokens = 0
    dflash_total_tokens = 0
    cached_total_tokens = 0
    for m, b, c in zip(mlxlm, baseline, cached):
        mlxlm_total_ttft += m["ttft_us"]
        dflash_total_ttft += b["ttft_us"]
        cached_total_ttft += c["ttft_us"]
        mlxlm_total_wall += m["total_us"]
        dflash_total_wall += b["total_us"]
        cached_total_wall += c["total_us"]
        mlxlm_total_tokens += m["n_tokens"]
        dflash_total_tokens += b["n_tokens"]
        cached_total_tokens += c["n_tokens"]
        speedup = m["total_us"] / max(c["total_us"], 1.0)
        print(
            f"{m['turn']:>4d} | {m['prompt_len']:>6d} | "
            f"{m['ttft_us']/1000:>10.1f}ms {m['total_us']/1000:>10.1f}ms | "
            f"{b['ttft_us']/1000:>10.1f}ms {b['total_us']/1000:>10.1f}ms | "
            f"{c['ttft_us']/1000:>10.1f}ms {c['total_us']/1000:>10.1f}ms | "
            f"{speedup:>22.2f}x"
        )
    print()
    print(f"Session totals (sum across {len(cached)} turns):")
    print(f"  mlx_lm AR        : TTFT {mlxlm_total_ttft/1000:>9.1f} ms  wall {mlxlm_total_wall/1000:>9.1f} ms")
    print(f"  DFlash (no cache): TTFT {dflash_total_ttft/1000:>9.1f} ms  wall {dflash_total_wall/1000:>9.1f} ms")
    print(f"  DFlash + cache   : TTFT {cached_total_ttft/1000:>9.1f} ms  wall {cached_total_wall/1000:>9.1f} ms")
    print(f"  generated tokens : mlx_lm {mlxlm_total_tokens} | DFlash {dflash_total_tokens} | +cache {cached_total_tokens}")
    print()
    speedup_mlxlm = mlxlm_total_wall / max(cached_total_wall, 1.0)
    speedup_dflash = dflash_total_wall / max(cached_total_wall, 1.0)
    print(
        f"DFlash+cache vs mlx_lm AR (session wall): {speedup_mlxlm:.2f}×  "
        f"saved {(mlxlm_total_wall - cached_total_wall)/1000:.1f} ms"
    )
    print(
        f"DFlash+cache vs DFlash no-cache (session wall): {speedup_dflash:.2f}×  "
        f"saved {(dflash_total_wall - cached_total_wall)/1000:.1f} ms"
    )
    print("cache stats:", cache.stats())

    out_path = os.environ.get(
        "BENCH_OUT",
        f"benchmark/results/apple-m5-max/prefix_cache_multiturn_{time.strftime('%Y%m%dT%H%M%SZ')}.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "target": target_ref,
                "draft": draft_ref,
                "turns": n_turns,
                "system_target_tokens": system_target_tokens,
                "max_tokens": max_tokens,
                "mlxlm": mlxlm,
                "baseline": baseline,
                "cached": cached,
                "cache_stats": cache.stats(),
                "session_totals": {
                    "mlxlm_ttft_ms": mlxlm_total_ttft / 1000,
                    "mlxlm_wall_ms": mlxlm_total_wall / 1000,
                    "dflash_ttft_ms": dflash_total_ttft / 1000,
                    "dflash_wall_ms": dflash_total_wall / 1000,
                    "cached_ttft_ms": cached_total_ttft / 1000,
                    "cached_wall_ms": cached_total_wall / 1000,
                    "mlxlm_generated_tokens": mlxlm_total_tokens,
                    "dflash_generated_tokens": dflash_total_tokens,
                    "cached_generated_tokens": cached_total_tokens,
                    "speedup_vs_mlxlm": speedup_mlxlm,
                    "speedup_vs_dflash_no_cache": speedup_dflash,
                },
            },
            f,
            indent=2,
        )
    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

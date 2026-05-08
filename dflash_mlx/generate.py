# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any, Optional

from dflash_mlx.metal_limits import apply_metal_limits
from dflash_mlx.engine.target_ops import bind_draft_to_target, resolve_target_ops
from dflash_mlx.runtime import (
    VerifyConfig,
    get_stop_token_ids,
    load_draft_bundle,
    load_target_bundle,
    stream_dflash_generate,
)
from dflash_mlx.bet_optimal_drafting import BODConfig, BODController
from dflash_mlx.runtime_context import (
    build_offline_runtime_context,
)

DRAFT_REGISTRY = {
    "Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
    "Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
    "Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
    "Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
    "Qwen3.6-27B": "z-lab/Qwen3.6-27B-DFlash",
    "Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
    "Qwen3-4B": "z-lab/Qwen3-4B-DFlash-b16",
    "Qwen3-8B": "z-lab/Qwen3-8B-DFlash-b16",
    "gemma-4-31b-it": "z-lab/gemma-4-31B-it-DFlash",
    "gemma-4-26b-a4b-it": "z-lab/gemma-4-26B-A4B-it-DFlash",
}

_NORMALIZED_DRAFT_REGISTRY = {
    key.lower(): value for key, value in DRAFT_REGISTRY.items()
}

def _supported_base_models() -> str:
    return ", ".join(DRAFT_REGISTRY.keys())

def _strip_model_org(model_ref: str) -> str:
    return str(model_ref).rsplit("/", 1)[-1].strip()

def resolve_optional_draft_ref(model_ref: str, draft_ref: Optional[str]) -> Optional[str]:
    if draft_ref:
        return draft_ref

    stripped_name = _strip_model_org(model_ref)
    lowered_name = stripped_name.lower()

    exact = _NORMALIZED_DRAFT_REGISTRY.get(lowered_name)
    if exact is not None:
        return exact

    matching_bases = [
        base_name
        for base_name in _NORMALIZED_DRAFT_REGISTRY
        if lowered_name == base_name
        or lowered_name.startswith(base_name + "-")
        or lowered_name.startswith(base_name + "_")
    ]
    if not matching_bases:
        return None

    best_match = max(matching_bases, key=len)
    return _NORMALIZED_DRAFT_REGISTRY[best_match]

def decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        return str(tokenizer.decode(int(token_id)))

def generation_tps_from_summary(summary: dict[str, Any]) -> float:
    elapsed_us = float(summary.get("elapsed_us", 0.0))
    phase_timings = dict(summary.get("phase_timings_us", {}))
    prefill_us = float(summary.get("prefill_us", phase_timings.get("prefill", 0.0)))
    generation_tokens = int(summary.get("generation_tokens", 0))
    generation_us = max(0.0, elapsed_us - prefill_us)
    return (generation_tokens / (generation_us / 1e6)) if generation_us > 0.0 else 0.0

def load_runtime_components(
    *,
    model_ref: str,
    draft_ref: Optional[str],
    draft_quant: Optional[str] = None,
    verify_config: Optional[VerifyConfig] = None,
):
    resolved_draft_ref = resolve_optional_draft_ref(model_ref, draft_ref)
    if not resolved_draft_ref:
        raise ValueError(
            f"No DFlash draft model found for '{model_ref}'.\n"
            f"Use --draft to specify one, or check https://huggingface.co/z-lab for available drafts.\n"
            f"Supported base models: {_supported_base_models()}"
        )
    target_model, tokenizer, _ = load_target_bundle(
        model_ref,
        lazy=True,
        verify_config=verify_config,
    )
    try:
        draft_model, _ = load_draft_bundle(resolved_draft_ref, lazy=True, draft_quant=draft_quant)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Failed to load DFlash draft model '{resolved_draft_ref}' for '{model_ref}'."
        ) from exc
    target_ops = resolve_target_ops(target_model)
    bind_draft_to_target(draft_model, target_model, target_ops=target_ops)
    return target_model, tokenizer, draft_model, resolved_draft_ref

def run_generate(
    *,
    model_ref: str,
    prompt: str,
    max_tokens: int,
    use_chat_template: bool,
    draft_ref: Optional[str],
    target_fa_window: int = 0,
    prefill_step_size: int | None = None,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
    verify_len_cap: int = 0,
    verify_mode: str = "auto",
    draft_quant: Optional[str] = None,
    bod_enabled: bool = False,
    bod_mode: str = "chain",
    bod_min_bet: int = 2,
    bod_max_bet: int = 16,
    bod_default_scale_cost: float = 8.0,
    bod_default_fixed_cost: float = 47.0,
) -> int:
    runtime_context = build_offline_runtime_context(
        target_fa_window=target_fa_window,
        prefill_step_size=prefill_step_size,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
        verify_mode=verify_mode,
        bod_enabled=bod_enabled,
        bod_mode=bod_mode,
        bod_min_bet=bod_min_bet,
        bod_max_bet=bod_max_bet,
        bod_default_scale_cost=bod_default_scale_cost,
        bod_default_fixed_cost=bod_default_fixed_cost,
    )
    target_model, tokenizer, draft_model, _ = load_runtime_components(
        model_ref=model_ref,
        draft_ref=draft_ref,
        draft_quant=draft_quant,
        verify_config=runtime_context.verify,
    )
    stop_token_ids = get_stop_token_ids(tokenizer)
    stream = stream_dflash_generate(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=use_chat_template,
        stop_token_ids=stop_token_ids,
        runtime_context=runtime_context,
    )

    summary: Optional[dict[str, Any]] = None
    for event in stream:
        if event.get("event") == "token":
            sys.stdout.write(decode_token(tokenizer, int(event["token_id"])))
            sys.stdout.flush()
        elif event.get("event") == "summary":
            summary = event

    if summary is None:
        return 1

    tps = generation_tps_from_summary(summary)
    acceptance_pct = float(summary.get("acceptance_ratio", 0.0)) * 100.0
    token_count = int(summary.get("generation_tokens", 0))
    sys.stderr.write(
        f"\n{token_count} tokens | {tps:.1f} tok/s | {acceptance_pct:.1f}% acceptance\n"
    )
    sys.stderr.flush()
    return 0

def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    apply_metal_limits()
    parser = argparse.ArgumentParser(prog=prog, description="Generate text with DFlash on MLX.")
    parser.add_argument("--model", required=True, help="Target model reference.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--draft", default=None, help="Optional draft model override.")
    parser.add_argument(
        "--draft-quant",
        default=None,
        metavar="SPEC",
        help="Optional in-memory draft quantization, e.g. w4:gs64.",
    )
    parser.add_argument(
        "--verify-mode",
        choices=("auto", "off"),
        default="auto",
        help="Verify path mode. Use off only for debug/parity.",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="Prompt prefill chunk size. Default: profile balanced value, 4096.",
    )
    parser.add_argument(
        "--target-fa-window",
        type=int,
        default=0,
        help=(
            "Experimental target verifier full-attention KV window. "
            "0 keeps full KV cache; N>0 uses rotating KV cache for target FA layers only."
        ),
    )
    parser.add_argument(
        "--draft-sink-size",
        type=int,
        default=64,
        help="Draft context cache sink tokens kept before the rolling window.",
    )
    parser.add_argument(
        "--draft-window-size",
        type=int,
        default=1024,
        help="Draft context cache rolling window tokens.",
    )
    parser.add_argument(
        "--verify-len-cap",
        type=int,
        default=0,
        help="Max tokens verified per target forward; 0 uses the block size.",
    )
    # -- Bet-Optimal Drafting --
    parser.add_argument(
        "--bod-enabled",
        action="store_true",
        default=False,
        help="Enable Bet-Optimal Drafting for dynamic block size selection.",
    )
    parser.add_argument(
        "--bod-mode",
        choices=("chain", "tree"),
        default="chain",
        help="BOD betting mode.",
    )
    parser.add_argument(
        "--bod-min-bet",
        type=int,
        default=2,
        help="Minimum bet size for BOD.",
    )
    parser.add_argument(
        "--bod-max-bet",
        type=int,
        default=16,
        help="Maximum bet size for BOD.",
    )
    parser.add_argument(
        "--bod-default-scale-cost",
        type=float,
        default=8.0,
        help="Default per-token draft cost (chain) or per-node verify cost (tree).",
    )
    parser.add_argument(
        "--bod-default-fixed-cost",
        type=float,
        default=47.0,
        help="Default fixed verify cost (chain) or draft cost (tree).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.prefill_step_size is not None and args.prefill_step_size <= 0:
        raise SystemExit("--prefill-step-size must be > 0")
    if args.target_fa_window < 0:
        raise SystemExit("--target-fa-window must be >= 0")
    if args.draft_sink_size < 0:
        raise SystemExit("--draft-sink-size must be >= 0")
    if args.draft_window_size <= 0:
        raise SystemExit("--draft-window-size must be > 0")
    if args.verify_len_cap < 0:
        raise SystemExit("--verify-len-cap must be >= 0")
    raise SystemExit(
        run_generate(
            model_ref=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            use_chat_template=not args.no_chat_template,
            draft_ref=args.draft,
            target_fa_window=args.target_fa_window,
            prefill_step_size=args.prefill_step_size,
            draft_sink_size=args.draft_sink_size,
            draft_window_size=args.draft_window_size,
            verify_len_cap=args.verify_len_cap,
            verify_mode=args.verify_mode,
            draft_quant=args.draft_quant,
            bod_enabled=args.bod_enabled,
            bod_mode=args.bod_mode,
            bod_min_bet=args.bod_min_bet,
            bod_max_bet=args.bod_max_bet,
            bod_default_scale_cost=args.bod_default_scale_cost,
            bod_default_fixed_cost=args.bod_default_fixed_cost,
        )
    )

if __name__ == "__main__":
    main()

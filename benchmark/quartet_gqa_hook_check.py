# Copyright 2026 bstnxbt
# MIT License — see LICENSE file

from __future__ import annotations

import argparse

import mlx.core as mx

from dflash_mlx.runtime import (
    _target_text_model,
    _verify_target_block,
    configure_quartet_gqa,
    load_draft_bundle,
    load_target_bundle,
    make_target_cache,
)

README_MATH_PROMPT = (
    "The function $f$ satisfies the functional equation "
    "\\[ f(x) + f(y) = f(x + y) - xy - 1 \\] for all real numbers $x$ and $y$. "
    "If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. "
    "Enter all such integers, separated by commas. Please reason step by step, "
    "and put your final answer within \\boxed{}."
)


def _build_long_token_stream(tokenizer, *, seed_text: str, total_tokens: int) -> list[int]:
    chunk = list(tokenizer.encode(seed_text + "\n\n"))
    if not chunk:
        raise ValueError("Tokenizer produced no tokens for hook-check seed prompt")
    tokens: list[int] = []
    while len(tokens) < total_tokens:
        tokens.extend(chunk)
    return tokens[:total_tokens]


def _prefill_cache(
    *,
    target_model,
    prefix_tokens: list[int],
    cache: list[object],
    chunk_size: int,
) -> None:
    prefix = mx.array(prefix_tokens, dtype=mx.uint32)[None]
    for start in range(0, int(prefix.shape[1]), chunk_size):
        chunk = prefix[:, start : start + chunk_size]
        logits = target_model(chunk, cache=cache)
        mx.eval(logits)


def _set_only_quartet_layer(target_model, enabled_layer: int | None) -> list[int]:
    fa_layers = configure_quartet_gqa(target_model, enabled=False)
    if enabled_layer is None:
        return fa_layers
    inner = _target_text_model(target_model)
    inner.layers[enabled_layer].self_attn._dflash_quartet_enabled = True
    return fa_layers


def _run_verify_logits(
    *,
    target_model,
    prefix_tokens: list[int],
    verify_tokens: list[int],
    capture_layer_ids: set[int],
    enabled_layer: int | None,
    prefill_chunk_size: int,
) -> mx.array:
    _set_only_quartet_layer(target_model, enabled_layer)
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
    )
    _prefill_cache(
        target_model=target_model,
        prefix_tokens=prefix_tokens,
        cache=cache,
        chunk_size=prefill_chunk_size,
    )
    verify_ids = mx.array(verify_tokens, dtype=mx.uint32)[None]
    logits, captured = _verify_target_block(
        target_model=target_model,
        verify_ids=verify_ids,
        target_cache=cache,
        verify_chunk_tokens=None,
        capture_layer_ids=capture_layer_ids,
    )
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-by-layer quartet-GQA hook numerical check.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--draft", default="z-lab/Qwen3.5-9B-DFlash")
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--verify-tokens", type=int, default=16)
    parser.add_argument("--prefill-chunk-size", type=int, default=2048)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    target_model, tokenizer, _ = load_target_bundle(
        args.model,
        lazy=True,
        split_full_attention_sdpa=True,
    )
    draft_model, _ = load_draft_bundle(args.draft, lazy=True)

    total = int(args.context_tokens) + int(args.verify_tokens)
    token_stream = _build_long_token_stream(
        tokenizer,
        seed_text=README_MATH_PROMPT,
        total_tokens=total,
    )
    prefix_tokens = token_stream[: int(args.context_tokens)]
    verify_tokens = token_stream[int(args.context_tokens) : total]
    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}

    fa_layers = _set_only_quartet_layer(target_model, enabled_layer=None)
    ref_logits = _run_verify_logits(
        target_model=target_model,
        prefix_tokens=prefix_tokens,
        verify_tokens=verify_tokens,
        capture_layer_ids=capture_layer_ids,
        enabled_layer=None,
        prefill_chunk_size=int(args.prefill_chunk_size),
    )

    failures: list[dict[str, float | int | bool]] = []
    for layer_index in fa_layers:
        test_logits = _run_verify_logits(
            target_model=target_model,
            prefix_tokens=prefix_tokens,
            verify_tokens=verify_tokens,
            capture_layer_ids=capture_layer_ids,
            enabled_layer=layer_index,
            prefill_chunk_size=int(args.prefill_chunk_size),
        )
        diff = mx.abs(ref_logits.astype(mx.float32) - test_logits.astype(mx.float32))
        max_abs_diff = float(mx.max(diff).item())
        allclose = bool(mx.allclose(ref_logits, test_logits, atol=args.atol, rtol=args.rtol).item())
        result = {
            "layer_index": layer_index,
            "max_abs_diff": max_abs_diff,
            "allclose": allclose,
        }
        print(result)
        if not allclose:
            failures.append(result)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

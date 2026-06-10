# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any

import mlx.core as mx


def prepare_prompt_tokens(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool,
) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))


def build_suppress_token_mask(
    vocab_size: int,
    suppress_token_ids: list[int] | None,
) -> mx.array | None:
    token_ids = sorted(
        {
            int(token_id)
            for token_id in (suppress_token_ids or [])
            if 0 <= int(token_id) < vocab_size
        }
    )
    if not token_ids:
        return None
    vocab_indices = mx.arange(vocab_size, dtype=mx.int32)
    token_array = mx.array(token_ids, dtype=mx.int32)
    return mx.any(mx.equal(vocab_indices[:, None], token_array[None, :]), axis=1)


def greedy_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: mx.array | None = None,
) -> mx.array:
    if suppress_token_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    floor = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(suppress_token_mask, floor, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)


def masked_topk_arrays(
    logits_2d: mx.array,
    suppress_token_mask: mx.array | None,
    *,
    width: int,
) -> tuple[mx.array, mx.array]:
    """Per-row top-`width` ids (desc) and masked log-softmax values, as lazy arrays.

    Mirrors greedy_tokens_with_mask masking, so row argmax is always id 0 (up to
    bf16 ties). No eval here: callers fold both arrays into an existing eval point.
    """
    top_width = int(width)
    if top_width <= 0:
        raise ValueError("width must be positive")
    masked = logits_2d
    if suppress_token_mask is not None:
        floor = mx.array(-1e9, dtype=logits_2d.dtype)
        masked = mx.where(suppress_token_mask, floor, logits_2d)
    top = mx.argpartition(masked, kth=-top_width, axis=-1)[:, -top_width:]
    top_logits = mx.take_along_axis(masked, top, axis=-1)
    order = mx.argsort(top_logits, axis=-1)[:, ::-1]
    top = mx.take_along_axis(top, order, axis=-1)
    log_probs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
    values = mx.take_along_axis(log_probs, top, axis=-1)
    return top, values


def eval_logits_and_captured(
    logits: mx.array,
    captured: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)


def ns_to_us(ns: int | float) -> float:
    return float(ns) / 1_000.0

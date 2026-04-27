# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any, Optional

import mlx.core as mx


def stream_baseline_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    fallback_reason: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    from dflash_mlx.runtime import (
        _prepare_prompt_tokens,
        build_suppress_token_mask,
        greedy_tokens_with_mask,
        make_target_cache,
    )
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    prompt_len = len(prompt_tokens)
    stop_token_ids = list(stop_token_ids or [])
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=quantize_kv_cache,
    )
    start_ns = time.perf_counter_ns()
    _yield_pause_ns = 0

    prefill_start_ns = time.perf_counter_ns()
    logits = target_model(prompt_array, cache=cache)
    mx.eval(logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns
    suppress_token_mask = build_suppress_token_mask(int(logits.shape[-1]), suppress_token_ids)
    next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
    generated_tokens = [next_token]

    _pre_yield = time.perf_counter_ns()
    yield {
        "event": "prefill",
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": prompt_len,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }
    _yield_pause_ns += time.perf_counter_ns() - _pre_yield

    _pre_yield = time.perf_counter_ns()
    yield {
        "event": "token",
        "token_id": next_token,
        "generated_tokens": 1,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }
    _yield_pause_ns += time.perf_counter_ns() - _pre_yield

    while len(generated_tokens) < max_new_tokens:
        if next_token in stop_token_ids:
            break
        token_array = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(token_array, cache=cache)
        next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
        generated_tokens.append(next_token)
        _pre_yield = time.perf_counter_ns()
        yield {
            "event": "token",
            "token_id": next_token,
            "generated_tokens": len(generated_tokens),
            "acceptance_ratio": 0.0,
            "cycles_completed": 0,
            "fallback_ar": True,
            "fallback_reason": fallback_reason,
        }
        _yield_pause_ns += time.perf_counter_ns() - _pre_yield

    elapsed_us = (time.perf_counter_ns() - start_ns - _yield_pause_ns) / 1_000.0
    del cache
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    yield {
        "event": "summary",
        "elapsed_us": elapsed_us,
        "prompt_token_count": prompt_len,
        "generated_token_ids": generated_tokens,
        "generation_tokens": len(generated_tokens),
        "accepted_from_draft": 0,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "phase_timings_us": {
            "prefill": prefill_ns / 1_000.0,
            "draft": 0.0,
            "draft_prefill": 0.0,
            "draft_incremental": 0.0,
            "verify": 0.0,
            "replay": 0.0,
            "commit": 0.0,
        },
        "verify_len_cap": None,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }

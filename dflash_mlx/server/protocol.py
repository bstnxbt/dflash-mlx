# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

import mlx_lm.server as mlx_server

STATEFUL_SERVER_API = "state" in getattr(mlx_server.Response, "__annotations__", {})

def build_generation_context(tokenizer, prompt, stop_words=None, sequences=None):
    if STATEFUL_SERVER_API:
        return mlx_server.GenerationContext(
            has_thinking=tokenizer.has_thinking,
            has_tool_calling=tokenizer.has_tool_calling,
            tool_parser=tokenizer.tool_parser,
            sequences=sequences or {},
            prompt=prompt,
        )
    return mlx_server.GenerationContext(
        has_tool_calling=tokenizer.has_tool_calling,
        tool_call_start=tokenizer.tool_call_start,
        tool_call_end=tokenizer.tool_call_end,
        tool_parser=tokenizer.tool_parser,
        has_thinking=tokenizer.has_thinking,
        think_start_id=tokenizer.think_start_id,
        think_end=tokenizer.think_end,
        think_end_id=tokenizer.think_end_id,
        eos_token_ids=tokenizer.eos_token_ids,
        stop_token_sequences=[
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in (stop_words or [])
        ],
        prompt=prompt,
    )

def make_response(
    *,
    text: str,
    token: int,
    state: Optional[str],
    match: Optional[tuple[int, ...]],
    finish_reason: Optional[str],
):
    if STATEFUL_SERVER_API:
        return mlx_server.Response(
            text,
            token,
            state or "normal",
            match,
            0.0,
            finish_reason,
            (),
        )
    return mlx_server.Response(
        text,
        token,
        0.0,
        finish_reason,
        (),
    )

def match_stream_token(
    sm: Optional[Any],
    sm_state: Optional[Any],
    token: int,
) -> tuple[Optional[Any], Optional[tuple[int, ...]], Optional[str], bool]:
    if sm is None:
        return sm_state, None, "normal", False
    if sm_state is None or sm_state[0] is None:
        return sm_state, None, None, True
    next_state, match_sequence, current_state = sm.match(sm_state, token)
    terminal = match_sequence is not None and current_state is None
    return next_state, match_sequence, current_state, terminal

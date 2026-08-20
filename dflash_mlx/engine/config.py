# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from dflash_mlx.diagnostics import DiagnosticsConfig


@dataclass(frozen=True)
class SpeculativeCycleConfig:
    draft_block_size: int
    requested_block_tokens: int
    effective_block_tokens: int
    verify_len_cap: int


def resolve_verify_len_cap(runtime_config: Any, block_tokens: int) -> int:
    requested = int(getattr(runtime_config, "verify_len_cap", 0) or 0)
    if requested <= 0:
        return int(block_tokens)
    return max(1, min(int(block_tokens), requested))

def verify_token_count_for_block(block_len: int, verify_len_cap: int) -> int:
    return max(1, min(int(block_len), int(verify_len_cap)))


def draft_capability(draft_model: Any, name: str, default: Any) -> Any:
    capabilities = getattr(draft_model, "capabilities", None)
    if capabilities is None:
        return default
    return getattr(capabilities, name, default)


def resolve_speculative_cycle_config(
    runtime_config: Any,
    draft_model: Any,
    block_tokens: Optional[int],
) -> SpeculativeCycleConfig:
    draft_block_size = int(draft_model.block_size)
    default_block_tokens = int(
        draft_capability(draft_model, "default_block_tokens", draft_block_size)
        or draft_block_size
    )
    max_block_tokens = int(
        draft_capability(draft_model, "max_block_tokens", draft_block_size)
        or draft_block_size
    )
    requested_block_tokens = (
        default_block_tokens if block_tokens is None else int(block_tokens)
    )
    effective_block_tokens = max(
        1,
        min(requested_block_tokens, draft_block_size, max_block_tokens),
    )
    return SpeculativeCycleConfig(
        draft_block_size=draft_block_size,
        requested_block_tokens=requested_block_tokens,
        effective_block_tokens=effective_block_tokens,
        verify_len_cap=resolve_verify_len_cap(runtime_config, effective_block_tokens),
    )


def resolve_draft_window(
    runtime_config: Any,
    draft_model: Any,
    *,
    context_len: Optional[int] = None,
    allow_full_attention_context: bool = False,
) -> tuple[int, int]:
    sink = int(getattr(runtime_config, "draft_sink_size", 64))
    requested_window = int(getattr(runtime_config, "draft_window_size", 1024))
    return sink, _effective_draft_window_size(
        draft_model,
        requested_window,
        context_len=context_len,
        allow_full_attention_context=allow_full_attention_context,
    )

def _is_unwindowed_full_attention_draft(draft_model: Any) -> bool:
    args = getattr(draft_model, "args", None)
    if args is None:
        return False
    if int(getattr(args, "sliding_window", 0) or 0) > 0:
        return False
    layer_types = tuple(str(kind) for kind in (getattr(args, "layer_types", ()) or ()))
    if not layer_types:
        return False
    return all(kind == "full_attention" for kind in layer_types)

def _effective_draft_window_size(
    draft_model: Any,
    requested_window: int,
    *,
    context_len: Optional[int] = None,
    allow_full_attention_context: bool = False,
) -> int:
    sliding_window = int(getattr(getattr(draft_model, "args", None), "sliding_window", 0) or 0)
    window = max(1, int(requested_window), sliding_window)
    if (
        allow_full_attention_context
        and context_len is not None
        and _is_unwindowed_full_attention_draft(draft_model)
    ):
        window = max(window, int(context_len))
    return window

def _profile_dflash_cycles_enabled(
    diagnostics: Optional[DiagnosticsConfig] = None,
) -> bool:
    return bool(diagnostics is not None and diagnostics.trace.cycle_events)

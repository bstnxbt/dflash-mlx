from __future__ import annotations

import os
import sys
from typing import Any, Optional


def _resolve_verify_len_cap(target_model: Any, block_tokens: int) -> int:
    override_raw = os.environ.get("DFLASH_VERIFY_LEN", "").strip()
    if override_raw:
        try:
            override = int(override_raw)
        except ValueError:
            override = 0
        if override > 0:
            return max(1, min(int(block_tokens), override))
    return int(block_tokens)


def _resolve_dflash_max_ctx() -> int:
    raw = os.environ.get("DFLASH_MAX_CTX", "0").strip()
    try:
        max_ctx = int(raw)
    except ValueError:
        max_ctx = 0
    if max_ctx <= 0:
        return sys.maxsize
    return max_ctx


def _resolve_draft_window() -> tuple[int, int]:
    sink = int(os.environ.get("DFLASH_DRAFT_SINK", "64").strip())
    window = int(os.environ.get("DFLASH_DRAFT_WINDOW", "1024").strip())
    return max(0, sink), max(1, window)


def _draft_window_override_enabled() -> bool:
    return bool(os.environ.get("DFLASH_DRAFT_WINDOW", "").strip())


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
    allow_full_attention_context: bool = True,
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


def _profile_dflash_cycles_enabled() -> bool:
    raw = os.environ.get("DFLASH_PROFILE", "").strip().lower()
    if raw not in {"", "0", "false", "no"}:
        return True
    return bool(os.environ.get("DFLASH_BENCH_LOG_DIR", "").strip())

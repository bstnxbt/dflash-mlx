# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from typing import Any, Optional

from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.policies import read_budget_env
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.engine.config import _resolve_draft_window


def make_prefix_cache() -> DFlashPrefixCache:
    max_entries, max_bytes = read_budget_env()
    cache = DFlashPrefixCache(max_entries=max_entries, max_bytes=max_bytes)
    sys.stderr.write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix cache enabled "
        f"(max_entries={max_entries}, max_bytes={max_bytes})\n"
    )
    sys.stderr.flush()
    return cache


def format_stats_line(cache: DFlashPrefixCache, label: str = "") -> None:
    stats = cache.stats()
    prefix = f" [{label}]" if label else ""
    sys.stderr.write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix-cache-stats{prefix} "
        f"entries={stats['current_entries']}/{stats['max_entries']} "
        f"bytes={stats['current_bytes']}/{stats['max_bytes']} "
        f"hits={stats['exact_hits']}+{stats['prefix_hits']} "
        f"misses={stats['misses']} "
        f"insertions={stats['insertions']} "
        f"evictions={stats['evictions']} "
        f"prefill_tokens_saved={stats['prefill_tokens_saved']}\n"
    )
    sys.stderr.flush()


def build_prefix_key(model_provider: Any, draft_model: Any) -> DFlashPrefixKey:
    model_key = getattr(model_provider, "model_key", None) or ("", None, "")
    target_id = str(model_key[0]) if len(model_key) > 0 else ""
    draft_id = (
        str(model_key[2]) if len(model_key) > 2 and model_key[2] is not None else ""
    )
    capture_ids = tuple(
        int(x) for x in getattr(draft_model, "target_layer_ids", ()) or ()
    )
    sink, window = _resolve_draft_window()
    return DFlashPrefixKey(
        target_model_id=target_id,
        draft_model_id=draft_id,
        capture_layer_ids=capture_ids,
        draft_sink_size=int(sink),
        draft_window_size=int(window),
    )


def chat_template_marker_ids(
    tokenizer: Any,
) -> tuple[Optional[int], Optional[int]]:
    im_start = None
    assistant = None
    try:
        ids = tokenizer.convert_tokens_to_ids(["<|im_start|>", "assistant"])
        if ids and ids[0] is not None and ids[0] != tokenizer.unk_token_id:
            im_start = int(ids[0])
        if ids and len(ids) > 1 and ids[1] is not None and ids[1] != tokenizer.unk_token_id:
            assistant = int(ids[1])
    except Exception:
        pass
    return im_start, assistant

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from typing import Any, Optional

from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.prefix_l2 import DFlashPrefixL2Cache

def make_prefix_cache(runtime_context: Any) -> DFlashPrefixCache:
    runtime_config = runtime_context.runtime
    max_entries = int(runtime_config.prefix_cache_max_entries)
    max_bytes = int(runtime_config.prefix_cache_max_bytes)
    l2: Optional[DFlashPrefixL2Cache] = None
    if runtime_config.prefix_cache_l2:
        try:
            l2 = DFlashPrefixL2Cache(
                cache_dir=runtime_config.prefix_cache_l2_dir,
                max_bytes=runtime_config.prefix_cache_l2_max_bytes,
            )
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix L2 cache enabled "
                f"(dir={l2.cache_dir}, max_bytes={runtime_config.prefix_cache_l2_max_bytes}, "
                f"writable={l2.writable})\n"
            )
        except Exception as e:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix L2 cache disabled: {e}\n"
            )
            l2 = None
    trace_config = (
        runtime_context.diagnostics.trace
        if runtime_context is not None
        else None
    )
    cache = DFlashPrefixCache(
        max_entries=max_entries,
        max_bytes=max_bytes,
        l2=l2,
        trace_config=trace_config,
        max_snapshot_tokens=runtime_config.max_snapshot_tokens,
    )
    sys.stderr.write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix cache enabled "
        f"(max_entries={max_entries}, max_bytes={max_bytes})\n"
    )
    sys.stderr.flush()
    return cache

def format_stats_line(cache: DFlashPrefixCache, label: str = "") -> None:
    stats = cache.stats()
    prefix = f" [{label}]" if label else ""
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix-cache-stats{prefix} "
        f"entries={stats['current_entries']}/{stats['max_entries']} "
        f"bytes={stats['current_bytes']}/{stats['max_bytes']} "
        f"hits={stats['exact_hits']}+{stats['prefix_hits']} "
        f"misses={stats['misses']} "
        f"insertions={stats['insertions']} "
        f"evictions={stats['evictions']} "
        f"prefill_tokens_saved={stats['prefill_tokens_saved']}"
    )
    l2 = stats.get("l2")
    if l2:
        line += (
            f" l2_hits={stats.get('l2_hits', 0)} l2_misses={stats.get('l2_misses', 0)} "
            f"l2_writes={l2.get('writes', 0)} l2_bytes={l2.get('current_bytes', 0)}/{l2.get('max_bytes', 0)}"
        )
    sys.stderr.write(line + "\n")
    sys.stderr.flush()

def build_prefix_key(
    model_provider: Any,
    draft_model: Any,
    runtime_context: Optional[Any] = None,
) -> DFlashPrefixKey:
    model_key = getattr(model_provider, "model_key", None) or ("", None, "")
    target_id = str(model_key[0]) if len(model_key) > 0 else ""
    draft_id = (
        str(model_key[2]) if len(model_key) > 2 and model_key[2] is not None else ""
    )
    capture_ids = tuple(
        int(x) for x in getattr(draft_model, "target_layer_ids", ()) or ()
    )
    if runtime_context is not None:
        runtime_config = runtime_context.runtime
        sink = int(getattr(runtime_config, "draft_sink_size", 64))
        window = int(getattr(runtime_config, "draft_window_size", 1024))
    else:
        sink = 64
        window = 1024
    target_fa_window = (
        runtime_context.runtime.target_fa_window
        if runtime_context is not None
        else 0
    )
    return DFlashPrefixKey(
        target_model_id=target_id,
        draft_model_id=draft_id,
        capture_layer_ids=capture_ids,
        draft_sink_size=int(sink),
        draft_window_size=int(window),
        target_fa_window=int(target_fa_window),
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

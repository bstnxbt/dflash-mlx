# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.engine.config import _effective_draft_window_size
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

def _clone_array(a: Optional[mx.array]) -> Optional[mx.array]:
    if a is None:
        return None
    cloned = mx.array(a)
    mx.eval(cloned)
    return cloned

def _resolve_effective_trim_window(
    draft_model: Optional[Any],
    total_len: int,
    *,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
) -> tuple[int, int]:
    sink = int(draft_sink_size)
    requested = int(draft_window_size)
    if draft_model is None:
        return max(0, int(sink)), max(0, int(requested))
    effective = _effective_draft_window_size(
        draft_model,
        requested,
        context_len=int(total_len),
        allow_full_attention_context=True,
    )
    return max(0, int(sink)), max(0, int(effective))

def _build_target_hidden_chunks(
    target_hidden: mx.array,
    *,
    draft_model: Optional[Any] = None,
    trim_target_hidden: bool = True,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
) -> tuple[tuple[mx.array, ...], tuple[tuple[int, int], ...], int]:
    total_len = int(target_hidden.shape[1])
    if not trim_target_hidden:
        full = _clone_array(target_hidden)
        assert full is not None
        return (full,), ((0, total_len),), total_len
    sink, window = _resolve_effective_trim_window(
        draft_model,
        total_len,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
    )
    if total_len <= sink + window or sink + window == 0:
        full = _clone_array(target_hidden)
        assert full is not None
        return (full,), ((0, total_len),), total_len
    sink_chunk = _clone_array(target_hidden[:, :sink, :])
    tail_chunk = _clone_array(target_hidden[:, total_len - window :, :])
    assert sink_chunk is not None and tail_chunk is not None
    return (
        (sink_chunk, tail_chunk),
        ((0, sink), (total_len - window, total_len)),
        total_len,
    )

def target_cache_is_serializable(target_cache: list[Any]) -> bool:
    for entry in target_cache:
        if isinstance(entry, RecurrentRollbackCache):
            continue
        if isinstance(entry, RotatingKVCache):
            return False
        if isinstance(entry, KVCache):
            continue
        return False
    return True

def serialize_target_cache(
    target_cache: list[Any],
) -> tuple[
    tuple[Optional[tuple[mx.array, mx.array, int]], ...],
    tuple[Optional[tuple[Optional[mx.array], ...]], ...],
]:
    fa: list[Optional[tuple[mx.array, mx.array, int]]] = []
    gdn: list[Optional[tuple[Optional[mx.array], ...]]] = []
    for layer_idx, entry in enumerate(target_cache):
        if isinstance(entry, RecurrentRollbackCache):
            fa.append(None)
            gdn.append(tuple(_clone_array(a) for a in entry.cache))
        elif isinstance(entry, KVCache):
            state = entry.state
            if state is None or state[0] is None:
                fa.append(None)
                gdn.append(None)
            else:
                k, v = state
                fa.append((_clone_array(k), _clone_array(v), int(entry.offset)))
                gdn.append(None)
        else:
            raise TypeError(
                f"Cache entry type {type(entry).__name__} at layer {layer_idx} "
                "is not supported for prefix-cache serialization."
            )
    return tuple(fa), tuple(gdn)

def hydrate_target_cache(
    snapshot: DFlashPrefixSnapshot,
    template_cache: list[Any],
) -> list[Any]:
    if len(template_cache) != len(snapshot.fa_states):
        raise ValueError(
            f"Template cache length {len(template_cache)} != "
            f"snapshot layer count {len(snapshot.fa_states)}"
        )

    result: list[Any] = []
    for i, tmpl in enumerate(template_cache):
        fa_state = snapshot.fa_states[i]
        gdn_state = snapshot.gdn_states[i]

        if isinstance(tmpl, RecurrentRollbackCache):
            if gdn_state is None:
                raise ValueError(f"Snapshot missing GDN state at layer {i}")
            new_cache = RecurrentRollbackCache(
                size=len(tmpl.cache),
                conv_kernel_size=tmpl.conv_kernel_size,
            )
            new_cache.cache = [_clone_array(a) for a in gdn_state]
            result.append(new_cache)
        elif isinstance(tmpl, KVCache):
            if fa_state is None:
                raise ValueError(f"Snapshot missing FA state at layer {i}")
            k, v, offset = fa_state
            new_cache = KVCache()
            new_cache.keys = _clone_array(k)
            new_cache.values = _clone_array(v)
            new_cache.offset = offset
            result.append(new_cache)
        else:
            raise TypeError(
                f"Cannot hydrate cache of type {type(tmpl).__name__} at layer {i}"
            )
    return result

def build_snapshot(
    *,
    token_ids: list[int],
    target_cache: list[Any],
    target_hidden: mx.array,
    last_logits: Optional[mx.array],
    key: DFlashPrefixKey,
    kind: str = "prefill",
    draft_model: Optional[Any] = None,
    trim_target_hidden: bool = True,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
) -> DFlashPrefixSnapshot:
    fa, gdn = serialize_target_cache(target_cache)
    chunks, spans, total_len = _build_target_hidden_chunks(
        target_hidden,
        draft_model=draft_model,
        trim_target_hidden=trim_target_hidden,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
    )
    cloned_logits = _clone_array(last_logits) if last_logits is not None else None
    return DFlashPrefixSnapshot(
        token_ids=tuple(int(t) for t in token_ids),
        fa_states=fa,
        gdn_states=gdn,
        target_hidden_chunks=chunks,
        target_hidden_chunk_spans=spans,
        target_hidden_total_len=total_len,
        last_logits=cloned_logits,
        key=key,
        kind=kind,
    )

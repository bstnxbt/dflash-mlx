# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import (
    DFlashPrefixSnapshot,
    FAState,
    TargetHiddenChunks,
)
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
    allow_full_attention_context: bool = False,
) -> tuple[int, int]:
    sink = int(draft_sink_size)
    requested = int(draft_window_size)
    if draft_model is None:
        return max(0, int(sink)), max(0, int(requested))
    effective = _effective_draft_window_size(
        draft_model,
        requested,
        context_len=int(total_len),
        allow_full_attention_context=allow_full_attention_context,
    )
    if _requires_full_target_hidden(
        draft_model,
        allow_full_attention_context=allow_full_attention_context,
    ):
        effective = max(effective, int(total_len))
    return max(0, int(sink)), max(0, int(effective))

def _requires_full_target_hidden(
    draft_model: Any,
    *,
    allow_full_attention_context: bool,
) -> bool:
    if not allow_full_attention_context:
        return False
    args = getattr(draft_model, "args", None)
    layer_types = tuple(str(kind) for kind in (getattr(args, "layer_types", ()) or ()))
    return any(kind == "full_attention" for kind in layer_types)

def _target_hidden_slice(
    target_hidden: mx.array | TargetHiddenChunks,
    start: int,
    end: int,
) -> mx.array:
    if isinstance(target_hidden, TargetHiddenChunks):
        return target_hidden.slice(start, end)
    return target_hidden[:, start:end, :]

def _build_target_hidden_chunks(
    target_hidden: mx.array | TargetHiddenChunks,
    *,
    draft_model: Optional[Any] = None,
    trim_target_hidden: bool = True,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
    allow_full_attention_context: bool = False,
) -> tuple[tuple[mx.array, ...], tuple[tuple[int, int], ...], int]:
    total_len = int(target_hidden.shape[1])
    if not trim_target_hidden:
        full = _clone_array(_target_hidden_slice(target_hidden, 0, total_len))
        assert full is not None
        return (full,), ((0, total_len),), total_len
    sink, window = _resolve_effective_trim_window(
        draft_model,
        total_len,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        allow_full_attention_context=allow_full_attention_context,
    )
    if total_len <= sink + window or sink + window == 0:
        full = _clone_array(_target_hidden_slice(target_hidden, 0, total_len))
        assert full is not None
        return (full,), ((0, total_len),), total_len
    sink_chunk = _clone_array(_target_hidden_slice(target_hidden, 0, sink))
    tail_chunk = _clone_array(
        _target_hidden_slice(target_hidden, total_len - window, total_len)
    )
    assert sink_chunk is not None and tail_chunk is not None
    return (
        (sink_chunk, tail_chunk),
        ((0, sink), (total_len - window, total_len)),
        total_len,
    )

def serialize_target_cache(
    target_cache: list[Any],
) -> tuple[
    tuple[Optional[FAState], ...],
    tuple[Optional[tuple[Optional[mx.array], ...]], ...],
]:
    fa: list[Optional[FAState]] = []
    gdn: list[Optional[tuple[Optional[mx.array], ...]]] = []
    for layer_idx, entry in enumerate(target_cache):
        if isinstance(entry, RecurrentRollbackCache):
            fa.append(None)
            gdn.append(tuple(_clone_array(a) for a in entry.cache))
        elif isinstance(entry, RotatingKVCache):
            keys = getattr(entry, "keys", None)
            values = getattr(entry, "values", None)
            if keys is None or values is None:
                fa.append(None)
                gdn.append(None)
            else:
                fa.append(
                    (
                        _clone_array(keys),
                        _clone_array(values),
                        int(entry.offset),
                        int(entry._idx),
                    )
                )
                gdn.append(None)
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

def _validate_snapshot_cache_prefix_len(
    fa_states: tuple[Optional[FAState], ...],
    *,
    prefix_len: int,
) -> None:
    for layer_idx, state in enumerate(fa_states):
        if state is None:
            continue
        offset = int(state[2])
        if offset != int(prefix_len):
            raise ValueError(
                f"Snapshot FA cache offset {offset} at layer {layer_idx} "
                f"!= token prefix length {int(prefix_len)}"
            )

def hydrate_target_cache(
    snapshot: DFlashPrefixSnapshot,
    template_cache: list[Any],
) -> list[Any]:
    if len(template_cache) != len(snapshot.fa_states):
        raise ValueError(
            f"Template cache length {len(template_cache)} != "
            f"snapshot layer count {len(snapshot.fa_states)}"
        )
    _validate_snapshot_cache_prefix_len(
        snapshot.fa_states,
        prefix_len=snapshot.prefix_len,
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
            k, v, offset = fa_state[:3]
            new_cache = KVCache()
            new_cache.keys = _clone_array(k)
            new_cache.values = _clone_array(v)
            new_cache.offset = offset
            result.append(new_cache)
        elif isinstance(tmpl, RotatingKVCache):
            if fa_state is None:
                raise ValueError(f"Snapshot missing rotating FA state at layer {i}")
            if len(fa_state) != 4:
                raise ValueError(
                    f"Snapshot missing rotating FA ring index at layer {i}"
                )
            k, v, offset = fa_state[:3]
            new_cache = RotatingKVCache(
                max_size=int(tmpl.max_size),
                keep=int(tmpl.keep),
            )
            new_cache.keys = _clone_array(k)
            new_cache.values = _clone_array(v)
            new_cache.offset = int(offset)
            new_cache._idx = int(fa_state[3])
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
    target_hidden: mx.array | TargetHiddenChunks,
    last_logits: Optional[mx.array],
    key: DFlashPrefixKey,
    kind: str = "prefill",
    draft_model: Optional[Any] = None,
    trim_target_hidden: bool = True,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
    allow_full_attention_context: bool = False,
) -> DFlashPrefixSnapshot:
    token_tuple = tuple(int(t) for t in token_ids)
    prefix_len = len(token_tuple)
    hidden_len = int(target_hidden.shape[1])
    if hidden_len < prefix_len:
        raise ValueError(
            f"Snapshot target_hidden length {hidden_len} "
            f"< token prefix length {prefix_len}"
        )
    if hidden_len > prefix_len:
        if isinstance(target_hidden, TargetHiddenChunks):
            target_hidden = TargetHiddenChunks(
                total_len=prefix_len,
                chunks=tuple(
                    chunk[:, : max(0, min(end, prefix_len) - start), :]
                    for chunk, (start, end) in zip(
                        target_hidden.chunks,
                        target_hidden.spans,
                    )
                    if start < prefix_len
                ),
                spans=tuple(
                    (start, min(end, prefix_len))
                    for start, end in target_hidden.spans
                    if start < prefix_len
                ),
            )
        else:
            target_hidden = target_hidden[:, :prefix_len, :]

    fa, gdn = serialize_target_cache(target_cache)
    _validate_snapshot_cache_prefix_len(fa, prefix_len=prefix_len)
    chunks, spans, total_len = _build_target_hidden_chunks(
        target_hidden,
        draft_model=draft_model,
        trim_target_hidden=trim_target_hidden,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        allow_full_attention_context=allow_full_attention_context,
    )
    cloned_logits = _clone_array(last_logits) if last_logits is not None else None
    return DFlashPrefixSnapshot(
        token_ids=token_tuple,
        fa_states=fa,
        gdn_states=gdn,
        target_hidden_chunks=chunks,
        target_hidden_chunk_spans=spans,
        target_hidden_total_len=total_len,
        last_logits=cloned_logits,
        key=key,
        kind=kind,
    )


@dataclass(frozen=True)
class PrefixSnapshotBuilder:
    key: DFlashPrefixKey
    draft_model: Optional[Any] = None
    draft_sink_size: int = 64
    draft_window_size: int = 1024

    def build(
        self,
        *,
        token_ids: list[int],
        target_cache: list[Any],
        target_hidden: mx.array | TargetHiddenChunks,
        last_logits: Optional[mx.array],
        kind: str,
        allow_full_attention_context: bool = False,
    ) -> DFlashPrefixSnapshot:
        return build_snapshot(
            token_ids=token_ids,
            target_cache=target_cache,
            target_hidden=target_hidden,
            last_logits=last_logits,
            key=self.key,
            kind=kind,
            draft_model=self.draft_model,
            draft_sink_size=self.draft_sink_size,
            draft_window_size=self.draft_window_size,
            allow_full_attention_context=allow_full_attention_context,
        )

# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from collections.abc import Sequence
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.cache.fingerprints import DFlashPrefixKey

FAState = tuple[mx.array, mx.array, int] | tuple[mx.array, mx.array, int, int]

@dataclass(frozen=True)
class TargetHiddenChunks:
    """Sparse target-hidden spans with their logical sequence length."""

    total_len: int
    chunks: tuple[mx.array, ...]
    spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("TargetHiddenChunks requires at least one chunk")
        if len(self.chunks) != len(self.spans):
            raise ValueError("TargetHiddenChunks chunks/spans length mismatch")
        previous_end = 0
        for chunk, (start, end) in zip(self.chunks, self.spans):
            if start < 0 or end < start or end > int(self.total_len):
                raise ValueError(f"Invalid target-hidden span ({start}, {end})")
            if start < previous_end:
                raise ValueError("Target-hidden spans must be ordered and non-overlapping")
            if int(chunk.shape[1]) != end - start:
                raise ValueError(
                    f"Target-hidden chunk length {int(chunk.shape[1])} "
                    f"!= span length {end - start}"
                )
            previous_end = end

    @property
    def shape(self) -> tuple[int, int, int]:
        ref = self.chunks[0]
        return (int(ref.shape[0]), int(self.total_len), int(ref.shape[-1]))

    @property
    def nbytes(self) -> int:
        return sum(int(chunk.nbytes) for chunk in self.chunks)

    @property
    def dtype(self) -> Any:
        return self.chunks[0].dtype

    def astype(self, dtype: Any) -> "TargetHiddenChunks":
        if all(chunk.dtype == dtype for chunk in self.chunks):
            return self
        return TargetHiddenChunks(
            total_len=int(self.total_len),
            chunks=tuple(chunk.astype(dtype) for chunk in self.chunks),
            spans=self.spans,
        )

    def slice(self, start: int, end: int) -> mx.array:
        start = max(0, int(start))
        end = min(int(self.total_len), int(end))
        if end < start:
            raise ValueError(f"Invalid target-hidden slice ({start}, {end})")
        pieces: list[mx.array] = []
        cursor = start
        for chunk, (chunk_start, chunk_end) in zip(self.chunks, self.spans):
            if chunk_end <= cursor:
                continue
            if chunk_start > cursor:
                break
            copy_end = min(end, chunk_end)
            if copy_end > cursor:
                offset = cursor - chunk_start
                pieces.append(chunk[:, offset : offset + copy_end - cursor, :])
                cursor = copy_end
            if cursor >= end:
                break
        if cursor != end:
            raise ValueError(
                f"Target-hidden spans do not cover requested slice ({start}, {end})"
            )
        if not pieces:
            ref = self.chunks[0]
            return mx.zeros(
                (int(ref.shape[0]), 0, int(ref.shape[-1])),
                dtype=ref.dtype,
            )
        return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)

    def materialize(self) -> mx.array:
        ref = self.chunks[0]
        result = mx.zeros(
            (int(ref.shape[0]), int(self.total_len), int(ref.shape[-1])),
            dtype=ref.dtype,
        )
        for chunk, (start, end) in zip(self.chunks, self.spans):
            result[:, start:end, :] = chunk
        mx.eval(result)
        return result

@dataclass
class DFlashPrefixSnapshot:
    token_ids: tuple[int, ...]
    fa_states: tuple[Optional[FAState], ...]
    gdn_states: tuple[Optional[tuple[Optional[mx.array], ...]], ...]
    target_hidden_chunks: tuple[mx.array, ...]
    target_hidden_chunk_spans: tuple[tuple[int, int], ...]
    target_hidden_total_len: int
    last_logits: Optional[mx.array]
    key: DFlashPrefixKey
    kind: str = "prefill"
    created_at: float = field(default_factory=time.time)

    @property
    def prefix_len(self) -> int:
        return len(self.token_ids)

    @property
    def nbytes(self) -> int:
        return sum(self.nbytes_breakdown().values())

    def nbytes_breakdown(self) -> dict[str, int]:
        draft_context_bytes = sum(int(c.nbytes) for c in self.target_hidden_chunks)
        last_logits_bytes = int(self.last_logits.nbytes) if self.last_logits is not None else 0
        fa_bytes = 0
        for fa in self.fa_states:
            if fa is not None:
                k, v = fa[0], fa[1]
                fa_bytes += int(k.nbytes) + int(v.nbytes)
        gdn_bytes = 0
        for gdn in self.gdn_states:
            if gdn is not None:
                for a in gdn:
                    if a is not None:
                        gdn_bytes += int(a.nbytes)
        return {
            "fa_kv": fa_bytes,
            "gdn_state": gdn_bytes,
            "draft_context": draft_context_bytes,
            "last_logits": last_logits_bytes,
        }

def validate_prefix_snapshot(
    snapshot: Optional[DFlashPrefixSnapshot],
    prompt_tokens: Sequence[int],
) -> int:
    if snapshot is None:
        return 0
    snap_len = snapshot.prefix_len
    if snap_len == 0 or snap_len > len(prompt_tokens):
        return 0
    if tuple(prompt_tokens[:snap_len]) != snapshot.token_ids:
        return 0
    return snap_len

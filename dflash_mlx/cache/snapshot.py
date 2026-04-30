# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

from dflash_mlx.cache.fingerprints import DFlashPrefixKey

@dataclass
class DFlashPrefixSnapshot:
    token_ids: tuple[int, ...]
    fa_states: tuple[Optional[tuple[mx.array, mx.array, int]], ...]
    gdn_states: tuple[Optional[tuple[Optional[mx.array], ...]], ...]
    target_hidden_chunks: tuple[mx.array, ...]
    target_hidden_chunk_spans: tuple[tuple[int, int], ...]
    target_hidden_total_len: int
    last_logits: Optional[mx.array]
    key: DFlashPrefixKey
    kind: str = "prefill"
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_full_target_hidden(
        cls,
        *,
        token_ids,
        fa_states,
        gdn_states,
        target_hidden: mx.array,
        last_logits: Optional[mx.array],
        key: DFlashPrefixKey,
        kind: str = "prefill",
    ) -> "DFlashPrefixSnapshot":
        total_len = int(target_hidden.shape[1])
        return cls(
            token_ids=token_ids,
            fa_states=fa_states,
            gdn_states=gdn_states,
            target_hidden_chunks=(target_hidden,),
            target_hidden_chunk_spans=((0, total_len),),
            target_hidden_total_len=total_len,
            last_logits=last_logits,
            key=key,
            kind=kind,
        )

    @property
    def prefix_len(self) -> int:
        return len(self.token_ids)

    @property
    def nbytes(self) -> int:
        return sum(self.nbytes_breakdown().values())

    def nbytes_breakdown(self) -> dict[str, int]:
        target_hidden_bytes = sum(int(c.nbytes) for c in self.target_hidden_chunks)
        last_logits_bytes = int(self.last_logits.nbytes) if self.last_logits is not None else 0
        fa_bytes = 0
        for fa in self.fa_states:
            if fa is not None:
                k, v, _ = fa
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
            "target_hidden": target_hidden_bytes,
            "last_logits": last_logits_bytes,
        }

def validate_prefix_snapshot(
    snapshot: Optional[DFlashPrefixSnapshot],
    prompt_tokens: list[int],
) -> int:
    if snapshot is None:
        return 0
    snap_len = snapshot.prefix_len
    if snap_len == 0 or snap_len > len(prompt_tokens):
        return 0
    if tuple(prompt_tokens[:snap_len]) != snapshot.token_ids:
        return 0
    return snap_len

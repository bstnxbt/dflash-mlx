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
    target_hidden: mx.array
    last_logits: Optional[mx.array]
    key: DFlashPrefixKey
    kind: str = "prefill"
    created_at: float = field(default_factory=time.time)

    @property
    def prefix_len(self) -> int:
        return len(self.token_ids)

    @property
    def nbytes(self) -> int:
        total = int(self.target_hidden.nbytes)
        if self.last_logits is not None:
            total += int(self.last_logits.nbytes)
        for fa in self.fa_states:
            if fa is not None:
                k, v, _ = fa
                total += int(k.nbytes) + int(v.nbytes)
        for gdn in self.gdn_states:
            if gdn is not None:
                for a in gdn:
                    if a is not None:
                        total += int(a.nbytes)
        return total


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

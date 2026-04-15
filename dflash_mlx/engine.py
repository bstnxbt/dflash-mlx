# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx


class _BaseEngine:
    kind: str

    def arm_rollback(
        self,
        target_cache: list[Any],
        *,
        prefix_len: int,
    ) -> None:
        from dflash_mlx import runtime as runtime_mod

        runtime_mod._arm_target_rollback_with_prefix(target_cache, prefix_len=prefix_len)

    def verify(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        verify_chunk_tokens: Optional[int],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        from dflash_mlx import runtime as runtime_mod

        return runtime_mod._verify_target_block(
            target_model=target_model,
            verify_ids=verify_ids,
            target_cache=target_cache,
            verify_chunk_tokens=verify_chunk_tokens,
            capture_layer_ids=capture_layer_ids,
        )

    def rollback(
        self,
        target_cache: list[Any],
        *,
        target_len: int,
        acceptance_len: int,
        drafted_tokens: int,
    ) -> int:
        from dflash_mlx import runtime as runtime_mod

        return runtime_mod._restore_target_cache_after_acceptance(
            target_cache,
            target_len=target_len,
            acceptance_length=acceptance_len,
            drafted_tokens=drafted_tokens,
        )


class FullAttentionEngine(_BaseEngine):
    """Verify + rollback backend for full-attention target models."""

    kind = "full_attention"


class HybridGDNEngine(_BaseEngine):
    """Verify + rollback backend for hybrid GDN target models."""

    kind = "hybrid_gdn"

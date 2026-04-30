# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

class _BaseEngine:
    def arm_rollback(
        self,
        target_cache: list[Any],
        *,
        prefix_len: int,
    ) -> None:
        from dflash_mlx.engine.rollback import arm_target_rollback_with_prefix

        arm_target_rollback_with_prefix(target_cache, prefix_len=prefix_len)

    def verify(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        from dflash_mlx.engine.target_verifier import verify_target_block

        return verify_target_block(
            target_model=target_model,
            verify_ids=verify_ids,
            target_cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def rollback(
        self,
        target_cache: list[Any],
        *,
        target_len: int,
        acceptance_len: int,
        drafted_tokens: int,
        force_replay: bool = False,
    ) -> int:
        from dflash_mlx.engine.rollback import restore_target_cache_after_acceptance

        return restore_target_cache_after_acceptance(
            target_cache,
            target_len=target_len,
            acceptance_length=acceptance_len,
            drafted_tokens=drafted_tokens,
            force_replay=force_replay,
        )

class FullAttentionEngine(_BaseEngine):
    pass

class HybridGDNEngine(_BaseEngine):
    pass

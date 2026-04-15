# Copyright 2026 bstnxbt
# MIT License — see LICENSE file

from __future__ import annotations

from typing import Any

import mlx.core as mx

from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
)


class EagerDraftBackend:
    kind = "eager"

    def make_cache(
        self,
        *,
        draft_model: DFlashDraftModel,
        sink_size: int,
        window_size: int,
    ) -> list[Any]:
        return [
            ContextOnlyDraftKVCache(
                sink_size=sink_size,
                window_size=window_size,
            )
            for _ in range(len(draft_model.layers))
        ]

    def draft_greedy(
        self,
        *,
        target_model: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        target_hidden: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
    ) -> mx.array:
        if int(block_len) <= 1:
            raise ValueError("draft_greedy requires block_len > 1")

        from dflash_mlx import runtime as runtime_mod

        block_token_ids = mx.concatenate(
            [staged_first[:1], mask_token_tail[: int(block_len) - 1]],
            axis=0,
        )
        noise_embedding = runtime_mod._target_embed_tokens(target_model)(block_token_ids[None])
        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = runtime_mod._lm_head_logits(target_model, draft_hidden[:, 1:, :])
        drafted = runtime_mod.greedy_tokens_with_mask(
            draft_logits,
            suppress_token_mask,
        ).squeeze(0)
        if async_launch:
            mx.async_eval(drafted)
        else:
            mx.eval(draft_logits)
        return drafted


def make_draft_backend() -> EagerDraftBackend:
    return EagerDraftBackend()

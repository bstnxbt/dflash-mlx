# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional, Protocol

import mlx.core as mx

from dflash_mlx.engine.sampling import greedy_tokens_with_mask, masked_topk_arrays
from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
    FullContextDraftKVCache,
)


class DraftBackend(Protocol):
    def make_cache(
        self,
        *,
        draft_model: DFlashDraftModel,
        sink_size: int,
        window_size: int,
        allow_full_context_layers: bool = False,
    ) -> list[Any]:
        ...

    def draft_greedy(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
    ) -> mx.array:
        ...

    def propose_block(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        capture_q: bool = False,
    ) -> Any:
        ...

    def draft_with_topk(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        prefix_tokens: mx.array,
        draft_context: mx.array,
        block_len: int,
        suppress_token_mask: Optional[mx.array],
        top_width: int,
    ) -> tuple[mx.array, list[list[int]], list[list[float]]]:
        ...

    def draft_greedy_capture(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
        top_width: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        ...

    def draft_branch_blocks_batch(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        branch_prefixes: list[mx.array],
        draft_context: mx.array,
        block_len: int,
        suppress_token_mask: Optional[mx.array],
    ) -> list[mx.array]:
        ...

    def advance_context(
        self,
        *,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        draft_context: mx.array,
    ) -> None:
        ...


class EagerDraftBackend:
    def make_cache(
        self,
        *,
        draft_model: DFlashDraftModel,
        sink_size: int,
        window_size: int,
        allow_full_context_layers: bool = False,
    ) -> list[Any]:
        caches: list[Any] = []
        layer_types = tuple(getattr(draft_model.args, "layer_types", ()) or ())
        for index in range(len(draft_model.layers)):
            layer_type = str(layer_types[index] if index < len(layer_types) else "")
            if allow_full_context_layers and layer_type == "full_attention":
                caches.append(FullContextDraftKVCache())
            else:
                caches.append(
                    ContextOnlyDraftKVCache(
                        sink_size=sink_size,
                        window_size=window_size,
                    )
                )
        return caches

    def _draft_hidden_and_logits(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
    ) -> mx.array:
        if int(block_len) <= 1:
            raise ValueError("draft_greedy requires block_len > 1")

        block_token_ids = mx.concatenate(
            [staged_first[:1], mask_token_tail[: int(block_len) - 1]],
            axis=0,
        )
        draft_dtype = _draft_compute_dtype(draft_model)
        noise_embedding = target_ops.embed_tokens(target_model)(
            block_token_ids[None]
        )
        if draft_dtype is not None:
            noise_embedding = _astype_if_needed(noise_embedding, draft_dtype)
            draft_context = _astype_if_needed(draft_context, draft_dtype)
        draft_hidden = draft_model.forward_projected_context(
            noise_embedding=noise_embedding,
            draft_context=draft_context,
            cache=draft_cache,
        )
        logits = target_ops.logits_from_hidden(
            target_model,
            draft_hidden[:, 1:, :],
        )
        return draft_hidden[:, 1:, :], logits

    def _draft_block_logits(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
    ) -> mx.array:
        _draft_hidden, draft_logits = self._draft_hidden_and_logits(
            target_model=target_model,
            target_ops=target_ops,
            draft_model=draft_model,
            draft_cache=draft_cache,
            staged_first=staged_first,
            draft_context=draft_context,
            block_len=block_len,
            mask_token_tail=mask_token_tail,
        )
        return draft_logits

    def propose_block(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        capture_q: bool = False,
    ) -> Any:
        selector = getattr(draft_model, "select_proposal", None)
        if not callable(selector):
            raise ValueError("draft model does not expose a proposal selector")
        draft_hidden, logits = self._draft_hidden_and_logits(
            target_model=target_model,
            target_ops=target_ops,
            draft_model=draft_model,
            draft_cache=draft_cache,
            staged_first=staged_first,
            draft_context=draft_context,
            block_len=block_len,
            mask_token_tail=mask_token_tail,
        )
        if suppress_token_mask is not None:
            floor = mx.array(-1e9, dtype=logits.dtype)
            logits = mx.where(suppress_token_mask, floor, logits)
        proposal = selector(
            draft_hidden=draft_hidden,
            logits=logits,
            anchor_ids=staged_first[:1],
            temperature=float(temperature),
            top_p=float(top_p),
            min_p=float(min_p),
            top_k=int(top_k),
            capture_q=bool(capture_q),
        )
        expected_rows = int(block_len) - 1
        if int(proposal.token_ids.shape[0]) != expected_rows:
            raise ValueError(
                "draft proposal token rows must match the verify width: "
                f"expected {expected_rows}, got {proposal.token_ids.shape[0]}"
            )
        return proposal

    def draft_greedy(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
    ) -> mx.array:
        if callable(getattr(draft_model, "select_proposal", None)):
            proposal = self.propose_block(
                target_model=target_model,
                target_ops=target_ops,
                draft_model=draft_model,
                draft_cache=draft_cache,
                staged_first=staged_first,
                draft_context=draft_context,
                block_len=block_len,
                mask_token_tail=mask_token_tail,
                suppress_token_mask=suppress_token_mask,
            )
            drafted = proposal.token_ids.astype(mx.uint32)
            if async_launch:
                mx.async_eval(drafted)
            else:
                mx.eval(drafted)
            return drafted

        draft_logits = self._draft_block_logits(
            target_model=target_model,
            target_ops=target_ops,
            draft_model=draft_model,
            draft_cache=draft_cache,
            staged_first=staged_first,
            draft_context=draft_context,
            block_len=block_len,
            mask_token_tail=mask_token_tail,
        )
        drafted = greedy_tokens_with_mask(
            draft_logits,
            suppress_token_mask,
        ).squeeze(0)
        if async_launch:
            mx.async_eval(drafted)
        else:
            mx.eval(draft_logits)
        return drafted

    def draft_greedy_capture(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        draft_context: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
        top_width: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if callable(getattr(draft_model, "select_proposal", None)):
            proposal = self.propose_block(
                target_model=target_model,
                target_ops=target_ops,
                draft_model=draft_model,
                draft_cache=draft_cache,
                staged_first=staged_first,
                draft_context=draft_context,
                block_len=block_len,
                mask_token_tail=mask_token_tail,
                suppress_token_mask=suppress_token_mask,
                capture_q=True,
            )
            if proposal.q_token_ids is None or proposal.q_probs is None:
                raise ValueError("draft proposal did not return capture candidates")
            width = min(int(top_width), int(proposal.q_token_ids.shape[-1]))
            order = mx.argsort(proposal.q_probs, axis=-1)[..., -width:]
            order = mx.flip(order, axis=-1)
            top_ids = mx.take_along_axis(proposal.q_token_ids, order, axis=-1)
            top_logprobs = mx.log(
                mx.take_along_axis(proposal.q_probs, order, axis=-1)
            )
            drafted = proposal.token_ids.astype(mx.uint32)
            if async_launch:
                mx.async_eval(drafted, top_ids, top_logprobs)
            else:
                mx.eval(drafted, top_ids, top_logprobs)
            return drafted, top_ids, top_logprobs
        draft_logits = self._draft_block_logits(
            target_model=target_model,
            target_ops=target_ops,
            draft_model=draft_model,
            draft_cache=draft_cache,
            staged_first=staged_first,
            draft_context=draft_context,
            block_len=block_len,
            mask_token_tail=mask_token_tail,
        )
        drafted = greedy_tokens_with_mask(
            draft_logits,
            suppress_token_mask,
        ).squeeze(0)
        top_ids, top_logprobs = masked_topk_arrays(
            draft_logits.squeeze(0),
            suppress_token_mask,
            width=top_width,
        )
        if async_launch:
            mx.async_eval(drafted, top_ids, top_logprobs)
        else:
            mx.eval(draft_logits, top_ids, top_logprobs)
        return drafted, top_ids, top_logprobs

    def draft_with_topk(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        prefix_tokens: mx.array,
        draft_context: mx.array,
        block_len: int,
        suppress_token_mask: Optional[mx.array],
        top_width: int,
    ) -> tuple[mx.array, list[list[int]], list[list[float]]]:
        if callable(getattr(draft_model, "select_proposal", None)):
            raise ValueError("DFlash2 drafts do not support DDTree/top-k draft paths")
        from dflash_mlx.engine.ddtree import draft_block_with_topk

        drafted, top_ids, top_values, _draft_us = draft_block_with_topk(
            target_model=target_model,
            target_ops=target_ops,
            draft_model=draft_model,
            draft_cache=draft_cache,
            prefix_tokens=prefix_tokens,
            draft_context=draft_context,
            block_len=block_len,
            suppress_token_mask=suppress_token_mask,
            top_width=top_width,
        )
        return drafted, top_ids, top_values

    def draft_branch_blocks_batch(
        self,
        *,
        target_model: Any,
        target_ops: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        branch_prefixes: list[mx.array],
        draft_context: mx.array,
        block_len: int,
        suppress_token_mask: Optional[mx.array],
    ) -> list[mx.array]:
        if callable(getattr(draft_model, "select_proposal", None)):
            raise ValueError("DFlash2 drafts do not support DDTree/top-k draft paths")
        from dflash_mlx.engine.ddtree import draft_branch_blocks_batch

        candidate_ids, _draft_us = draft_branch_blocks_batch(
            target_model=target_model,
            target_ops=target_ops,
            draft_model=draft_model,
            draft_cache=draft_cache,
            branch_prefixes=branch_prefixes,
            draft_context=draft_context,
            block_len=block_len,
            suppress_token_mask=suppress_token_mask,
        )
        return candidate_ids

    def advance_context(
        self,
        *,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        draft_context: mx.array,
    ) -> None:
        draft_model.advance_projected_context_cache(
            draft_context=_astype_if_needed(
                draft_context,
                _draft_compute_dtype(draft_model),
            ),
            cache=draft_cache,
        )


def _draft_compute_dtype(draft_model: DFlashDraftModel) -> Any | None:
    for attr_path in (
        ("hidden_norm", "weight"),
        ("norm", "weight"),
        ("fc", "scales"),
        ("fc", "weight"),
    ):
        value: Any = draft_model
        for attr in attr_path:
            value = getattr(value, attr, None)
            if value is None:
                break
        if hasattr(value, "dtype") and mx.issubdtype(value.dtype, mx.floating):
            return value.dtype
    return None


def _astype_if_needed(value: mx.array, dtype: Any | None) -> mx.array:
    if dtype is None:
        return value
    from dflash_mlx.cache.snapshot import TargetHiddenChunks

    if isinstance(value, TargetHiddenChunks):
        return value.astype(dtype)
    if value.dtype == dtype:
        return value
    return value.astype(dtype)

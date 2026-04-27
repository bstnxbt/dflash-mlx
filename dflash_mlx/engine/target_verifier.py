# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)
from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def _target_text_wrapper(target_model: Any) -> Any:
    if hasattr(target_model, "model"):
        return target_model
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    raise AttributeError(f"Unsupported target model wrapper: {type(target_model)!r}")


def _target_text_model(target_model: Any) -> Any:
    wrapper = _target_text_wrapper(target_model)
    if hasattr(wrapper, "model"):
        return wrapper.model
    raise AttributeError(f"Unsupported target text model: {type(wrapper)!r}")


def _lm_head_logits(target_model: Any, hidden_states: mx.array) -> mx.array:
    wrapper = _target_text_wrapper(target_model)
    if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True):
        return wrapper.model.embed_tokens.as_linear(hidden_states)
    return wrapper.lm_head(hidden_states)


def extract_context_feature_from_dict(
    captured_dict: dict[int, mx.array],
    target_layer_ids: list[int],
) -> mx.array:
    selected = [captured_dict[layer_id + 1] for layer_id in target_layer_ids]
    return mx.concatenate(selected, axis=-1)


def target_forward_with_hidden_states(
    target_model: Any,
    *,
    input_ids: Optional[mx.array] = None,
    cache: Optional[list[Any]] = None,
    input_embeddings: Optional[mx.array] = None,
    capture_layer_ids: Optional[set[int]] = None,
) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
    inner = _target_text_model(target_model)
    hidden_states = input_embeddings if input_embeddings is not None else inner.embed_tokens(input_ids)
    if cache is None:
        cache = [None] * len(inner.layers)
    capture_all = capture_layer_ids is None
    if capture_all:
        captured: list[mx.array] | dict[int, mx.array] = [hidden_states]
    else:
        capture_layer_ids = set(capture_layer_ids)
        captured = {0: hidden_states} if 0 in capture_layer_ids else {}
    h = hidden_states

    if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
        fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            h = layer(h, mask=mask, cache=layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h
    else:
        mask = create_attention_mask(hidden_states, cache[0])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            h = layer(h, mask, layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h
    normalized = inner.norm(h)
    logits = _lm_head_logits(target_model, normalized)
    return logits, captured


def verify_target_block(
    *,
    target_model: Any,
    verify_ids: mx.array,
    target_cache: list[Any],
    capture_layer_ids: Optional[set[int]] = None,
) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
    if int(verify_ids.shape[1]) <= 0:
        raise ValueError("verify block must contain at least one token")
    return target_forward_with_hidden_states(
        target_model,
        input_ids=verify_ids,
        cache=target_cache,
        capture_layer_ids=capture_layer_ids,
    )

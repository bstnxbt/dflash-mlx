# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import time
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models import cache as cache_mod

from dflash_mlx.engine.gqa_sdpa import (
    async_per_head_gqa_sdpa,
    grouped_gqa_sdpa,
    per_head_gqa_sdpa,
    repeat_gqa_mask,
)
from dflash_mlx.engine.sparse_rope import SPARSE_POSITIONS_ATTR
from dflash_mlx.engine.target_ops import TargetCapabilities


def _cache_offset(cache: list[Any]) -> int:
    for entry in cache:
        if entry is not None:
            return int(getattr(entry, "offset", 0) or 0)
    return 0


def _sparse_layer_masks(inner: Any, h: mx.array, positions: mx.array, cache: list[Any]):
    """Per-layer attention masks for sparse prefill, using true token positions.

    Full-attention layers get a plain causal mask (positions are monotonic, so
    causal-over-positions equals causal-over-index). Sliding layers additionally
    require the query/key *position* distance to be within the window — this is
    what differs from the dense contiguous mask. Keys span the already-cached
    selected tokens plus the current chunk.
    """
    n_q = int(h.shape[1])
    offset = _cache_offset(cache)
    total = offset + n_q
    key_pos = positions[:total]
    q_pos = key_pos[offset:total][:, None]
    k_pos = key_pos[None, :]
    causal = q_pos >= k_pos
    window = int(getattr(inner, "window_size", 0) or 0)
    sliding = (causal & (q_pos - k_pos < window)) if window > 0 else causal
    masks = []
    for layer in inner.layers:
        is_sliding = getattr(layer, "layer_type", None) == "sliding_attention"
        masks.append(sliding if is_sliding else causal)
    return masks


def _model_args(model: Any) -> Any:
    args = getattr(model, "args", None)
    if args is not None:
        return args
    if hasattr(model, "language_model"):
        args = getattr(model.language_model, "args", None)
        if args is not None:
            return args
    return None


def _logit_softcap(logits: mx.array, softcap: Optional[float]) -> mx.array:
    if softcap is None:
        return logits
    cap = float(softcap)
    if cap <= 0.0:
        return logits
    return mx.tanh(logits / cap) * cap


def _trim_recent_cache(cache_entry: Any, n: int) -> int:
    n = max(0, int(n))
    if n <= 0:
        return 0

    offset = int(getattr(cache_entry, "offset", 0) or 0)
    n = min(offset, n)
    if n <= 0:
        return 0

    if isinstance(cache_entry, cache_mod.RotatingKVCache):
        keys = getattr(cache_entry, "keys", None)
        values = getattr(cache_entry, "values", None)
        if keys is not None and values is not None:
            keys = cache_entry._temporal_order(keys)
            values = cache_entry._temporal_order(values)
            keep_len = max(0, int(keys.shape[2]) - n)
            cache_entry.keys = keys[..., :keep_len, :]
            cache_entry.values = values[..., :keep_len, :]
            cache_entry._idx = keep_len
        else:
            cache_entry._idx = max(0, int(getattr(cache_entry, "_idx", 0) or 0) - n)
        cache_entry.offset = offset - n
        return n

    if hasattr(cache_entry, "trim"):
        return int(cache_entry.trim(n))
    if hasattr(cache_entry, "offset"):
        cache_entry.offset = offset - n
        return n
    return 0


def _gemma4_split_candidate(attn: Any, x: mx.array, mask: Optional[Any], cache: Any) -> bool:
    if cache is None or getattr(attn, "is_sliding", False):
        return False
    if not getattr(attn, "has_kv", True):
        return False
    if not (
        mask is None
        or (isinstance(mask, str) and mask == "causal")
        or isinstance(mask, mx.array)
    ):
        return False

    q_len = int(x.shape[1])
    head_dim = int(getattr(attn, "head_dim", 0) or 0)
    query_heads = int(getattr(attn, "n_heads", 0) or 0)
    kv_heads = int(getattr(attn, "n_kv_heads", 0) or 0)
    if (
        head_dim != 512
        or q_len <= 0
        or q_len > 16
        or kv_heads <= 0
        or query_heads % kv_heads != 0
    ):
        return False

    kv_len = int(getattr(cache, "offset", 0) or 0) + q_len
    if kv_heads <= 2:
        return kv_len >= 8192
    if q_len <= 4:
        return kv_len >= 16384
    return q_len == 16 and kv_len >= 32768


def _gemma4_full_gqa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any],
) -> mx.array:
    _, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, kv_len, _ = keys.shape
    if kv_heads <= 0 or query_heads == kv_heads or query_heads % kv_heads != 0:
        return grouped_gqa_sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        )

    gqa = query_heads // kv_heads
    if (
        int(head_dim) != 512
        or int(q_len) <= 0
        or int(q_len) > 16
        or queries.dtype not in (mx.bfloat16, mx.float16)
    ):
        return grouped_gqa_sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        )
    if int(kv_heads) <= 2 and int(kv_len) >= 8192:
        grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
        return per_head_gqa_sdpa(
            queries,
            keys,
            values,
            scale=scale,
            mask=grouped_mask,
            gqa=gqa,
        )
    if int(kv_heads) >= 4 and int(q_len) <= 4 and int(kv_len) >= 16384:
        grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
        return per_head_gqa_sdpa(
            queries,
            keys,
            values,
            scale=scale,
            mask=grouped_mask,
            gqa=gqa,
        )
    if int(kv_heads) >= 4 and int(q_len) == 16 and int(kv_len) >= 32768:
        grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
        return async_per_head_gqa_sdpa(
            queries,
            keys,
            values,
            scale=scale,
            mask=grouped_mask,
            gqa=gqa,
        )
    return grouped_gqa_sdpa(
        queries,
        keys,
        values,
        cache=cache,
        scale=scale,
        mask=mask,
    )


def _install_full_attention_gqa_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_full_attention_gqa_installed", False):
        return

    original_call = cls.__call__

    def attention_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
    ) -> Any:
        if (
            shared_kv is not None
            or not _gemma4_split_candidate(self, x, mask, cache)
        ):
            return original_call(
                self,
                x,
                mask=mask,
                cache=cache,
                shared_kv=shared_kv,
                offset=offset,
            )

        batch_size, seq_len, _ = x.shape
        queries = self.q_proj(x).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        queries = self.q_norm(queries)

        keys = self.k_proj(x).reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        values = keys
        if not getattr(self, "use_k_eq_v", False):
            values = self.v_proj(x).reshape(
                batch_size, seq_len, self.n_kv_heads, self.head_dim
            )

        offset = mx.array(cache.offset)

        keys = self.k_norm(keys)
        keys = keys.transpose(0, 2, 1, 3)
        keys = self.rope(keys, offset=offset)

        values = self.v_norm(values)
        values = values.transpose(0, 2, 1, 3)

        queries = queries.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        keys, values = cache.update_and_fetch(keys, values)
        output = _gemma4_full_gqa_sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.o_proj(output), (keys, values), offset

    cls.__call__ = attention_call
    cls._dflash_full_attention_gqa_installed = True


class Gemma4TargetOps:
    backend_name = "gemma4"

    def model_type(self, target_model: Any) -> str:
        args = _model_args(target_model)
        value = getattr(args, "model_type", None)
        if value is not None:
            return str(value).lower()
        if hasattr(target_model, "language_model"):
            value = getattr(
                getattr(target_model.language_model, "args", None),
                "model_type",
                None,
            )
            if value is not None:
                return str(value).lower()
        config = getattr(target_model, "config", None)
        if isinstance(config, dict):
            text_config = config.get("text_config", config)
            return str(text_config.get("model_type", config.get("model_type", ""))).lower()
        return ""

    def supports_model(self, target_model: Any) -> bool:
        model_type = self.model_type(target_model)
        if model_type not in ("gemma4", "gemma4_text"):
            return False
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        args = getattr(self.text_wrapper(target_model), "args", None)
        return (
            hasattr(inner, "layers")
            and hasattr(inner, "embed_tokens")
            and hasattr(args, "layer_types")
        )

    def family(self, target_model: Any) -> str:
        return "gemma4_swa"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        args = getattr(self.text_wrapper(target_model), "args", None)
        shared_kv_layers_raw = getattr(args, "num_kv_shared_layers", None)
        if type(shared_kv_layers_raw) is int:
            shared_kv_layers = shared_kv_layers_raw
        else:
            shared_kv_layers = -1
        shared_kv_known = shared_kv_layers >= 0
        shared_kv = bool(shared_kv_known and shared_kv_layers > 0)
        prefix_snapshot_safe = bool(shared_kv_known and shared_kv_layers == 0)
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=False,
            supports_kv_trim=True,
            supports_prefix_snapshot=prefix_snapshot_safe,
            supports_rotating_cache_snapshot=prefix_snapshot_safe,
            supports_shared_kv=shared_kv,
            supports_target_hidden_capture=True,
            supports_verify_linear=True,
            supports_full_context_draft_layers=True,
            supports_tree_verify=False,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        del cache_entries
        return False

    def text_wrapper(self, target_model: Any) -> Any:
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model"):
            return target_model
        raise AttributeError(f"Unsupported Gemma4 model wrapper: {type(target_model)!r}")

    def text_model(self, target_model: Any) -> Any:
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "model"):
            return wrapper.model
        raise AttributeError(f"Unsupported Gemma4 text model: {type(wrapper)!r}")

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(self, target_model: Any, hidden_states: mx.array) -> mx.array:
        wrapper = self.text_wrapper(target_model)
        inner = self.text_model(target_model)
        tied = getattr(
            wrapper,
            "tie_word_embeddings",
            getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True),
        )
        if bool(tied):
            logits = inner.embed_tokens.as_linear(hidden_states)
        else:
            logits = wrapper.lm_head(hidden_states)
        softcap = getattr(
            wrapper,
            "final_logit_softcapping",
            getattr(getattr(wrapper, "args", None), "final_logit_softcapping", None),
        )
        return _logit_softcap(
            logits,
            softcap,
        )

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
    ) -> list[Any]:
        if quantize_kv_cache:
            raise ValueError("Gemma4 target KV quantization is not supported yet")
        if target_fa_window is not None and int(target_fa_window) > 0:
            raise ValueError("Gemma4 uses its config-defined SWA/full attention cache")
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "make_cache"):
            return wrapper.make_cache()
        if hasattr(target_model, "make_cache"):
            return target_model.make_cache()
        raise AttributeError("Gemma4 target must expose make_cache()")

    def install_speculative_hooks(self, target_model: Any) -> None:
        text_model = self.text_model(target_model)
        if getattr(text_model, "_dflash_speculative_hooks_installed", False):
            return
        for layer in text_model.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                _install_full_attention_gqa_hook(attn)
        text_model._dflash_speculative_hooks_installed = True

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        inner = self.text_model(target_model)
        if input_embeddings is None:
            input_embeddings = inner.embed_tokens(input_ids)
        h = input_embeddings * getattr(inner, "embed_scale", 1.0)

        per_layer_inputs = [None] * len(inner.layers)
        if getattr(inner, "hidden_size_per_layer_input", 0):
            pli = inner._get_per_layer_inputs(input_ids, input_embeddings)
            pli = inner._project_per_layer_inputs(h, pli)
            per_layer_inputs = [pli[:, :, i, :] for i, _ in enumerate(inner.layers)]

        if cache is None:
            cache = [None] * len(inner.layers)
        else:
            cache = list(cache) + [None] * (len(inner.layers) - len(cache))

        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [h]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: h} if 0 in capture_layer_ids else {}

        sparse_positions = getattr(inner, SPARSE_POSITIONS_ATTR, None)
        if sparse_positions is not None:
            masks = _sparse_layer_masks(inner, h, sparse_positions, cache)
        else:
            masks = inner._make_masks(h, cache)
        intermediates = [(None, None)] * len(inner.layers)
        for idx, (layer, layer_cache, mask, prev_idx, per_layer_input) in enumerate(
            zip(
                inner.layers,
                cache,
                masks,
                inner.previous_kvs,
                per_layer_inputs,
                strict=True,
            )
        ):
            shared_kv, offset = intermediates[prev_idx]
            h, shared_kv, offset = layer(
                h,
                mask,
                layer_cache,
                per_layer_input=per_layer_input,
                shared_kv=shared_kv,
                offset=offset,
            )
            capture_key = idx + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h
            intermediates[idx] = (shared_kv, offset)

        normalized = inner.norm(h)
        if logits_last_only and isinstance(captured, dict):
            captured[-1] = normalized
        logits_hidden = normalized[:, -1:, :] if logits_last_only else normalized
        logits = self.logits_from_hidden(target_model, logits_hidden)
        return logits, captured

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        if int(verify_ids.shape[1]) <= 0:
            raise ValueError("verify block must contain at least one token")
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        del target_model, tree_inputs, target_cache, capture_layer_ids
        raise NotImplementedError("Gemma4 DDTree target-tree verification is not implemented")

    def restore_after_tree_acceptance(
        self,
        cache_entries: list[Any],
        *,
        accepted_tree_indices: list[int],
    ) -> int:
        del cache_entries, accepted_tree_indices
        raise NotImplementedError("Gemma4 DDTree target-tree cache commit is not implemented")

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array] | list[mx.array],
        target_layer_ids: list[int],
    ) -> mx.array:
        selected = [captured_dict[int(layer_id) + 1] for layer_id in target_layer_ids]
        return mx.concatenate(selected, axis=-1)

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        return None

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        replay_ns_total = 0
        for cache_entry in cache_entries:
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset <= target_len:
                continue
            trim_n = offset - int(target_len)
            replay_start_ns = time.perf_counter_ns()
            _trim_recent_cache(cache_entry, trim_n)
            replay_ns_total += time.perf_counter_ns() - replay_start_ns
        return replay_ns_total

    def cleanup_generation_caches(
        self,
        target_cache: list[Any],
        draft_cache: list[Any],
    ) -> None:
        draft_cache.clear()
        target_cache.clear()

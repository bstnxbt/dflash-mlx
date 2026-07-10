# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask, scaled_dot_product_attention
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers <= 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (index * span) / (num_draft_layers - 1)))
        for index in range(num_draft_layers)
    ]

_DRAFT_LAYER_TYPES = frozenset(("full_attention", "sliding_attention"))
_GEMMA4_MODEL_TYPES = frozenset(("gemma4", "gemma4_text"))
_GEMMA4_DEFAULT_SLIDING_WINDOW = 512
_GEMMA4_DEFAULT_SLIDING_WINDOW_PATTERN = 5


def _is_gemma4_model_type(model_type: str) -> bool:
    return str(model_type or "").lower() in _GEMMA4_MODEL_TYPES


def _default_draft_layer_types(
    *,
    model_type: str,
    num_hidden_layers: int,
    sliding_window_pattern: int | None,
) -> tuple[str, ...]:
    layer_count = int(num_hidden_layers)
    if not _is_gemma4_model_type(model_type):
        return ()
    pattern_len = int(sliding_window_pattern or _GEMMA4_DEFAULT_SLIDING_WINDOW_PATTERN)
    pattern_len = max(1, pattern_len)
    pattern = ("sliding_attention",) * (pattern_len - 1) + ("full_attention",)
    repeats = (layer_count // len(pattern)) + 1
    return (pattern * repeats)[:layer_count]


class ContextOnlyDraftKVCache:
    def __init__(self, sink_size: int = 64, window_size: int = 1024):
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)
        self.keys = None
        self.values = None
        self.positions = None
        self.offset = 0

    def append_context(
        self,
        context_keys: mx.array,
        context_values: mx.array,
        num_positions: int,
        *,
        positions: Optional[mx.array] = None,
        advance_positions: Optional[int] = None,
    ) -> None:
        if context_keys is None or context_values is None or int(num_positions) <= 0:
            return
        append_len = int(context_keys.shape[2])
        if append_len <= 0:
            self.offset += int(advance_positions if advance_positions is not None else num_positions)
            return
        if positions is None:
            new_positions = mx.arange(
                self.offset,
                self.offset + append_len,
                dtype=mx.int32,
            )
        else:
            if int(positions.shape[0]) != append_len:
                raise ValueError(
                    f"positions length {positions.shape[0]} does not match cache append length {append_len}"
                )
            new_positions = positions
        if self.keys is None:
            self.keys = context_keys
            self.values = context_values
            self.positions = new_positions
        else:
            self.keys = mx.concatenate([self.keys, context_keys], axis=2)
            self.values = mx.concatenate([self.values, context_values], axis=2)
            self.positions = mx.concatenate([self.positions, new_positions], axis=0)
        self.offset += int(advance_positions if advance_positions is not None else num_positions)
        self._apply_window()

    def context_spans_to_append(self, num_positions: int) -> list[tuple[int, int]]:
        num_positions = int(num_positions)
        if num_positions <= 0:
            return []
        cache_len = self.cache_length()
        max_len = self.sink_size + self.window_size
        if cache_len == 0:
            if num_positions <= max_len:
                return [(0, num_positions)]
            spans: list[tuple[int, int]] = []
            sink_end = min(self.sink_size, num_positions)
            if sink_end > 0:
                spans.append((0, sink_end))
            tail_start = max(sink_end, num_positions - self.window_size)
            if tail_start < num_positions:
                spans.append((tail_start, num_positions))
            return spans
        if num_positions <= self.window_size:
            return [(0, num_positions)]
        return [(num_positions - self.window_size, num_positions)]

    def _apply_window(self) -> None:
        if self.keys is None or self.values is None:
            return
        cache_len = int(self.keys.shape[2])
        max_len = self.sink_size + self.window_size
        if cache_len <= max_len:
            return
        sink_k = self.keys[:, :, : self.sink_size, :]
        sink_v = self.values[:, :, : self.sink_size, :]
        sink_p = self.positions[: self.sink_size]
        window_k = self.keys[:, :, -self.window_size :, :]
        window_v = self.values[:, :, -self.window_size :, :]
        window_p = self.positions[-self.window_size :]
        self.keys = mx.concatenate([sink_k, window_k], axis=2)
        self.values = mx.concatenate([sink_v, window_v], axis=2)
        self.positions = mx.concatenate([sink_p, window_p], axis=0)

    def fetch(self) -> tuple[Optional[mx.array], Optional[mx.array]]:
        return self.keys, self.values

    def position_indices(self) -> Optional[mx.array]:
        return self.positions

    def cache_length(self) -> int:
        if self.keys is None:
            return 0
        return int(self.keys.shape[2])


class FullContextDraftKVCache(ContextOnlyDraftKVCache):
    # In-place step-grown buffer: a per-cycle concat churns the Metal heap
    # (progressive ×5 wall at ~7000 cycles).
    step = 256

    def __init__(self):
        super().__init__(sink_size=0, window_size=0)
        self._length = 0

    def _grow(self, template_keys: mx.array, template_values: mx.array, needed: int) -> None:
        capacity = ((int(needed) + self.step - 1) // self.step) * self.step
        batch = int(template_keys.shape[0])
        heads = int(template_keys.shape[1])
        key_dim = int(template_keys.shape[3])
        value_dim = int(template_values.shape[3])
        new_keys = mx.zeros((batch, heads, capacity, key_dim), dtype=template_keys.dtype)
        new_values = mx.zeros((batch, heads, capacity, value_dim), dtype=template_values.dtype)
        if self.keys is not None and self._length > 0:
            new_keys[:, :, : self._length, :] = self.keys[:, :, : self._length, :]
            new_values[:, :, : self._length, :] = self.values[:, :, : self._length, :]
        self.keys = new_keys
        self.values = new_values

    def append_context(
        self,
        context_keys: mx.array,
        context_values: mx.array,
        num_positions: int,
        *,
        positions: Optional[mx.array] = None,
        advance_positions: Optional[int] = None,
    ) -> None:
        if context_keys is None or context_values is None or int(num_positions) <= 0:
            return
        append_len = int(context_keys.shape[2])
        advance = int(advance_positions if advance_positions is not None else num_positions)
        if append_len <= 0:
            self.offset += advance
            return
        if positions is not None and int(positions.shape[0]) != append_len:
            raise ValueError(
                f"positions length {positions.shape[0]} does not match cache append length {append_len}"
            )
        needed = self._length + append_len
        if self.keys is None or needed > int(self.keys.shape[2]):
            self._grow(context_keys, context_values, needed)
        self.keys[:, :, self._length : needed, :] = context_keys
        self.values[:, :, self._length : needed, :] = context_values
        self._length = needed
        self.offset += advance

    def context_spans_to_append(self, num_positions: int) -> list[tuple[int, int]]:
        num_positions = int(num_positions)
        if num_positions <= 0:
            return []
        return [(0, num_positions)]

    def fetch(self) -> tuple[Optional[mx.array], Optional[mx.array]]:
        if self.keys is None or self._length <= 0:
            return None, None
        return (
            self.keys[:, :, : self._length, :],
            self.values[:, :, : self._length, :],
        )

    def fetch_with_block(
        self,
        block_keys: mx.array,
        block_values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        block_len = int(block_keys.shape[2])
        if self.keys is None or self._length <= 0:
            return block_keys, block_values
        needed = self._length + block_len
        if needed > int(self.keys.shape[2]):
            self._grow(block_keys, block_values, needed)
        self.keys[:, :, self._length : needed, :] = block_keys
        self.values[:, :, self._length : needed, :] = block_values
        return (
            self.keys[:, :, :needed, :],
            self.values[:, :, :needed, :],
        )

    def position_indices(self) -> Optional[mx.array]:
        if self._length <= 0:
            return None
        return mx.arange(self._length, dtype=mx.int32)

    def cache_length(self) -> int:
        return self._length


@dataclass
class DFlashDraftModelArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    num_target_layers: int
    block_size: int
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_scaling: Optional[dict[str, Any]] = None
    layer_types: tuple[str, ...] = ()
    sliding_window: Optional[int] = None
    sliding_window_pattern: Optional[int] = None
    dflash_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "DFlashDraftModelArgs":
        data = dict(params)
        # transformers >= 5.x nests rope fields under rope_parameters
        # (e.g. the z-lab gemma4 drafters); hoist them so the required
        # top-level rope_theta is satisfied for both conventions.
        rope_params = data.get("rope_parameters")
        if isinstance(rope_params, dict):
            if "rope_theta" not in data and "rope_theta" in rope_params:
                data["rope_theta"] = rope_params["rope_theta"]
            if (
                "rope_scaling" not in data
                and rope_params.get("rope_type", "default") != "default"
            ):
                data["rope_scaling"] = rope_params
        layer_types = tuple(data.get("layer_types") or ())
        model_type = str(data.get("model_type", ""))
        if (
            not layer_types
            and "num_hidden_layers" in data
            and _is_gemma4_model_type(model_type)
        ):
            layer_types = _default_draft_layer_types(
                model_type=model_type,
                num_hidden_layers=int(data["num_hidden_layers"]),
                sliding_window_pattern=data.get("sliding_window_pattern"),
            )
            if (
                "sliding_window" not in data
                and "sliding_attention" in layer_types
            ):
                data["sliding_window"] = _GEMMA4_DEFAULT_SLIDING_WINDOW
        data["layer_types"] = layer_types
        data["dflash_config"] = dict(data.get("dflash_config") or {})
        return cls(
            **{key: value for key, value in data.items() if key in cls.__annotations__}
        )

    def __post_init__(self) -> None:
        layer_types = tuple(self.layer_types or ())
        if not layer_types and _is_gemma4_model_type(self.model_type):
            layer_types = _default_draft_layer_types(
                model_type=self.model_type,
                num_hidden_layers=int(self.num_hidden_layers),
                sliding_window_pattern=self.sliding_window_pattern,
            )
            if (
                self.sliding_window is None
                and "sliding_attention" in layer_types
            ):
                self.sliding_window = _GEMMA4_DEFAULT_SLIDING_WINDOW
        if layer_types and len(layer_types) != int(self.num_hidden_layers):
            raise ValueError(
                "DFlash draft layer_types length must match num_hidden_layers: "
                f"{len(layer_types)} != {int(self.num_hidden_layers)}"
            )
        unknown = sorted(set(layer_types) - _DRAFT_LAYER_TYPES)
        if unknown:
            raise ValueError(f"Unknown DFlash draft layer type(s): {', '.join(unknown)}")
        if "sliding_attention" in layer_types and int(self.sliding_window or 0) <= 0:
            raise ValueError("sliding_attention draft layers require a positive sliding_window")
        self.layer_types = layer_types

class DFlashAttention(nn.Module):
    def __init__(self, args: DFlashDraftModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        layer_type = args.layer_types[layer_idx] if layer_idx < len(args.layer_types) else ""
        self.sliding_window = (
            int(args.sliding_window or 0)
            if layer_type == "sliding_attention" and args.sliding_window
            else None
        )
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def _attention_mask(
        self,
        *,
        block_len: int,
        query_offset: int,
        key_len: int,
        key_positions: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        if self.sliding_window is None:
            return None
        full_key_len = query_offset + block_len
        if key_positions is None and int(key_len) == full_key_len:
            return create_causal_mask(
                block_len,
                offset=query_offset,
                window_size=self.sliding_window,
            )

        query_positions = mx.arange(
            query_offset,
            query_offset + block_len,
            dtype=mx.int32,
        )
        if key_positions is None:
            key_start = full_key_len - int(key_len)
            key_positions = mx.arange(key_start, full_key_len, dtype=mx.int32)
        return (query_positions[:, None] >= key_positions[None, :]) & (
            query_positions[:, None] < key_positions[None, :] + self.sliding_window
        )

    def _context_segments_for_cache(
        self,
        target_hidden: Any,
        cache: Any,
    ) -> tuple[mx.array, list[tuple[int, int]]]:
        from dflash_mlx.cache.snapshot import TargetHiddenChunks

        is_chunks = isinstance(target_hidden, TargetHiddenChunks)
        total_len = int(target_hidden.shape[1])
        if not isinstance(cache, ContextOnlyDraftKVCache):
            if is_chunks:
                return target_hidden.slice(0, total_len), [(0, total_len)]
            return target_hidden, [(0, total_len)]
        spans = cache.context_spans_to_append(total_len)
        if not spans:
            if is_chunks:
                return target_hidden.slice(0, 0), []
            return target_hidden[:, :0, :], []
        if is_chunks:
            pieces = [target_hidden.slice(start, end) for start, end in spans]
            return (
                pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1),
                spans,
            )
        if len(spans) == 1:
            start, end = spans[0]
            return target_hidden[:, start:end, :], spans
        return mx.concatenate([target_hidden[:, start:end, :] for start, end in spans], axis=1), spans

    def append_projected_context_cache(
        self,
        *,
        target_hidden: mx.array,
        cache: Any,
    ) -> None:
        if not isinstance(cache, ContextOnlyDraftKVCache):
            raise TypeError("draft context advance requires a DFlash draft KV cache")
        ctx_len = int(target_hidden.shape[1])
        if ctx_len <= 0:
            return
        context_hidden, context_spans = self._context_segments_for_cache(target_hidden, cache)
        selected_ctx_len = int(context_hidden.shape[1])
        context_keys = self.k_proj(context_hidden)
        context_keys = self.k_norm(
            context_keys.reshape(target_hidden.shape[0], selected_ctx_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        context_values = self.v_proj(context_hidden).reshape(
            target_hidden.shape[0], selected_ctx_len, self.n_kv_heads, -1,
        ).transpose(0, 2, 1, 3)
        context_keys, context_values, context_positions = self._rope_context_segments(
            context_keys,
            context_values,
            cache_offset=int(cache.offset),
            spans=context_spans,
        )
        cache.append_context(
            context_keys,
            context_values,
            ctx_len,
            positions=context_positions,
            advance_positions=ctx_len,
        )

    def _rope_context_segments(
        self,
        context_keys: mx.array,
        context_values: mx.array,
        *,
        cache_offset: int,
        spans: list[tuple[int, int]],
    ) -> tuple[mx.array, mx.array, mx.array]:
        key_segments = []
        value_segments = []
        position_segments = []
        cursor = 0
        for start, end in spans:
            seg_len = int(end) - int(start)
            if seg_len <= 0:
                continue
            key_segment = context_keys[:, :, cursor : cursor + seg_len, :]
            value_segment = context_values[:, :, cursor : cursor + seg_len, :]
            key_segments.append(self.rope(key_segment, offset=cache_offset + int(start)))
            value_segments.append(value_segment)
            position_segments.append(
                mx.arange(
                    cache_offset + int(start),
                    cache_offset + int(end),
                    dtype=mx.int32,
                )
            )
            cursor += seg_len
        if not key_segments:
            return (
                context_keys[:, :, :0, :],
                context_values[:, :, :0, :],
                mx.array([], dtype=mx.int32),
            )
        if len(key_segments) == 1:
            return key_segments[0], value_segments[0], position_segments[0]
        return (
            mx.concatenate(key_segments, axis=-2),
            mx.concatenate(value_segments, axis=-2),
            mx.concatenate(position_segments, axis=0),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, block_len, _ = hidden_states.shape
        ctx_len = int(target_hidden.shape[1])

        queries = self.q_proj(hidden_states)
        queries = self.q_norm(queries.reshape(batch, block_len, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )

        context_hidden, context_spans = self._context_segments_for_cache(target_hidden, cache)
        selected_ctx_len = int(context_hidden.shape[1])

        context_keys = self.k_proj(context_hidden)
        context_keys = self.k_norm(
            context_keys.reshape(batch, selected_ctx_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        context_values = self.v_proj(context_hidden).reshape(
            batch, selected_ctx_len, self.n_kv_heads, -1,
        ).transpose(0, 2, 1, 3)

        noise_keys = self.k_proj(hidden_states)
        noise_keys = self.k_norm(
            noise_keys.reshape(batch, block_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        noise_values = self.v_proj(hidden_states).reshape(
            batch, block_len, self.n_kv_heads, -1,
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            if isinstance(cache, FullContextDraftKVCache):
                cache_offset = int(cache.offset)
                query_offset = cache_offset + ctx_len
                queries = self.rope(queries, offset=query_offset)
                context_keys, context_values, context_positions = self._rope_context_segments(
                    context_keys,
                    context_values,
                    cache_offset=cache_offset,
                    spans=context_spans,
                )
                noise_keys = self.rope(noise_keys, offset=query_offset)

                cache.append_context(
                    context_keys,
                    context_values,
                    ctx_len,
                    positions=context_positions,
                    advance_positions=ctx_len,
                )
                keys, values = cache.fetch_with_block(noise_keys, noise_values)
                mask = None
                if self.sliding_window is not None:
                    noise_positions = mx.arange(
                        query_offset,
                        query_offset + block_len,
                        dtype=mx.int32,
                    )
                    cached_positions = cache.position_indices()
                    key_positions = (
                        noise_positions
                        if cached_positions is None
                        else mx.concatenate([cached_positions, noise_positions], axis=0)
                    )
                    mask = self._attention_mask(
                        block_len=block_len,
                        query_offset=query_offset,
                        key_len=int(keys.shape[-2]),
                        key_positions=key_positions,
                    )
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=None,
                    scale=self.scale,
                    mask=mask,
                )
            elif isinstance(cache, ContextOnlyDraftKVCache):
                cache_offset = int(cache.offset)
                query_offset = cache_offset + ctx_len
                queries = self.rope(queries, offset=query_offset)
                context_keys, context_values, context_positions = self._rope_context_segments(
                    context_keys,
                    context_values,
                    cache_offset=cache_offset,
                    spans=context_spans,
                )
                noise_keys = self.rope(noise_keys, offset=query_offset)

                cache.append_context(
                    context_keys,
                    context_values,
                    ctx_len,
                    positions=context_positions,
                    advance_positions=ctx_len,
                )
                cached_keys, cached_values = cache.fetch()
                keys = mx.concatenate([cached_keys, noise_keys], axis=-2)
                values = mx.concatenate([cached_values, noise_values], axis=-2)
                cached_positions = cache.position_indices()
                noise_positions = mx.arange(
                    query_offset,
                    query_offset + block_len,
                    dtype=mx.int32,
                )
                key_positions = mx.concatenate([cached_positions, noise_positions], axis=0)
                mask = self._attention_mask(
                    block_len=block_len,
                    query_offset=query_offset,
                    key_len=int(keys.shape[-2]),
                    key_positions=key_positions,
                )
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=None,
                    scale=self.scale,
                    mask=mask,
                )
            else:
                cache_offset = int(getattr(cache, "offset", 0) or 0)
                query_offset = cache_offset + ctx_len
                queries = self.rope(queries, offset=query_offset)
                context_keys = self.rope(context_keys, offset=cache_offset)
                noise_keys = self.rope(noise_keys, offset=query_offset)

                keys = mx.concatenate([context_keys, noise_keys], axis=-2)
                values = mx.concatenate([context_values, noise_values], axis=-2)
                keys, values = cache.update_and_fetch(keys, values)
                mask = self._attention_mask(
                    block_len=block_len,
                    query_offset=query_offset,
                    key_len=int(keys.shape[-2]),
                )
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=self.scale,
                    mask=mask,
                )
        else:
            queries = self.rope(queries, offset=ctx_len)
            context_keys = self.rope(context_keys, offset=0)
            noise_keys = self.rope(noise_keys, offset=ctx_len)
            if self.sliding_window is None and hasattr(mx.fast, "dflash_cross_attention"):
                output = mx.fast.dflash_cross_attention(
                    queries,
                    context_keys,
                    context_values,
                    noise_keys,
                    noise_values,
                    scale=self.scale,
                )
            else:
                keys = mx.concatenate([context_keys, noise_keys], axis=-2)
                values = mx.concatenate([context_values, noise_values], axis=-2)
                mask = self._attention_mask(
                    block_len=block_len,
                    query_offset=ctx_len,
                    key_len=int(keys.shape[-2]),
                )
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=None,
                    scale=self.scale,
                    mask=mask,
                )

        output = output.transpose(0, 2, 1, 3).reshape(batch, block_len, -1)
        return self.o_proj(output)

class DFlashDecoderLayer(nn.Module):
    def __init__(self, args: DFlashDraftModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = DFlashAttention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            target_hidden=target_hidden,
            cache=cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def advance_projected_context_cache(
        self,
        *,
        target_hidden: mx.array,
        cache: Any,
    ) -> None:
        self.self_attn.append_projected_context_cache(
            target_hidden=target_hidden,
            cache=cache,
        )

class DFlashDraftModel(nn.Module):
    def __init__(self, args: DFlashDraftModelArgs):
        super().__init__()
        self.args = args
        self.model_type = "dflash_qwen3"
        self.layers = [
            DFlashDecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        target_layer_ids = list((args.dflash_config or {}).get("target_layer_ids") or ())
        self.target_layer_ids = target_layer_ids or build_target_layer_ids(
            args.num_target_layers,
            args.num_hidden_layers,
        )
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fc = nn.Linear(len(self.target_layer_ids) * args.hidden_size, args.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.block_size = int(args.block_size)
        self.mask_token_id = int((args.dflash_config or {}).get("mask_token_id", 0) or 0)
        self.embed_scale = 1.0

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        text_model = target_ops.text_model(target_model)
        self.embed_scale = getattr(text_model, "embed_scale", 1.0)

    def project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        return self.hidden_norm(self.fc(target_hidden))

    def forward_projected_context(
        self,
        *,
        noise_embedding: mx.array,
        draft_context: mx.array,
        cache: Optional[list[Any]] = None,
    ) -> mx.array:
        hidden_states = noise_embedding * self.embed_scale

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden_states = layer(
                hidden_states,
                target_hidden=draft_context,
                cache=layer_cache,
            )
        return self.norm(hidden_states)

    def advance_projected_context_cache(
        self,
        *,
        draft_context: mx.array,
        cache: list[Any],
    ) -> None:
        if cache is None:
            raise ValueError("draft context cache is required")
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            layer.advance_projected_context_cache(
                target_hidden=draft_context,
                cache=layer_cache,
            )

    def __call__(
        self,
        *,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: Optional[list[Any]] = None,
    ) -> mx.array:
        return self.forward_projected_context(
            noise_embedding=noise_embedding,
            draft_context=self.project_target_hidden(target_hidden),
            cache=cache,
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return weights

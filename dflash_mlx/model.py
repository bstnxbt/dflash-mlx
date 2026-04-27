# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
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
    dflash_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "DFlashDraftModelArgs":
        data = dict(params)
        data["layer_types"] = tuple(data.get("layer_types") or ())
        data["dflash_config"] = dict(data.get("dflash_config") or {})
        return cls(
            **{key: value for key, value in data.items() if key in cls.__annotations__}
        )

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
        target_hidden: mx.array,
        cache: Any,
    ) -> tuple[mx.array, list[tuple[int, int]]]:
        if not isinstance(cache, ContextOnlyDraftKVCache):
            return target_hidden, [(0, int(target_hidden.shape[1]))]
        spans = cache.context_spans_to_append(int(target_hidden.shape[1]))
        if not spans:
            return target_hidden[:, :0, :], []
        if len(spans) == 1:
            start, end = spans[0]
            return target_hidden[:, start:end, :], spans
        return mx.concatenate([target_hidden[:, start:end, :] for start, end in spans], axis=1), spans

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
            if isinstance(cache, ContextOnlyDraftKVCache):
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

    def _project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        return self.hidden_norm(self.fc(target_hidden))

    def __call__(
        self,
        *,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: Optional[list[Any]] = None,
    ) -> mx.array:
        hidden_states = noise_embedding
        projected_hidden = self._project_target_hidden(target_hidden)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden_states = layer(
                hidden_states,
                target_hidden=projected_hidden,
                cache=layer_cache,
            )
        return self.norm(hidden_states)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return weights

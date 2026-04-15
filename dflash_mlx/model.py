# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
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


def extract_context_feature(
    hidden_states: list[mx.array],
    layer_ids: list[int],
) -> mx.array:
    selected = [hidden_states[layer_id + 1] for layer_id in layer_ids]
    return mx.concatenate(selected, axis=-1)


class ContextOnlyDraftKVCache:
    def __init__(self, sink_size: int = 64, window_size: int = 1024):
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)
        self.keys = None
        self.values = None
        self.offset = 0

    def append_context(
        self,
        context_keys: mx.array,
        context_values: mx.array,
        num_positions: int,
    ) -> None:
        if context_keys is None or context_values is None or int(num_positions) <= 0:
            return
        if self.keys is None:
            self.keys = context_keys
            self.values = context_values
        else:
            self.keys = mx.concatenate([self.keys, context_keys], axis=2)
            self.values = mx.concatenate([self.values, context_values], axis=2)
        self.offset += int(num_positions)
        self._apply_window()

    def _apply_window(self) -> None:
        if self.keys is None or self.values is None:
            return
        cache_len = int(self.keys.shape[2])
        max_len = self.sink_size + self.window_size
        if cache_len <= max_len:
            return
        sink_k = self.keys[:, :, : self.sink_size, :]
        sink_v = self.values[:, :, : self.sink_size, :]
        window_k = self.keys[:, :, -self.window_size :, :]
        window_v = self.values[:, :, -self.window_size :, :]
        self.keys = mx.concatenate([sink_k, window_k], axis=2)
        self.values = mx.concatenate([sink_v, window_v], axis=2)

    def fetch(self) -> tuple[Optional[mx.array], Optional[mx.array]]:
        return self.keys, self.values

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
    def __init__(self, args: DFlashDraftModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
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

        context_keys = self.k_proj(target_hidden)
        context_keys = self.k_norm(
            context_keys.reshape(batch, ctx_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        context_values = self.v_proj(target_hidden).reshape(
            batch,
            ctx_len,
            self.n_kv_heads,
            -1,
        ).transpose(0, 2, 1, 3)

        noise_keys = self.k_proj(hidden_states)
        noise_keys = self.k_norm(
            noise_keys.reshape(batch, block_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        noise_values = self.v_proj(hidden_states).reshape(
            batch,
            block_len,
            self.n_kv_heads,
            -1,
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            if isinstance(cache, ContextOnlyDraftKVCache):
                cache_offset = int(cache.offset)
                query_offset = cache_offset + ctx_len
                queries = self.rope(queries, offset=query_offset)
                context_keys = self.rope(context_keys, offset=cache_offset)
                noise_keys = self.rope(noise_keys, offset=query_offset)

                cache.append_context(context_keys, context_values, ctx_len)
                cached_keys, cached_values = cache.fetch()
                keys = mx.concatenate([cached_keys, noise_keys], axis=-2)
                values = mx.concatenate([cached_values, noise_values], axis=-2)
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=None,
                    scale=self.scale,
                    mask=None,
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
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=self.scale,
                    mask=None,
                )
        else:
            queries = self.rope(queries, offset=ctx_len)
            context_keys = self.rope(context_keys, offset=0)
            noise_keys = self.rope(noise_keys, offset=ctx_len)
            if hasattr(mx.fast, "dflash_cross_attention"):
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
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=None,
                    scale=self.scale,
                    mask=None,
                )

        output = output.transpose(0, 2, 1, 3).reshape(batch, block_len, -1)
        return self.o_proj(output)


class DFlashDecoderLayer(nn.Module):
    def __init__(self, args: DFlashDraftModelArgs):
        super().__init__()
        self.self_attn = DFlashAttention(args)
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
        if hasattr(mx.fast, "dflash_gated_mlp") and not isinstance(
            self.mlp.gate_proj, nn.QuantizedLinear
        ):
            hidden_states = mx.fast.dflash_gated_mlp(
                hidden_states,
                self.mlp.gate_proj.weight,
                self.mlp.up_proj.weight,
                self.mlp.down_proj.weight,
            )
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, args: DFlashDraftModelArgs):
        super().__init__()
        self.args = args
        self.model_type = "dflash_qwen3"
        self.layers = [DFlashDecoderLayer(args) for _ in range(args.num_hidden_layers)]
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

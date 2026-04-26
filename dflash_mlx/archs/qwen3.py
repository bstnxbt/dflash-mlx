# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""
Qwen3 DFlash architecture implementation.

This module provides the DFlash draft model implementation for Qwen3-based models.
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import MLP as Qwen3MLP
from mlx_lm.models.rope_utils import initialize_rope

from dflash_mlx.kernels import make_qwen3_no_cache_attn
from dflash_mlx.archs.base import (
    DFlashArgs,
    DFlashAttention,
    DFlashCache,
    DFlashDecoderLayer,
    DFlashModel,
    DFlashMLP,
    DFlashNorm,
    DFlashRope,
    ArchitectureSpec,
    build_target_layer_ids,
    register_architecture,
)


# =============================================================================
# Qwen3-specific Norm (RMSNorm with optional Qwen3 specifics)
# =============================================================================


class Qwen3DFlashNorm(nn.RMSNorm):
    """Qwen3-specific RMSNorm implementation."""

    pass


class Qwen3DFlashRope:
    """Qwen3-specific RoPE implementation."""

    def __init__(
        self,
        head_dim: int,
        base: float,
        max_position_embeddings: int,
        scaling_config: Optional[dict[str, Any]] = None,
    ):
        self.rope = initialize_rope(
            head_dim,
            base=base,
            traditional=False,
            scaling_config=scaling_config,
            max_position_embeddings=max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        offset: int = 0,
    ) -> mx.array:
        return self.rope(x, offset=offset)


# =============================================================================
# Qwen3 Attention (with Q/K normalization)
# =============================================================================


class Qwen3DFlashAttention(nn.Module, DFlashAttention):
    """
    Qwen3-specific DFlash cross-attention layer.

    This attention implementation includes Q/K normalization which is
    specific to Qwen3 architecture.
    """

    def __init__(self, args: DFlashArgs):
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

        # Qwen3-specific Q/K normalization
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = Qwen3DFlashRope(
            self.head_dim,
            base=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            scaling_config=args.rope_scaling,
        )

        self._compiled_no_cache_attn = make_qwen3_no_cache_attn(
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.q_norm,
            self.k_norm,
            self.rope,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            scale=self.scale,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[DFlashCache] = None,
    ) -> mx.array:
        batch, block_len, _ = hidden_states.shape
        ctx_len = int(target_hidden.shape[1])

        # Phew-optimized path: fully compiled, ~1.2-1.35x vs unfused baseline.
        # Fuses projections, RMSNorm, RoPE, GQA expand, and SDPA in one trace.
        if cache is None and not hasattr(mx.fast, "dflash_cross_attention"):
            return self._compiled_no_cache_attn(hidden_states, target_hidden, ctx_len)

        # Project and reshape queries
        queries = self.q_proj(hidden_states)
        queries = self.q_norm(
            queries.reshape(batch, block_len, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)

        # Fuse context and noise projections: 2 matmuls instead of 4
        kv_states = mx.concatenate([target_hidden, hidden_states], axis=1)
        all_keys = self.k_norm(
            self.k_proj(kv_states).reshape(batch, ctx_len + block_len, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        all_values = self.v_proj(kv_states).reshape(
            batch, ctx_len + block_len, self.n_kv_heads, -1
        ).transpose(0, 2, 1, 3)

        context_keys = all_keys[:, :, :ctx_len, :]
        context_values = all_values[:, :, :ctx_len, :]
        noise_keys = all_keys[:, :, ctx_len:, :]
        noise_values = all_values[:, :, ctx_len:, :]

        if cache is not None:
            if isinstance(cache, Qwen3ContextOnlyCache):
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
                # Use standard cache update
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
            # No cache - use standard attention path
            queries = self.rope(queries, offset=ctx_len)
            context_keys = self.rope(context_keys, offset=0)
            noise_keys = self.rope(noise_keys, offset=ctx_len)

            # Try to use optimized DFlash kernel if available
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


class Qwen3ContextOnlyCache:
    """Qwen3-specific context-only KV cache with sliding window."""

    def __init__(self, sink_size: int = 64, window_size: int = 1024):
        self.sink_size = int(sink_size)
        self.window_size = int(window_size)
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0

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


# =============================================================================
# Qwen3 MLP (using mlx_lm's Qwen3 MLP)
# =============================================================================


class Qwen3DFlashMLP(Qwen3MLP, DFlashMLP):
    """Qwen3-specific MLP using the mlx_lm Qwen3 MLP implementation."""

    pass


# =============================================================================
# Qwen3 Decoder Layer
# =============================================================================


class Qwen3DFlashDecoderLayer(nn.Module, DFlashDecoderLayer):
    """Qwen3-specific decoder layer."""

    def __init__(self, args: DFlashArgs):
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(args)
        self.mlp = Qwen3DFlashMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = Qwen3DFlashNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = Qwen3DFlashNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[DFlashCache] = None,
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


# =============================================================================
# Qwen3 Full Model
# =============================================================================


class Qwen3DFlashModel(nn.Module, DFlashModel):
    """
    Qwen3-based DFlash draft model.

    This model takes noise token embeddings (from the target model's embed_tokens)
    and target hidden states, and produces draft logits for block-diffusion
    speculative decoding.
    """

    def __init__(self, args: DFlashArgs):
        super().__init__()
        self.args = args
        self.model_type = "dflash_qwen3"

        # Create decoder layers
        self.layers = [
            Qwen3DFlashDecoderLayer(args) for _ in range(args.num_hidden_layers)
        ]

        # Get target layer IDs from config or build defaults
        target_layer_ids = list(args.dflash_config.get("target_layer_ids") or [])
        self.target_layer_ids = target_layer_ids or build_target_layer_ids(
            args.num_target_layers,
            args.num_hidden_layers,
        )

        # Output projection
        self.norm = Qwen3DFlashNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Project concatenated target hidden states
        self.fc = nn.Linear(
            len(self.target_layer_ids) * args.hidden_size,
            args.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3DFlashNorm(args.hidden_size, eps=args.rms_norm_eps)

        self.block_size = int(args.block_size)
        self.mask_token_id = int(args.dflash_config.get("mask_token_id", 0) or 0)

    def _project_target_hidden(self, target_hidden: mx.array) -> mx.array:
        """Project and normalize target hidden states."""
        return self.hidden_norm(self.fc(target_hidden))

    def __call__(
        self,
        *,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: Optional[list[Optional[DFlashCache]]] = None,
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
        """Sanitize model weights after loading."""
        return weights


# =============================================================================
# Register Qwen3 Architecture
# =============================================================================


qwen3_spec = ArchitectureSpec(
    name="qwen3",
    model_class=Qwen3DFlashModel,
    attention_class=Qwen3DFlashAttention,
    mlp_class=Qwen3DFlashMLP,
    norm_class=Qwen3DFlashNorm,
    rope_class=Qwen3DFlashRope,
    cache_class=Qwen3ContextOnlyCache,
    model_type_patterns=("qwen3", "qwen2.5", "qwen2", "kimi", "qwen3_moe"),
)

register_architecture(qwen3_spec)
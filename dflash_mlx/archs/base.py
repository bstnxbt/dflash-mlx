# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""
Base protocols and abstractions for DFlash architecture system.

This module defines the interfaces that each architecture must implement,
enabling a pluggable system for supporting multiple model architectures.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional, Protocol, Type, TypeVar, runtime_checkable

import mlx.core as mx
import mlx.nn as nn


# =============================================================================
# Protocol Definitions (interfaces for architecture implementations)
# =============================================================================


@runtime_checkable
class DFlashNorm(Protocol):
    """Normalization layer protocol."""

    def __call__(self, x: mx.array) -> mx.array:
        """Apply normalization."""
        ...


@runtime_checkable
class DFlashRope(Protocol):
    """Rotary Positional Embedding protocol."""

    def __call__(
        self,
        x: mx.array,
        *,
        offset: int = 0,
    ) -> mx.array:
        """Apply RoPE with given offset."""
        ...


@runtime_checkable
class DFlashAttention(Protocol):
    """Attention layer protocol for DFlash cross-attention."""

    n_heads: int
    n_kv_heads: int
    head_dim: int
    scale: float

    def __init__(
        self,
        args: DFlashArgs,
    ) -> None:
        """Initialize attention with model arguments."""
        ...

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[DFlashCache] = None,
    ) -> mx.array:
        """Forward pass with cross-attention to target hidden states."""
        ...


@runtime_checkable
class DFlashMLP(Protocol):
    """MLP/feed-forward network protocol."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        """Initialize MLP."""
        ...

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        ...


@runtime_checkable
class DFlashCache(Protocol):
    """KV cache protocol for attention layers."""

    offset: int

    def append_context(
        self,
        context_keys: mx.array,
        context_values: mx.array,
        num_positions: int,
    ) -> None:
        """Append context KV to cache."""
        ...

    def fetch(self) -> tuple[Optional[mx.array], Optional[mx.array]]:
        """Fetch cached keys and values."""
        ...

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Update cache with new keys/values and fetch."""
        ...

    def cache_length(self) -> int:
        """Get current cache length."""
        ...


@runtime_checkable
class DFlashDecoderLayer(Protocol):
    """Single decoder layer protocol."""

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[DFlashCache] = None,
    ) -> mx.array:
        """Forward pass through decoder layer."""
        ...


@runtime_checkable
class DFlashModel(Protocol):
    """Full DFlash draft model protocol."""

    model_type: str
    target_layer_ids: list[int]
    block_size: int
    mask_token_id: int
    args: DFlashArgs

    def __call__(
        self,
        *,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: Optional[list[Optional[DFlashCache]]] = None,
    ) -> mx.array:
        """Forward pass through the full model."""
        ...

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Sanitize model weights after loading."""
        ...


# =============================================================================
# Model Arguments Dataclass
# =============================================================================


@dataclass
class DFlashArgs:
    """Configuration arguments for DFlash draft models."""

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
    dflash_config: dict[str, Any] = field(default_factory=dict)

    # Architecture-specific attributes (set by architecture implementation)
    architecture: Optional[str] = None

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "DFlashArgs":
        """Create args from config dictionary.
        
        Handles both standard DFlash config and Gemma-style speculator config
        (which uses transformer_layer_config to embed the actual model config).
        """
        data = dict(params)
        data["layer_types"] = tuple(data.get("layer_types") or ())
        data["dflash_config"] = dict(data.get("dflash_config") or {})

        # Handle Gemma-style config with embedded transformer_layer_config
        transformer_config = data.get("transformer_layer_config", {})
        if transformer_config:
            # This is a Gemma-style speculator config
            # Extract model params from the embedded transformer config
            for key in [
                "hidden_size", "num_hidden_layers", "intermediate_size",
                "num_attention_heads", "rms_norm_eps", "vocab_size",
                "num_key_value_heads", "max_position_embeddings",
                "rope_theta", "head_dim", "tie_word_embeddings",
                "attention_bias", "attention_dropout",
            ]:
                if key in transformer_config and key not in data:
                    data[key] = transformer_config[key]
            
            # Get rope parameters
            rope_params = transformer_config.get("rope_parameters", {})
            if rope_params and "rope_theta" not in data:
                data["rope_theta"] = rope_params.get("rope_theta", 1e6)
            
            # Set model type from transformer config
            if "model_type" not in data:
                data["model_type"] = transformer_config.get("model_type", "llama")
            
            # Extract dflash_config from top-level if not present
            if "dflash_config" not in data or not data["dflash_config"]:
                data["dflash_config"] = {
                    "target_layer_ids": data.get("aux_hidden_state_layer_ids"),
                    "mask_token_id": data.get("mask_token_id"),
                }
            
            # Set num_target_layers
            if "num_target_layers" not in data:
                data["num_target_layers"] = transformer_config.get("num_hidden_layers", 62)

        # Determine architecture from model_type or config
        model_type = data.get("model_type", "")
        arch = _infer_architecture(model_type, data)
        data["architecture"] = arch

        return cls(
            **{key: value for key, value in data.items() if key in cls.__annotations__}
        )


def _infer_architecture(model_type: str, config: dict[str, Any]) -> str:
    """Infer the architecture name from model type and config."""
    model_type_lower = model_type.lower()

    # Check for Llama-based models (Gemma, Llama, etc.)
    if "llama" in model_type_lower or "gemma" in model_type_lower:
        return "llama"

    # Check for Qwen models
    if "qwen" in model_type_lower:
        return "qwen3"

    # Check transformer_layer_config for Llama (Gemma spec format)
    transformer_config = config.get("transformer_layer_config", {})
    if transformer_config:
        inner_type = transformer_config.get("model_type", "").lower()
        if "llama" in inner_type or "gemma" in inner_type:
            return "llama"

    # Default to qwen3 for backward compatibility
    return "qwen3"


# =============================================================================
# Architecture Registry
# =============================================================================


@dataclass
class ArchitectureSpec:
    """Specification for a DFlash architecture implementation."""

    name: str
    model_class: Type[DFlashModel]
    attention_class: Type[DFlashAttention]
    mlp_class: Type[DFlashMLP]
    norm_class: Optional[Type[DFlashNorm]] = None
    rope_class: Optional[Type[DFlashRope]] = None
    cache_class: Optional[Type[DFlashCache]] = None
    # Patterns that identify this architecture in model type strings
    model_type_patterns: tuple[str, ...] = ()


class ArchitectureRegistry:
    """Registry for DFlash architecture implementations."""

    _architectures: ClassVar[dict[str, ArchitectureSpec]] = {}
    _fallback: ClassVar[Optional[Type[DFlashModel]]] = None

    @classmethod
    def register(cls, spec: ArchitectureSpec) -> None:
        """Register an architecture implementation."""
        cls._architectures[spec.name] = spec
        for pattern in spec.model_type_patterns:
            cls._architectures[pattern] = spec

    @classmethod
    def get(cls, name: str) -> Optional[ArchitectureSpec]:
        """Get architecture by name."""
        return cls._architectures.get(name)

    @classmethod
    def get_for_model_type(cls, model_type: str) -> ArchitectureSpec:
        """Get the appropriate architecture for a model type."""
        model_type_lower = model_type.lower()

        # Direct match
        if model_type_lower in cls._architectures:
            return cls._architectures[model_type_lower]

        # Pattern matching
        for name, spec in cls._architectures.items():
            if name in model_type_lower:
                return spec

        # Fallback to qwen3
        if "qwen3" in cls._architectures:
            return cls._architectures["qwen3"]

        raise ValueError(f"No architecture found for model type: {model_type}")


def register_architecture(spec: ArchitectureSpec) -> None:
    """Register a DFlash architecture implementation."""
    ArchitectureRegistry.register(spec)


def get_architecture_for_model_type(model_type: str) -> ArchitectureSpec:
    """Get the appropriate architecture for a model type."""
    return ArchitectureRegistry.get_for_model_type(model_type)


def list_supported_architectures() -> list[str]:
    """List all supported architecture names."""
    return list(set(ArchitectureRegistry._architectures.keys()))


# =============================================================================
# Model Factory
# =============================================================================


def create_dflash_model(config: dict[str, Any]) -> DFlashModel:
    """
    Create a DFlash model from configuration.

    Args:
        config: Model configuration dictionary (from config.json)

    Returns:
        Instance of the appropriate DFlash model for the architecture
    """
    args = DFlashArgs.from_dict(config)
    arch_spec = get_architecture_for_model_type(args.model_type)
    return arch_spec.model_class(args)


# Backward compatibility - export the original class names
def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Build default target layer IDs for draft model."""
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
    """Extract and concatenate hidden states at specified layer IDs."""
    selected = [hidden_states[layer_id + 1] for layer_id in layer_ids]
    return mx.concatenate(selected, axis=-1)
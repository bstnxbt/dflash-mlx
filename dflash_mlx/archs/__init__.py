# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""
DFlash architecture modular system.

This module provides a pluggable architecture system supporting multiple
model architectures (Qwen3, Llama/Gemma, etc.) with custom attention,
MLP, normalization, and RoPE implementations.
"""

from dflash_mlx.archs.base import (
    DFlashAttention,
    DFlashArgs,
    DFlashCache,
    DFlashDecoderLayer,
    DFlashMLP,
    DFlashModel,
    DFlashNorm,
    DFlashRope,
    create_dflash_model,
    extract_context_feature,
    get_architecture_for_model_type,
    list_supported_architectures,
    register_architecture,
)
from dflash_mlx.archs.qwen3 import Qwen3DFlashModel, Qwen3DFlashAttention, Qwen3DFlashMLP
from dflash_mlx.archs.llama import LlamaDFlashModel, LlamaDFlashAttention, LlamaDFlashMLP

__all__ = [
    # Base classes
    "DFlashArgs",
    "DFlashModel",
    "DFlashAttention",
    "DFlashMLP",
    "DFlashNorm",
    "DFlashRope",
    "DFlashCache",
    "DFlashDecoderLayer",
    # Factory functions
    "create_dflash_model",
    "get_architecture_for_model_type",
    "list_supported_architectures",
    "register_architecture",
    # Architecture implementations
    "Qwen3DFlashModel",
    "Qwen3DFlashAttention",
    "Qwen3DFlashMLP",
    "LlamaDFlashModel",
    "LlamaDFlashAttention",
    "LlamaDFlashMLP",
]
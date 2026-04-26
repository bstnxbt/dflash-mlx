# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""
DFlash Model Module

This module provides backward compatibility by re-exporting from the
new architecture system in dflash_mlx.archs.

For new code, prefer importing directly from dflash_mlx.archs:
    from dflash_mlx.archs import create_dflash_model, DFlashArgs, DFlashModel
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

# Re-export everything from archs for backward compatibility
from dflash_mlx.archs.base import (
    DFlashArgs,
    DFlashModel,
    build_target_layer_ids,
    extract_context_feature,
)

# Keep old class names as aliases for backward compatibility
# These map to the Qwen3 implementation (the original/default)
from dflash_mlx.archs.qwen3 import (
    Qwen3DFlashModel as DFlashDraftModel,
    Qwen3DFlashAttention as DFlashAttention,
    Qwen3DFlashMLP as MLP,
    Qwen3ContextOnlyCache as ContextOnlyDraftKVCache,
)

# Keep the old DFlashDraftModelArgs as alias for DFlashArgs
DFlashDraftModelArgs = DFlashArgs


class RecurrentRollbackCache:
    """Legacy alias - use architecture-specific cache classes instead."""

    def __init__(self, sink_size: int = 64, window_size: int = 1024):
        from dflash_mlx.archs.qwen3 import Qwen3ContextOnlyCache
        self._cache = Qwen3ContextOnlyCache(sink_size, window_size)

    @property
    def offset(self) -> int:
        return self._cache.offset

    @property
    def keys(self) -> Optional[mx.array]:
        return self._cache.keys

    @property
    def values(self) -> Optional[mx.array]:
        return self._cache.values

    def append_context(
        self,
        context_keys: mx.array,
        context_values: mx.array,
        num_positions: int,
    ) -> None:
        self._cache.append_context(context_keys, context_values, num_positions)

    def fetch(self) -> tuple[Optional[mx.array], Optional[mx.array]]:
        return self._cache.fetch()

    def cache_length(self) -> int:
        return self._cache.cache_length()


__all__ = [
    # Main classes
    "DFlashDraftModel",
    "DFlashDraftModelArgs",
    "DFlashArgs",
    "DFlashModel",
    "DFlashAttention",
    "MLP",
    # Cache
    "ContextOnlyDraftKVCache",
    "RecurrentRollbackCache",
    # Utility functions
    "build_target_layer_ids",
    "extract_context_feature",
]
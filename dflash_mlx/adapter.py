# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any

from dflash_mlx.engine import FullAttentionEngine, HybridGDNEngine


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


def detect_engine(target_model: Any) -> FullAttentionEngine | HybridGDNEngine:
    inner = _target_text_model(target_model)
    if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
        return HybridGDNEngine()
    has_linear = any(
        hasattr(layer, "linear_attn") or hasattr(layer, "is_linear")
        for layer in inner.layers
    )
    return HybridGDNEngine() if has_linear else FullAttentionEngine()

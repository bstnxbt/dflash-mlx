# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any

from dflash_mlx.engine.target_verifier import _target_text_model
from dflash_mlx.verify_backend import FullAttentionEngine, HybridGDNEngine

def detect_engine(target_model: Any) -> FullAttentionEngine | HybridGDNEngine:
    inner = _target_text_model(target_model)
    if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
        return HybridGDNEngine()
    has_linear = any(
        hasattr(layer, "linear_attn") or hasattr(layer, "is_linear")
        for layer in inner.layers
    )
    return HybridGDNEngine() if has_linear else FullAttentionEngine()

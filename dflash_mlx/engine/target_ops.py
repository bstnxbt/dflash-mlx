# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Optional, Protocol

import mlx.core as mx

@dataclass(frozen=True)
class TargetCapabilities:
    supports_dflash: bool
    supports_recurrent_rollback: bool
    supports_kv_trim: bool
    supports_prefix_snapshot: bool
    supports_rotating_cache_snapshot: bool
    supports_shared_kv: bool
    supports_target_hidden_capture: bool
    supports_verify_linear: bool = False
    supports_full_context_draft_layers: bool = False
    supports_tree_verify: bool = False

class TargetOps(Protocol):
    backend_name: str

    def model_type(self, target_model: Any) -> str: ...

    def supports_model(self, target_model: Any) -> bool: ...

    def family(self, target_model: Any) -> str: ...

    def capabilities_for(self, target_model: Any) -> TargetCapabilities: ...

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool: ...

    def text_model(self, target_model: Any) -> Any: ...

    def embed_tokens(self, target_model: Any) -> Any: ...

    def logits_from_hidden(self, target_model: Any, hidden_states: mx.array) -> mx.array: ...

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
    ) -> list[Any]: ...

    def install_speculative_hooks(self, target_model: Any) -> None: ...

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]: ...

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]: ...

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]: ...

    def restore_after_tree_acceptance(
        self,
        cache_entries: list[Any],
        *,
        accepted_tree_indices: list[int],
    ) -> int: ...

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array],
        target_layer_ids: list[int],
    ) -> mx.array: ...

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None: ...

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int: ...

    def cleanup_generation_caches(
        self,
        target_cache: list[Any],
        draft_cache: list[Any],
    ) -> None: ...

TARGET_BACKENDS = [
    "dflash_mlx.engine.target_qwen_gdn:QwenGdnTargetOps",
    "dflash_mlx.engine.target_gemma4:Gemma4TargetOps",
    "dflash_mlx.engine.target_lfm2:Lfm2TargetOps",
]

def _load_backend_class(path: str) -> type[TargetOps]:
    module_name, class_name = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)

def _backend_instances() -> list[TargetOps]:
    return [_load_backend_class(path)() for path in TARGET_BACKENDS]

def resolve_target_ops(target_model: Any) -> TargetOps:
    backends = _backend_instances()
    for backend in backends:
        if backend.supports_model(target_model):
            return backend
    model_type = next(
        (backend.model_type(target_model) for backend in backends if backend.model_type(target_model)),
        "",
    )
    model_class = type(target_model).__name__
    supported = ", ".join(backend.backend_name for backend in backends)
    raise NotImplementedError(
        "Unsupported target architecture for DFlash target ops: "
        f"model_type={model_type or 'unknown'}, "
        f"model_class={model_class}, "
        f"supported target backends: {supported}"
    )

def bind_draft_to_target(
    draft_model: Any,
    target_model: Any,
    *,
    target_ops: TargetOps,
) -> TargetOps:
    bind_target = getattr(draft_model, "bind_target_model", None)
    if bind_target is not None:
        bind_target(target_model, target_ops=target_ops)
    return target_ops

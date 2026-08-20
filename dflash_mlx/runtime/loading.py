# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load, load_model

from dflash_mlx.engine.target_ops import TargetOps, resolve_target_ops
from dflash_mlx.internal_debug import (
    verify_linear_override as _debug_verify_linear_override,
    verify_qmm_enabled as _debug_verify_qmm_enabled,
)
from dflash_mlx.model import (
    DFlashDraftModel,
    DFlashDraftModelArgs,
)
from dflash_mlx.runtime import VerifyConfig
from dflash_mlx.runtime.chip_detect import ChipProfile, detect_chip


@dataclass(frozen=True)
class DraftQuantSpec:
    weight_bits: int
    group_size: int
    act_bits: int


@dataclass(frozen=True)
class LoadedTargetBundle:
    model: Any
    tokenizer: Any
    meta: dict[str, Any]
    target_ops: TargetOps


def resolve_model_ref(model_ref: str | Path | None, *, kind: str) -> str:
    if model_ref:
        candidate = Path(model_ref).expanduser()
        return str(candidate if candidate.exists() else model_ref)
    raise ValueError(f"{kind} model reference is required")


def _get_dflash_model_classes(config: dict[str, Any]):
    from dflash_mlx.draft.checkpoint import get_draft_model_classes

    return get_draft_model_classes(config)


def _resolve_local_model_path(model_ref: str | Path) -> Path:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return candidate
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"Model path does not exist and huggingface_hub is unavailable: {model_ref}"
        ) from exc

    snapshot_path = snapshot_download(
        repo_id=str(model_ref),
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "tokenizer*"],
    )
    return Path(snapshot_path)


_DRAFT_QUANT_RE = re.compile(
    r"^w(?P<wb>2|4|8)"
    r"(?:a(?P<ab>16|32))?"
    r"(?::gs(?P<gs>32|64|128))?$",
    re.IGNORECASE,
)


def parse_draft_quant_spec(spec: str) -> DraftQuantSpec:
    m = _DRAFT_QUANT_RE.match(spec.strip())
    if not m:
        raise ValueError(
            f"Invalid draft quant spec {spec!r}. "
            "Expected format: w4, w8a16, w4a32:gs128, etc. "
            "Weight bits: 2, 4, 8. Activation bits: 16 (bfloat16) or 32 (float32). "
            "Group size: 32, 64, 128."
        )
    wb = int(m.group("wb"))
    ab = int(m.group("ab") or 16)
    gs = int(m.group("gs") or 64)
    return DraftQuantSpec(weight_bits=wb, group_size=gs, act_bits=ab)


def _resolve_draft_quant(draft_quant: str | None) -> DraftQuantSpec | None:
    spec = (draft_quant or "").strip()
    if not spec:
        return None
    return parse_draft_quant_spec(spec)


def resolve_draft_load_dtype(
    quant_spec: DraftQuantSpec | None,
    *,
    chip_profile: ChipProfile | None = None,
) -> Any | None:
    if quant_spec is None or quant_spec.act_bits == 32:
        return None
    profile = chip_profile or detect_chip()
    if profile.bf16_emulated:
        return mx.float16
    return None


def _dtype_name(dtype: Any | None) -> str | None:
    if dtype is None:
        return None
    return str(dtype).rsplit(".", 1)[-1]


def _draft_lm_head_weight_names(model_path: Path) -> list[str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid draft safetensors index JSON: {index_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid draft safetensors index payload: {index_path}")
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid draft safetensors index weight_map: {index_path}")
        return sorted(
            str(name)
            for name in weight_map
            if str(name).startswith("lm_head.")
        )
    weight_files = sorted(model_path.glob("model*.safetensors"))
    if not weight_files:
        return []
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ValueError(
            f"Cannot inspect draft safetensors weights for unsupported lm_head: {model_path}"
        ) from exc
    names: list[str] = []
    for path in weight_files:
        with safe_open(str(path), framework="mlx") as handle:
            names.extend(
                str(name)
                for name in handle.keys()
                if str(name).startswith("lm_head.")
            )
    return sorted(names)


def _raise_if_unsupported_draft_lm_head(model_path: Path, model_ref: str) -> None:
    lm_head_names = _draft_lm_head_weight_names(model_path)
    if not lm_head_names:
        return
    sample = ", ".join(lm_head_names[:3])
    if len(lm_head_names) > 3:
        sample += ", ..."
    raise ValueError(
        f"DFlash draft checkpoint '{model_ref}' contains draft-owned lm_head weights "
        f"({sample}), but this runtime projects draft hidden states through the target "
        "TargetOps.logits_from_hidden(...) path and does not load a draft lm_head yet."
    )


def load_target_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    quantize_kv_cache: bool = False,
    verify_config: VerifyConfig | None = None,
) -> LoadedTargetBundle:
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    model, tokenizer, config = load(resolved_ref, lazy=lazy, return_config=True)
    target_ops = resolve_target_ops(model)
    target_family = target_ops.family(model)
    target_capabilities = target_ops.capabilities_for(model)
    target_ops.install_speculative_hooks(model)
    meta = {
        "resolved_model_ref": resolved_ref,
        "config": config,
        "quantize_kv_cache": bool(quantize_kv_cache),
        "target_family": target_family,
    }
    verify_linear_enabled = (
        bool(target_capabilities.supports_verify_linear)
        and _verify_enabled_for(verify_config=verify_config)
    )
    meta["verify_linear_enabled"] = bool(verify_linear_enabled)
    meta["verify_mode"] = (verify_config.mode if verify_config is not None else "env")
    if verify_linear_enabled:
        from dflash_mlx.verify_linear import install_verify_linears
        n_swapped = install_verify_linears(
            model,
            enable_qmm=_verify_qmm_enabled(verify_config),
        )
        meta["verify_linear_swapped"] = n_swapped
    return LoadedTargetBundle(
        model=model,
        tokenizer=tokenizer,
        meta=meta,
        target_ops=target_ops,
    )


def _verify_enabled_for(
    *,
    verify_config: VerifyConfig | None = None,
) -> bool:
    if verify_config is not None:
        return verify_config.mode != "off"
    override = _debug_verify_linear_override()
    if override is not None:
        return override
    return True


def _verify_qmm_enabled(verify_config: VerifyConfig | None) -> bool:
    if verify_config is not None:
        return bool(verify_config.enable_qmm)
    return _debug_verify_qmm_enabled()


def load_draft_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    draft_quant: str | None = None,
):
    resolved_ref = resolve_model_ref(model_ref, kind="draft")
    model_path = _resolve_local_model_path(resolved_ref)
    _raise_if_unsupported_draft_lm_head(model_path, str(resolved_ref))
    model, config = load_model(
        model_path,
        lazy=lazy,
        get_model_classes=_get_dflash_model_classes,
    )
    quant_spec = _resolve_draft_quant(draft_quant)
    if quant_spec is not None:
        nn.quantize(model, bits=quant_spec.weight_bits, group_size=quant_spec.group_size)
        draft_load_dtype = resolve_draft_load_dtype(quant_spec)
        if draft_load_dtype is not None:
            _cast_floating_model(model, draft_load_dtype)
        if quant_spec.act_bits == 32:

            def _cast_to_f32(x: mx.array) -> mx.array:
                if mx.issubdtype(x.dtype, mx.floating):
                    return x.astype(mx.float32)
                return x

            model.apply(_cast_to_f32)
            draft_load_dtype = mx.float32
        if quant_spec.weight_bits in (4, 8):
            from dflash_mlx.verify_linear import (
                install_verify_linears,
                prewarm_verify_kernels,
            )
            install_verify_linears(model, enable_qmm=True)
            prewarm_verify_kernels(
                model,
                input_dtype=draft_load_dtype or mx.bfloat16,
            )
    else:
        draft_load_dtype = None
    return model, {
        "resolved_model_ref": str(model_ref) if model_ref is not None else str(resolved_ref),
        "config": config,
        "draft_load_dtype": _dtype_name(draft_load_dtype),
        "draft_load_dtype_source": (
            "old_apple_bf16_emulation"
            if draft_load_dtype is not None and draft_load_dtype == mx.float16
            else None
        ),
        "draft_quant": (
            {
                "weight_bits": quant_spec.weight_bits,
                "group_size": quant_spec.group_size,
                "act_bits": quant_spec.act_bits,
            }
            if quant_spec is not None
            else None
        ),
    }


def _cast_floating_model(model: Any, dtype: Any) -> None:
    def _cast(x: mx.array) -> mx.array:
        if mx.issubdtype(x.dtype, mx.floating) and x.dtype != dtype:
            return x.astype(dtype)
        return x

    model.apply(_cast)

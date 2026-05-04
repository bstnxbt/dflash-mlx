# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load, load_model

from dflash_mlx.internal_debug import (
    verify_linear_override as _debug_verify_linear_override,
    verify_qmm_enabled as _debug_verify_qmm_enabled,
)
from dflash_mlx.engine.config import (
    _effective_draft_window_size,
)
from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.model import (
    DFlashDraftModel,
    DFlashDraftModelArgs,
)

def resolve_model_ref(model_ref: str | Path | None, *, kind: str) -> str:
    if model_ref:
        candidate = Path(model_ref).expanduser()
        return str(candidate if candidate.exists() else model_ref)
    raise ValueError(f"{kind} model reference is required")

def get_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_token_ids = list(getattr(tokenizer, "eos_token_ids", None) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in eos_token_ids:
        eos_token_ids.append(int(eos_token_id))
    return eos_token_ids

def default_split_sdpa_enabled(model_ref: str | Path | None) -> bool:
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    return False

def _get_dflash_model_classes(config: dict[str, Any]):
    return DFlashDraftModel, DFlashDraftModelArgs

def _resolve_local_model_path(model_ref: str | Path) -> Path:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return candidate
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise FileNotFoundError(f"Model path does not exist and huggingface_hub is unavailable: {model_ref}") from exc

    snapshot_path = snapshot_download(
        repo_id=str(model_ref),
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "tokenizer*"],
    )
    return Path(snapshot_path)

def _prepare_prompt_tokens(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))

def build_suppress_token_mask(
    vocab_size: int,
    suppress_token_ids: Optional[list[int]],
) -> Optional[mx.array]:
    token_ids = sorted(
        {
            int(token_id)
            for token_id in (suppress_token_ids or [])
            if 0 <= int(token_id) < vocab_size
        }
    )
    if not token_ids:
        return None
    vocab_indices = mx.arange(vocab_size, dtype=mx.int32)
    token_array = mx.array(token_ids, dtype=mx.int32)
    return mx.any(mx.equal(vocab_indices[:, None], token_array[None, :]), axis=1)

def greedy_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    floor = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(suppress_token_mask, floor, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)

def _eval_logits_and_captured(
    logits: mx.array,
    captured: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)

def _ns_to_us(ns: int | float) -> float:
    return float(ns) / 1_000.0

@dataclass(frozen=True)
class VerifyConfig:
    mode: str = "auto"
    enable_qmm: bool = True

    @classmethod
    def from_mode(cls, mode: str | None) -> "VerifyConfig":
        resolved = (mode or "auto").strip().lower()
        if resolved not in ("auto", "off"):
            raise ValueError("verify mode must be auto or off")
        return cls(mode=resolved)

@dataclass(frozen=True)
class DraftQuantSpec:
    weight_bits: int
    group_size: int
    act_bits: int

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

def pack_target_model_weights_selective(
    target_model: Any,
    *,
    validate: bool = True,
    pack_mlp: bool = True,
    pack_attention: bool = False,
) -> dict[str, Any]:
    text_model = resolve_target_ops(target_model).text_model(target_model)
    if getattr(text_model, "_dflash_pack_info", None) is not None:
        return text_model._dflash_pack_info

    pack_info = {
        "enabled": True,
        "validated": validate,
        "pack_mlp": pack_mlp,
        "pack_attention": pack_attention,
        "packed_mlp_layers": [],
        "packed_attention_layers": [],
    }
    text_model._dflash_pack_info = pack_info
    return pack_info

def load_target_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    pack_target_weights: bool = False,
    pack_attention_weights: bool = False,
    validate_packing: bool = True,
    split_full_attention_sdpa: Optional[bool] = None,
    split_full_attention_chunk_size: int = 8,
    quantize_kv_cache: bool = False,
    verify_config: VerifyConfig | None = None,
):
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    split_enabled = (
        default_split_sdpa_enabled(resolved_ref)
        if split_full_attention_sdpa is None
        else bool(split_full_attention_sdpa)
    )
    model, tokenizer, config = load(resolved_ref, lazy=lazy, return_config=True)
    target_ops = resolve_target_ops(model)
    target_family = target_ops.family(model)
    if target_family == "hybrid_gdn":
        target_ops.install_speculative_hooks(model)
        target_ops.configure_full_attention_split(
            model,
            enabled=split_enabled and not quantize_kv_cache,
            chunk_size=split_full_attention_chunk_size,
        )
    meta = {
        "resolved_model_ref": resolved_ref,
        "config": config,
        "quantize_kv_cache": bool(quantize_kv_cache),
        "target_family": target_family,
        "split_full_attention_sdpa": bool(split_enabled and not quantize_kv_cache),
        "split_full_attention_sdpa_requested": split_full_attention_sdpa,
    }
    if pack_target_weights:
        meta["packing"] = pack_target_model_weights_selective(
            model,
            validate=validate_packing,
            pack_mlp=True,
            pack_attention=pack_attention_weights,
        )
    verify_linear_enabled = _verify_enabled_for(config, verify_config=verify_config)
    meta["verify_linear_enabled"] = bool(verify_linear_enabled)
    meta["verify_mode"] = (verify_config.mode if verify_config is not None else "env")
    if verify_linear_enabled:
        from dflash_mlx.verify_linear import install_verify_linears
        n_swapped = install_verify_linears(
            model,
            enable_qmm=_verify_qmm_enabled(verify_config),
        )
        meta["verify_linear_swapped"] = n_swapped
    return model, tokenizer, meta

def _verify_enabled_for(
    config: Any,
    *,
    verify_config: VerifyConfig | None = None,
) -> bool:
    if verify_config is not None:
        if verify_config.mode == "off":
            return False
    else:
        override = _debug_verify_linear_override()
        if override is not None:
            return override
    try:
        text_cfg = config.get("text_config", config) if isinstance(config, dict) else config
        num_experts = int(text_cfg.get("num_experts", 0) or 0)
        num_layers = int(text_cfg.get("num_hidden_layers", 0) or 0)
        hidden_size = int(text_cfg.get("hidden_size", 0) or 0)
        num_heads = int(text_cfg.get("num_attention_heads", 0) or 0)
        num_kv_heads = int(text_cfg.get("num_key_value_heads", 0) or 0)
    except Exception:
        return False
    if num_experts > 0:
        if (
            num_layers == 40
            and hidden_size == 2048
            and num_heads == 16
            and num_kv_heads == 2
        ):
            return False
        return True
    return num_layers >= 40

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
    model, config = load_model(
        model_path,
        lazy=lazy,
        get_model_classes=_get_dflash_model_classes,
    )
    quant_spec = _resolve_draft_quant(draft_quant)
    if quant_spec is not None:
        nn.quantize(model, bits=quant_spec.weight_bits, group_size=quant_spec.group_size)
        if quant_spec.weight_bits in (4, 8):
            from dflash_mlx.verify_linear import install_verify_linears, prewarm_verify_kernels
            install_verify_linears(model, enable_qmm=True)
            prewarm_verify_kernels(model)
        if quant_spec.act_bits == 32:
            def _cast_to_f32(_, x: mx.array) -> mx.array:
                if x.dtype not in (mx.uint32, mx.int32):
                    return x.astype(mx.float32)
                return x
            model.apply(_cast_to_f32)
    return model, {
        "resolved_model_ref": str(model_ref) if model_ref is not None else str(resolved_ref),
        "config": config,
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

def stream_dflash_generate(**kwargs: Any) -> Iterator[dict[str, Any]]:
    if kwargs.get("runtime_context") is None:
        raise ValueError("runtime_context is required")
    runtime_ctx = kwargs["runtime_context"]
    runtime_cfg = runtime_ctx.runtime

    # Create BOD controller if enabled
    if getattr(runtime_cfg, "bod_enabled", False) and kwargs.get("bod_controller") is None:
        from dflash_mlx.bet_optimal_drafting import BODConfig, BODController

        bod_cfg = BODConfig(
            mode=runtime_cfg.bod_mode,
            min_bet=runtime_cfg.bod_min_bet,
            max_bet=runtime_cfg.bod_max_bet,
            default_scale_cost=runtime_cfg.bod_default_scale_cost,
            default_fixed_cost=runtime_cfg.bod_default_fixed_cost,
        )
        kwargs["bod_controller"] = BODController(bod_cfg)

    gen_stream = mx.default_stream(mx.default_device())
    with mx.stream(gen_stream):
        yield from _stream_dflash_generate_impl(**kwargs)

from dflash_mlx.engine.fallback import stream_baseline_generate
from dflash_mlx.engine.spec_epoch import (
    stream_dflash_generate_impl as _stream_dflash_generate_impl,
)

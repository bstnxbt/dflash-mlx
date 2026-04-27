# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import os
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import gated_delta as gated_delta_mod
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.utils import load, load_model

from dflash_mlx.adapter import detect_engine
from dflash_mlx.draft_backend import make_draft_backend
from dflash_mlx.engine.acceptance import match_acceptance_length as _match_acceptance_length
from dflash_mlx.engine.config import (
    _draft_window_override_enabled,
    _effective_draft_window_size,
    _is_unwindowed_full_attention_draft,
    _profile_dflash_cycles_enabled,
    _resolve_dflash_max_ctx,
    _resolve_draft_window,
    _resolve_verify_len_cap,
)
from dflash_mlx.engine.prefill import (
    compute_snapshot_boundary,
    init_target_hidden_from_snapshot,
)
from dflash_mlx.engine.rollback import (
    arm_target_rollback_with_prefix as _arm_target_rollback_with_prefix,
    cleanup_generation_caches as _cleanup_generation_caches,
    clear_rollback_state as _clear_rollback_state,
    restore_target_cache_after_acceptance as _restore_target_cache_after_acceptance,
)
from dflash_mlx.engine.target_verifier import (
    _lm_head_logits,
    _target_text_model,
    _target_text_wrapper,
    extract_context_feature_from_dict,
    target_forward_with_hidden_states,
    verify_target_block as _verify_target_block,
)
from dflash_mlx.model import (
    DFlashDraftModel,
    DFlashDraftModelArgs,
)
from dflash_mlx.cache.codecs import hydrate_target_cache
from dflash_mlx.cache.snapshot import (
    DFlashPrefixSnapshot,
    validate_prefix_snapshot as _validate_prefix_snapshot,
)
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

def resolve_model_ref(model_ref: str | Path | None, *, kind: str) -> str:
    if model_ref:
        candidate = Path(model_ref).expanduser()
        return str(candidate if candidate.exists() else model_ref)
    raise ValueError(f"{kind} model reference is required")

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

def detect_target_family(target_model: Any) -> str:
    inner = _target_text_model(target_model)
    has_linear = any(
        hasattr(layer, "linear_attn") or hasattr(layer, "is_linear")
        for layer in inner.layers
    )
    return "hybrid_gdn" if has_linear else "pure_attention"

def _target_embed_tokens(target_model: Any) -> Any:
    return _target_text_model(target_model).embed_tokens

def _ns_to_us(ns: int | float) -> float:
    return float(ns) / 1_000.0

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
    spec = draft_quant or os.environ.get("DFLASH_DRAFT_QUANT", "").strip()
    if not spec:
        return None
    return parse_draft_quant_spec(spec)

_EXACT_SMALL_PROJ_PAD_M = 16

class _ExactSmallProjPad(nn.Module):
    def __init__(self, linear: nn.Module, *, pad_m: int = _EXACT_SMALL_PROJ_PAD_M):
        super().__init__()
        self.linear = linear
        self.pad_m = int(pad_m)
        self._dflash_exact_small_proj_wrapped = True

    @property
    def weight(self) -> mx.array:
        return self.linear.weight

    @weight.setter
    def weight(self, value: mx.array) -> None:
        self.linear.weight = value

    @property
    def bias(self) -> Optional[mx.array]:
        return getattr(self.linear, "bias", None)

    @bias.setter
    def bias(self, value: Optional[mx.array]) -> None:
        self.linear.bias = value

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3 and x.shape[1] < self.pad_m:
            batch_size, seq_len, hidden_dim = x.shape
            pad = mx.zeros((batch_size, self.pad_m - seq_len, hidden_dim), dtype=x.dtype)
            out = self.linear(mx.concatenate([x, pad], axis=1))
            return out[:, :seq_len, :]
        return self.linear(x)

def _install_exact_small_proj_hooks(
    linear_attn: Any,
    *,
    pad_m: int = _EXACT_SMALL_PROJ_PAD_M,
) -> None:
    for attr_name in ("in_proj_b", "in_proj_a"):
        proj = getattr(linear_attn, attr_name, None)
        if proj is None or getattr(proj, "_dflash_exact_small_proj_wrapped", False):
            continue
        setattr(linear_attn, attr_name, _ExactSmallProjPad(proj, pad_m=pad_m))

def _attention_num_heads(attn: Any) -> int:
    for attr in ("num_attention_heads", "n_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing attention head count attribute")

def _attention_num_kv_heads(attn: Any) -> int:
    for attr in ("num_key_value_heads", "n_kv_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing KV head count attribute")

def _attention_has_gated_q_proj(attn: Any) -> bool:
    q_proj = getattr(attn, "q_proj", None)
    q_norm = getattr(attn, "q_norm", None)
    q_proj_weight = getattr(q_proj, "weight", None)
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_proj_weight is None or q_norm_weight is None:
        return False
    try:
        num_attention_heads = _attention_num_heads(attn)
    except AttributeError:
        return False
    expected_out_dim = 2 * num_attention_heads * int(q_norm_weight.shape[0])
    return int(q_proj_weight.shape[0]) == expected_out_dim

def pack_target_model_weights_selective(
    target_model: Any,
    *,
    validate: bool = True,
    pack_mlp: bool = True,
    pack_attention: bool = False,
) -> dict[str, Any]:
    text_model = _target_text_model(target_model)
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

def _install_speculative_linear_cache_hook(linear_attn: Any) -> None:
    cls = type(linear_attn)
    if getattr(cls, "_dflash_speculative_call_installed", False):
        return

    original_call = cls.__call__

    def speculative_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not isinstance(cache, RecurrentRollbackCache) or not getattr(cache, "_armed", False):
            return original_call(self, inputs, mask=mask, cache=cache)

        from mlx.nn.layers.distributed import sum_gradients

        B, S, _ = inputs.shape
        sharding_group = getattr(self, "sharding_group", None)

        if sharding_group is not None:
            inputs = sum_gradients(sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z_proj = self.in_proj_z(inputs)
        z = z_proj.reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(B, S, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]

        state = cache[1]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        g = gated_delta_mod.compute_g(self.A_log, a, self.dt_bias)
        beta = mx.sigmoid(b)

        if state is None:
            _, _, h_k, d_k = q.shape
            h_v, d_v = v.shape[-2:]
            state = mx.zeros((B, h_v, d_v, d_k), dtype=mx.float32)
        state_in = state

        if (
            mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and not self.training
        ):
            if getattr(cache, "_armed", False):
                from dflash_mlx.kernels import gated_delta_kernel_with_tape

                out, state, innovation_tape = gated_delta_kernel_with_tape(
                    q, k, v, g, beta, state, mask
                )
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )
            else:
                out, state = gated_delta_mod.gated_delta_kernel(q, k, v, g, beta, state, mask)
        else:
            out, state = gated_delta_mod.gated_delta_ops(q, k, v, g, beta, state, mask)
            if getattr(cache, "_armed", False):
                decay = g[..., None, :] if g.ndim == 4 else g[..., None, None]
                decayed_state = state_in[:, None, ...] * decay
                kv_mem = (decayed_state * k[..., None, :]).sum(axis=-1)
                innovation_tape = (v - kv_mem) * beta[..., None]
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )

        cache[1] = state
        cache.advance(S)
        out = self.norm(out, z)
        out_flat = out.reshape(B, S, -1)
        out = self.out_proj(out_flat)

        if sharding_group is not None:
            out = mx.distributed.all_sum(out, group=sharding_group)

        return out

    cls.__call__ = speculative_call
    cls._dflash_speculative_call_installed = True

def _split_sdpa_mask(
    mask: Optional[Any],
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> Optional[Any]:
    if mask is None or mask == "causal":
        return mask
    return mask[..., query_start:query_end, :key_end]

def _split_sdpa_output(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any],
    chunk_size: int,
    cached_prefix_len: int,
) -> mx.array:
    q_len = int(queries.shape[2])
    if q_len <= chunk_size:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    outputs: list[mx.array] = []
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        key_end = cached_prefix_len + end
        chunk_mask = _split_sdpa_mask(mask, query_start=start, query_end=end, key_end=key_end)
        outputs.append(
            scaled_dot_product_attention(
                queries[:, :, start:end, :],
                keys[:, :, :key_end, :],
                values[:, :, :key_end, :],
                cache=cache,
                scale=scale,
                mask=chunk_mask,
            )
        )
    return mx.concatenate(outputs, axis=2)

_HYBRID_SDPA_EXACT_KV_THRESHOLD = 1024

def _install_split_full_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_split_full_attention_installed", False):
        return

    original_call = cls.__call__

    def split_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not getattr(self, "_dflash_split_sdpa_enabled", False):
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        num_attention_heads = _attention_num_heads(self)
        num_key_value_heads = _attention_num_kv_heads(self)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        cached_prefix_len = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0
        if cache is not None:
            queries = self.rope(queries, offset=cached_prefix_len)
            keys = self.rope(keys, offset=cached_prefix_len)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        total_kv_len = int(keys.shape[2])
        exact_prefix_threshold = int(
            getattr(
                self,
                "_dflash_split_sdpa_exact_kv_threshold",
                _HYBRID_SDPA_EXACT_KV_THRESHOLD,
            )
        )
        should_split = (
            cache is not None
            and cached_prefix_len >= exact_prefix_threshold
            and (mask is None or mask == "causal" or isinstance(mask, mx.array))
        )
        should_use_batched_2pass = (
            should_split
            and int(queries.shape[2]) == 16
            and queries.dtype in (mx.bfloat16, mx.float16)
            and int(queries.shape[-1]) in (128, 256)
            and int(values.shape[-1]) in (128, 256)
        )
        if should_use_batched_2pass:
            from dflash_mlx.kernels import batched_sdpa_2pass_exact

            output = batched_sdpa_2pass_exact(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask if isinstance(mask, mx.array) else None,
            )
            if output is None:
                output = _split_sdpa_output(
                    queries=queries,
                    keys=keys,
                    values=values,
                    scale=self.scale,
                    mask=mask,
                    cache=cache,
                    chunk_size=1,
                    cached_prefix_len=cached_prefix_len,
                )
        elif should_split:
            output = _split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask,
                cache=cache,
                chunk_size=1,
                cached_prefix_len=cached_prefix_len,
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gated_output = output * mx.sigmoid(gate)
        return self.o_proj(gated_output)

    cls.__call__ = split_call
    cls._dflash_split_full_attention_installed = True

def _install_target_speculative_hooks(target_model: Any) -> None:
    text_model = _target_text_model(target_model)
    if getattr(text_model, "_dflash_speculative_hooks_installed", False):
        return
    if detect_target_family(target_model) == "pure_attention":
        text_model._dflash_speculative_hooks_installed = True
        return
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            _install_exact_small_proj_hooks(layer.linear_attn)
            _install_speculative_linear_cache_hook(layer.linear_attn)
        elif not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            _install_split_full_attention_hook(layer.self_attn)
    text_model._dflash_speculative_hooks_installed = True

def configure_full_attention_split(
    target_model: Any,
    *,
    enabled: bool,
    chunk_size: int = 8,
) -> None:
    text_model = _target_text_model(target_model)
    if detect_target_family(target_model) == "pure_attention":
        return
    _install_target_speculative_hooks(target_model)
    for layer in text_model.layers:
        if not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            layer.self_attn._dflash_split_sdpa_enabled = enabled
            layer.self_attn._dflash_split_sdpa_chunk_size = int(chunk_size)
            layer.self_attn._dflash_split_sdpa_exact_kv_threshold = (
                _HYBRID_SDPA_EXACT_KV_THRESHOLD
            )

def make_target_cache(
    target_model: Any,
    *,
    enable_speculative_linear_cache: bool,
    quantize_kv_cache: bool = False,
) -> list[Any]:
    text_model = _target_text_model(target_model)
    caches: list[Any] = []
    for layer_index, layer in enumerate(text_model.layers):
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            if enable_speculative_linear_cache:
                _install_target_speculative_hooks(target_model)
                conv_kernel_size = int(getattr(layer.linear_attn, "conv_kernel_size", 4))
                caches.append(
                    RecurrentRollbackCache(size=2, conv_kernel_size=conv_kernel_size)
                )
            else:
                caches.append(cache_mod.ArraysCache(size=2))
        else:
            if quantize_kv_cache:
                caches.append(cache_mod.QuantizedKVCache(group_size=64, bits=8))
            else:
                caches.append(cache_mod.KVCache())
    return caches

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
):
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    split_enabled = (
        default_split_sdpa_enabled(resolved_ref)
        if split_full_attention_sdpa is None
        else bool(split_full_attention_sdpa)
    )
    model, tokenizer, config = load(resolved_ref, lazy=lazy, return_config=True)
    target_family = detect_target_family(model)
    if target_family == "hybrid_gdn":
        _install_target_speculative_hooks(model)
        configure_full_attention_split(
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
    verify_linear_enabled = _verify_enabled_for(config)
    meta["verify_linear_enabled"] = bool(verify_linear_enabled)
    if verify_linear_enabled:
        os.environ.setdefault("DFLASH_VERIFY_QMM", "1")
        from dflash_mlx.verify_linear import install_verify_linears
        n_swapped = install_verify_linears(model)
        meta["verify_linear_swapped"] = n_swapped
    return model, tokenizer, meta

def _verify_enabled_for(config: Any) -> bool:
    override = os.environ.get("DFLASH_VERIFY_LINEAR", "").strip()
    if override == "1":
        return True
    if override == "0":
        return False
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

def load_draft_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    draft_quant: str | None = None,
    quantize_draft: bool = False,
):
    resolved_ref = resolve_model_ref(model_ref, kind="draft")
    model_path = _resolve_local_model_path(resolved_ref)
    model, config = load_model(
        model_path,
        lazy=lazy,
        get_model_classes=_get_dflash_model_classes,
    )
    if quantize_draft and not draft_quant:
        draft_quant = "w4a16"
    quant_spec = _resolve_draft_quant(draft_quant)
    if quant_spec is not None:
        nn.quantize(model, bits=quant_spec.weight_bits, group_size=quant_spec.group_size)
        if quant_spec.weight_bits in (4, 8):
            os.environ.setdefault("DFLASH_VERIFY_QMM", "1")
            from dflash_mlx.verify_linear import install_verify_linears, prewarm_verify_kernels
            install_verify_linears(model)
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
        "quantize_draft": quant_spec is not None,
    }

def stream_dflash_generate(**kwargs: Any) -> Iterator[dict[str, Any]]:
    gen_stream = mx.default_stream(mx.default_device())
    with mx.stream(gen_stream):
        yield from _stream_dflash_generate_impl(**kwargs)


from dflash_mlx.engine.fallback import stream_baseline_generate  # noqa: E402
from dflash_mlx.engine.spec_epoch import (  # noqa: E402
    stream_dflash_generate_impl as _stream_dflash_generate_impl,
)

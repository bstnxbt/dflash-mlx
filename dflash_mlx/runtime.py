# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import gated_delta as gated_delta_mod
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.utils import load, load_model

from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
    DFlashDraftModelArgs,
    extract_context_feature,
)
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache


def resolve_model_ref(model_ref: str | Path | None, *, kind: str) -> str:
    if model_ref:
        candidate = Path(model_ref).expanduser()
        return str(candidate if candidate.exists() else model_ref)
    raise ValueError(f"{kind} model reference is required")


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


def sample_tokens(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


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


def sample_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return sample_tokens(logits)
    floor = mx.array(-1e9, dtype=logits.dtype)
    return mx.argmax(mx.where(suppress_token_mask, floor, logits), axis=-1)


def greedy_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    floor = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(suppress_token_mask, floor, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)


def _match_acceptance_length(
    drafted_tokens: mx.array,
    posterior_tokens: mx.array,
) -> mx.array:
    if int(drafted_tokens.shape[0]) == 0:
        return mx.array(0, dtype=mx.int32)
    matches = mx.equal(drafted_tokens, posterior_tokens).astype(mx.int32)
    return mx.sum(mx.cumprod(matches, axis=0))


def _concat_hidden_state_chunks(
    hidden_state_chunks: list[list[mx.array]],
) -> list[mx.array]:
    if not hidden_state_chunks:
        raise ValueError("expected at least one hidden-state chunk")
    if len(hidden_state_chunks) == 1:
        return hidden_state_chunks[0]
    return [
        mx.concatenate([chunk[index] for chunk in hidden_state_chunks], axis=1)
        for index in range(len(hidden_state_chunks[0]))
    ]


def _concat_hidden_state_chunk_dicts(
    hidden_state_chunks: list[dict[int, mx.array]],
    capture_layer_ids: set[int],
) -> dict[int, mx.array]:
    if not hidden_state_chunks:
        raise ValueError("expected at least one hidden-state chunk")
    if len(hidden_state_chunks) == 1:
        return hidden_state_chunks[0]
    return {
        layer_id: mx.concatenate([chunk[layer_id] for chunk in hidden_state_chunks], axis=1)
        for layer_id in sorted(capture_layer_ids)
    }


def _eval_logits_and_captured(
    logits: mx.array,
    captured: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)


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


def detect_target_family(target_model: Any) -> str:
    inner = _target_text_model(target_model)
    has_linear = any(
        hasattr(layer, "linear_attn") or hasattr(layer, "is_linear")
        for layer in inner.layers
    )
    return "hybrid_gdn" if has_linear else "pure_attention"


def _target_embed_tokens(target_model: Any) -> Any:
    return _target_text_model(target_model).embed_tokens


def _lm_head_logits(target_model: Any, hidden_states: mx.array) -> mx.array:
    wrapper = _target_text_wrapper(target_model)
    if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True):
        return wrapper.model.embed_tokens.as_linear(hidden_states)
    return wrapper.lm_head(hidden_states)


def extract_context_feature_from_dict(
    captured_dict: dict[int, mx.array],
    target_layer_ids: list[int],
) -> mx.array:
    selected = [captured_dict[layer_id + 1] for layer_id in target_layer_ids]
    return mx.concatenate(selected, axis=-1)


def _resolve_verify_len_cap(target_model: Any, block_tokens: int) -> int:
    override_raw = os.environ.get("DFLASH_VERIFY_LEN", "").strip()
    if override_raw:
        try:
            override = int(override_raw)
        except ValueError:
            override = 0
        if override > 0:
            return max(1, min(int(block_tokens), override))
    return int(block_tokens)


def _resolve_dflash_max_ctx() -> int:
    raw = os.environ.get("DFLASH_MAX_CTX", "0").strip()
    try:
        max_ctx = int(raw)
    except ValueError:
        max_ctx = 0
    if max_ctx <= 0:
        return sys.maxsize
    return max_ctx


def _resolve_draft_window() -> tuple[int, int]:
    sink = int(os.environ.get("DFLASH_DRAFT_SINK", "64").strip())
    window = int(os.environ.get("DFLASH_DRAFT_WINDOW", "1024").strip())
    return max(0, sink), max(1, window)


def _profile_dflash_cycles_enabled() -> bool:
    raw = os.environ.get("DFLASH_PROFILE", "").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _ns_to_us(ns: int | float) -> float:
    return float(ns) / 1_000.0


def _should_quantize_draft(quantize_draft: bool = False) -> bool:
    if quantize_draft:
        return True
    raw = os.environ.get("DFLASH_QUANTIZE_DRAFT", "").strip().lower()
    return raw not in {"", "0", "false", "no"}




def _linear_forward(x: mx.array, weight: mx.array, bias: Optional[mx.array]) -> mx.array:
    out = x @ weight.T
    return out if bias is None else out + bias


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


def pack_target_model_weights(target_model: Any, *, validate: bool = True) -> dict[str, Any]:
    return pack_target_model_weights_selective(
        target_model,
        validate=validate,
        pack_mlp=True,
        pack_attention=False,
    )


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
            state = mx.zeros((B, h_v, d_v, d_k), dtype=q.dtype)
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
    split_full_attention_sdpa: bool = True,
    split_full_attention_chunk_size: int = 8,
    quantize_kv_cache: bool = False,
):
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    model, tokenizer, config = load(resolved_ref, lazy=lazy, return_config=True)
    target_family = detect_target_family(model)
    if target_family == "hybrid_gdn":
        _install_target_speculative_hooks(model)
        configure_full_attention_split(
            model,
            enabled=split_full_attention_sdpa and not quantize_kv_cache,
            chunk_size=split_full_attention_chunk_size,
        )
    meta = {
        "resolved_model_ref": resolved_ref,
        "config": config,
        "quantize_kv_cache": bool(quantize_kv_cache),
        "target_family": target_family,
    }
    if pack_target_weights:
        meta["packing"] = pack_target_model_weights_selective(
            model,
            validate=validate_packing,
            pack_mlp=True,
            pack_attention=pack_attention_weights,
        )
    return model, tokenizer, meta


def load_draft_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    quantize_draft: bool = False,
):
    resolved_ref = resolve_model_ref(model_ref, kind="draft")
    model_path = _resolve_local_model_path(resolved_ref)
    model, config = load_model(
        model_path,
        lazy=lazy,
        get_model_classes=_get_dflash_model_classes,
    )
    quantized = _should_quantize_draft(quantize_draft)
    if quantized:
        nn.quantize(model, bits=4, group_size=64)
    return model, {
        "resolved_model_ref": str(model_ref) if model_ref is not None else str(resolved_ref),
        "config": config,
        "quantize_draft": bool(quantized),
    }


def target_forward_with_hidden_states(
    target_model: Any,
    *,
    input_ids: Optional[mx.array] = None,
    cache: Optional[list[Any]] = None,
    input_embeddings: Optional[mx.array] = None,
    capture_layer_ids: Optional[set[int]] = None,
) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
    inner = _target_text_model(target_model)
    hidden_states = input_embeddings if input_embeddings is not None else inner.embed_tokens(input_ids)
    if cache is None:
        cache = [None] * len(inner.layers)
    capture_all = capture_layer_ids is None
    if capture_all:
        captured: list[mx.array] | dict[int, mx.array] = [hidden_states]
    else:
        capture_layer_ids = set(capture_layer_ids)
        captured = {0: hidden_states} if 0 in capture_layer_ids else {}
    h = hidden_states

    if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
        fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            h = layer(h, mask=mask, cache=layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h
    else:
        mask = create_attention_mask(hidden_states, cache[0])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            h = layer(h, mask, layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h
    normalized = inner.norm(h)
    logits = _lm_head_logits(target_model, normalized)
    return logits, captured


def trim_cache_to(cache_entries: list[Any], size: int) -> int:
    if not cache_entries:
        return 0
    current_size = int(getattr(cache_entries[0], "offset", 0) or 0)
    if current_size <= size:
        return 0
    return int(cache_mod.trim_prompt_cache(cache_entries, current_size - size) or 0)


def _arm_target_rollback(cache_entries: list[Any]) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback()


def _arm_target_rollback_with_prefix(
    cache_entries: list[Any],
    *,
    prefix_len: int,
) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback(prefix_len=int(prefix_len))


def _clear_rollback_state(cache_entry: Any) -> None:
    if hasattr(cache_entry, "clear_transients"):
        cache_entry.clear_transients()
        return
    if hasattr(cache_entry, "_armed"):
        cache_entry._armed = False
    if hasattr(cache_entry, "_tape"):
        cache_entry._tape = None
    if hasattr(cache_entry, "_tape_k"):
        cache_entry._tape_k = None
    if hasattr(cache_entry, "_tape_g"):
        cache_entry._tape_g = None
    if hasattr(cache_entry, "_tape_qkv"):
        cache_entry._tape_qkv = None
    if hasattr(cache_entry, "_snapshot"):
        cache_entry._snapshot = None


def _cleanup_generation_caches(
    target_cache: list[Any],
    draft_cache: list[Any],
) -> None:
    for cache_entry in target_cache:
        if hasattr(cache_entry, "clear_transients"):
            cache_entry.clear_transients()
    draft_cache.clear()
    target_cache.clear()


def _restore_target_cache_after_acceptance(
    cache_entries: list[Any],
    *,
    target_len: int,
    acceptance_length: int,
    drafted_tokens: int = 0,
) -> int:
    replay_ns_total = 0
    fully_accepted = drafted_tokens > 0 and acceptance_length == drafted_tokens
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "rollback"):
            if fully_accepted:
                _clear_rollback_state(cache_entry)
                continue
            replay_start_ns = time.perf_counter_ns()
            cache_entry.rollback(acceptance_length)
            replay_ns_total += time.perf_counter_ns() - replay_start_ns
        elif hasattr(cache_entry, "trim"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                replay_start_ns = time.perf_counter_ns()
                cache_entry.trim(offset - target_len)
                replay_ns_total += time.perf_counter_ns() - replay_start_ns
        elif hasattr(cache_entry, "offset"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                cache_entry.offset = target_len
        elif hasattr(cache_entry, "crop"):
            cache_entry.crop(target_len)
    return replay_ns_total


def _verify_target_block(
    *,
    target_model: Any,
    verify_ids: mx.array,
    target_cache: list[Any],
    verify_chunk_tokens: Optional[int],
    capture_layer_ids: Optional[set[int]] = None,
) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
    total_tokens = int(verify_ids.shape[1])
    if total_tokens <= 0:
        raise ValueError("verify block must contain at least one token")

    chunk_size = max(1, int(verify_chunk_tokens or total_tokens))
    if chunk_size >= total_tokens:
        verify_logits, verify_hidden_states = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        return verify_logits, verify_hidden_states

    logits_chunks: list[mx.array] = []
    hidden_state_chunks: list[list[mx.array]] | list[dict[int, mx.array]]
    hidden_state_chunks = []
    for offset in range(0, total_tokens, chunk_size):
        verify_chunk = verify_ids[:, offset : offset + chunk_size]
        chunk_logits, chunk_hidden_states = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_chunk,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        logits_chunks.append(chunk_logits)
        hidden_state_chunks.append(chunk_hidden_states)

    if capture_layer_ids is None:
        return mx.concatenate(logits_chunks, axis=1), _concat_hidden_state_chunks(hidden_state_chunks)
    return (
        mx.concatenate(logits_chunks, axis=1),
        _concat_hidden_state_chunk_dicts(hidden_state_chunks, capture_layer_ids),
    )


def generate_baseline_once(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    stop_token_ids = list(stop_token_ids or [])

    if max_new_tokens <= 0:
        return {
            "elapsed_us": 0.0,
            "prompt_token_count": len(prompt_tokens),
            "generated_token_ids": [],
            "generation_tokens": 0,
        }

    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=quantize_kv_cache,
    )
    start_ns = time.perf_counter_ns()

    prefill_start_ns = time.perf_counter_ns()
    logits = target_model(prompt_array, cache=cache)
    mx.eval(logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns
    suppress_token_mask = build_suppress_token_mask(int(logits.shape[-1]), suppress_token_ids)
    next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
    generated_tokens = [next_token]

    while len(generated_tokens) < max_new_tokens:
        if next_token in stop_token_ids:
            break
        token_array = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(token_array, cache=cache)
        next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
        generated_tokens.append(next_token)

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    return {
        "elapsed_us": elapsed_us,
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": len(prompt_tokens),
        "generated_token_ids": generated_tokens,
        "generation_tokens": len(generated_tokens),
        "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
    }


def stream_baseline_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    fallback_reason: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    prompt_len = len(prompt_tokens)
    stop_token_ids = list(stop_token_ids or [])
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=quantize_kv_cache,
    )
    start_ns = time.perf_counter_ns()

    prefill_start_ns = time.perf_counter_ns()
    logits = target_model(prompt_array, cache=cache)
    mx.eval(logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns
    suppress_token_mask = build_suppress_token_mask(int(logits.shape[-1]), suppress_token_ids)
    next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
    generated_tokens = [next_token]

    yield {
        "event": "prefill",
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": prompt_len,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }

    yield {
        "event": "token",
        "token_id": next_token,
        "generated_tokens": 1,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }

    while len(generated_tokens) < max_new_tokens:
        if next_token in stop_token_ids:
            break
        token_array = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(token_array, cache=cache)
        next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
        generated_tokens.append(next_token)
        yield {
            "event": "token",
            "token_id": next_token,
            "generated_tokens": len(generated_tokens),
            "acceptance_ratio": 0.0,
            "cycles_completed": 0,
            "fallback_ar": True,
            "fallback_reason": fallback_reason,
        }

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    yield {
        "event": "summary",
        "elapsed_us": elapsed_us,
        "prompt_token_count": prompt_len,
        "generated_token_ids": generated_tokens,
        "generation_tokens": len(generated_tokens),
        "accepted_from_draft": 0,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "phase_timings_us": {
            "prefill": prefill_ns / 1_000.0,
            "draft": 0.0,
            "draft_prefill": 0.0,
            "draft_incremental": 0.0,
            "verify": 0.0,
            "replay": 0.0,
            "commit": 0.0,
        },
        "verify_len_cap": None,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }


def generate_dflash_once(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    verify_chunk_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    if quantize_kv_cache:
        configure_full_attention_split(target_model, enabled=False)
    draft_sink_size, draft_window_size = _resolve_draft_window()

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    prompt_len = len(prompt_tokens)
    dflash_max_ctx = _resolve_dflash_max_ctx()
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        baseline = generate_baseline_once(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
        )
        baseline.update(
            {
                "accepted_from_draft": 0,
                "acceptance_ratio": 0.0,
                "cycles_completed": 0,
                "phase_timings_us": {
                    "prefill": baseline["elapsed_us"],
                    "draft": 0.0,
                    "draft_prefill": 0.0,
                    "draft_incremental": 0.0,
                    "verify": 0.0,
                    "replay": 0.0,
                    "commit": 0.0,
                },
                "speculative_linear_cache": False,
                "verify_chunk_tokens": None,
                "verify_len_cap": None,
                "quantize_kv_cache": bool(quantize_kv_cache),
                "fallback_ar": True,
                "fallback_reason": fallback_reason,
            }
        )
        return baseline
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    use_speculative_linear_cache = verify_chunk_tokens is None
    target_cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=use_speculative_linear_cache,
        quantize_kv_cache=quantize_kv_cache,
    )

    draft_cache = [
        ContextOnlyDraftKVCache(
            sink_size=draft_sink_size,
            window_size=draft_window_size,
        )
        for _ in range(len(draft_model.layers))
    ]
    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}

    try:
        start_ns = time.perf_counter_ns()
        prefill_start_ns = time.perf_counter_ns()
        prefill_step_size = 2048
        prefill_logits = None
        target_hidden_chunks: list[mx.array] = []
        for chunk_start in range(0, prompt_len, prefill_step_size):
            chunk_end = min(chunk_start + prefill_step_size, prompt_len)
            chunk_ids = prompt_array[:, chunk_start:chunk_end]
            prefill_logits, prefill_hidden_states = target_forward_with_hidden_states(
                target_model,
                input_ids=chunk_ids,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(prefill_logits, prefill_hidden_states)
            target_hidden_chunks.append(
                extract_context_feature_from_dict(
                    prefill_hidden_states,
                    list(draft_model.target_layer_ids),
                )
            )
        prefill_ns = time.perf_counter_ns() - prefill_start_ns

        suppress_token_mask = build_suppress_token_mask(int(prefill_logits.shape[-1]), suppress_token_ids)
        staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_token_mask).reshape(-1)
        target_hidden = (
            target_hidden_chunks[0]
            if len(target_hidden_chunks) == 1
            else mx.concatenate(target_hidden_chunks, axis=1)
        )

        draft_block_size = int(draft_model.block_size)
        requested_block_tokens = draft_block_size if block_tokens is None else int(block_tokens)
        effective_block_tokens = max(1, min(requested_block_tokens, draft_block_size))
        generated_token_buffer = mx.full((max_new_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
        block_token_buffer = mx.full((effective_block_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
        generated_token_count = 0
        accepted_from_draft = 0
        cycles_completed = 0
        verify_len_cap = _resolve_verify_len_cap(target_model, effective_block_tokens)
        start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        seen_draft_cycle = False
        acceptance_history: list[int] = []
        profile_cycles = _profile_dflash_cycles_enabled()
        cycle_profiles: list[dict[str, Any]] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
        }

        while generated_token_count < max_new_tokens:
            cycle_start_ns = time.perf_counter_ns()
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            commit_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            remaining = max_new_tokens - generated_token_count
            block_len = max(1, min(effective_block_tokens, remaining))
            block_token_buffer[:block_len] = draft_model.mask_token_id
            block_token_buffer[:1] = staged_first
            block_token_ids = block_token_buffer[:block_len]

            if block_len > 1:
                draft_start_ns = time.perf_counter_ns()
                noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
                draft_hidden = draft_model(
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    cache=draft_cache,
                )
                draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])
                mx.async_eval(draft_logits)
                mx.eval(draft_logits)
                drafted = greedy_tokens_with_mask(draft_logits, suppress_token_mask).squeeze(0)
                block_token_ids[1:block_len] = drafted
                draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            verify_token_ids = block_token_ids[: min(block_len, verify_len_cap)]
            verify_ids = verify_token_ids[None]
            if use_speculative_linear_cache:
                _arm_target_rollback_with_prefix(target_cache, prefix_len=start)
            verify_start_ns = time.perf_counter_ns()
            verify_logits, verify_hidden_states = _verify_target_block(
                target_model=target_model,
                verify_ids=verify_ids,
                target_cache=target_cache,
                verify_chunk_tokens=verify_chunk_tokens,
                capture_layer_ids=capture_layer_ids,
            )
            if profile_cycles:
                _eval_logits_and_captured(verify_logits, verify_hidden_states)
            verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
            verify_ns_total += verify_cycle_ns

            acceptance_start_ns = time.perf_counter_ns()
            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
            acceptance_len = int(
                _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
            )
            acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            acceptance_history.append(acceptance_len)

            hidden_extract_start_ns = time.perf_counter_ns()
            committed_hidden = extract_context_feature_from_dict(
                verify_hidden_states,
                list(draft_model.target_layer_ids),
            )[:, : (1 + acceptance_len), :]
            mx.eval(committed_hidden, posterior)
            hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            generated_token_buffer[generated_token_count : generated_token_count + commit_count] = committed_segment
            generated_token_count += commit_count
            accepted_from_draft += acceptance_len

            commit_start_ns = time.perf_counter_ns()
            start += commit_count
            target_hidden = committed_hidden
            replay_cycle_ns = _restore_target_cache_after_acceptance(
                target_cache,
                target_len=start,
                acceptance_length=acceptance_len,
                drafted_tokens=block_len - 1,
            )
            replay_ns_total += replay_cycle_ns
            cycles_completed += 1
            commit_wall_ns = time.perf_counter_ns() - commit_start_ns
            commit_ns_total += commit_wall_ns
            commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            if stop_hit:
                break

            staged_first = posterior[acceptance_len : acceptance_len + 1]

            if profile_cycles:
                cycle_total_ns = time.perf_counter_ns() - cycle_start_ns
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profiles.append(
                    {
                        "cycle": cycles_completed,
                        "block_len": int(block_len),
                        "commit_count": int(commit_count),
                        "acceptance_len": int(acceptance_len),
                        "draft_us": _ns_to_us(draft_cycle_ns),
                        "verify_us": _ns_to_us(verify_cycle_ns),
                        "acceptance_us": _ns_to_us(acceptance_cycle_ns),
                        "hidden_extraction_us": _ns_to_us(hidden_extract_cycle_ns),
                        "rollback_us": _ns_to_us(replay_cycle_ns),
                        "other_us": _ns_to_us(other_cycle_ns),
                        "cycle_total_us": _ns_to_us(cycle_total_ns),
                    }
                )
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns

        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        generated_token_ids = (
            generated_token_buffer[:generated_token_count].tolist()
            if generated_token_count > 0
            else []
        )
        first_20 = acceptance_history[:20]
        last_20 = acceptance_history[-20:]
        result = {
            "elapsed_us": elapsed_us,
            "prompt_token_count": prompt_len,
            "generated_token_ids": generated_token_ids,
            "generation_tokens": len(generated_token_ids),
            "accepted_from_draft": accepted_from_draft,
            "acceptance_ratio": (
                accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
            ),
            "block_tokens": effective_block_tokens,
            "cycles_completed": cycles_completed,
            "phase_timings_us": {
                "prefill": prefill_ns / 1_000.0,
                "draft": draft_ns_total / 1_000.0,
                "draft_prefill": draft_prefill_ns / 1_000.0,
                "draft_incremental": draft_incremental_ns / 1_000.0,
                "verify": verify_ns_total / 1_000.0,
                "replay": replay_ns_total / 1_000.0,
                "commit": commit_ns_total / 1_000.0,
            },
            "speculative_linear_cache": use_speculative_linear_cache,
            "verify_chunk_tokens": int(verify_chunk_tokens) if verify_chunk_tokens else None,
            "verify_len_cap": int(verify_len_cap),
            "quantize_kv_cache": bool(quantize_kv_cache),
            "tokens_per_cycle": (len(generated_token_ids) / cycles_completed) if cycles_completed > 0 else 0.0,
            "acceptance_first_20_avg": (sum(first_20) / len(first_20)) if first_20 else 0.0,
            "acceptance_last_20_avg": (sum(last_20) / len(last_20)) if last_20 else 0.0,
            "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
        }
        if profile_cycles:
            result["cycle_profile_us"] = cycle_profiles
            result["cycle_profile_totals_us"] = {key: _ns_to_us(value) for key, value in profile_totals_ns.items()}
        return result
    finally:
        _cleanup_generation_caches(target_cache, draft_cache)
        del draft_cache
        del target_cache


def stream_dflash_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
) -> Iterator[dict[str, Any]]:
    if quantize_kv_cache:
        configure_full_attention_split(target_model, enabled=False)
    draft_sink_size, draft_window_size = _resolve_draft_window()

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    fallback_ar = False
    fallback_reason: Optional[str] = None

    prompt_len = len(prompt_tokens)
    dflash_max_ctx = _resolve_dflash_max_ctx()
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        yield from stream_baseline_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            fallback_reason=fallback_reason,
        )
        return
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    target_cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=quantize_kv_cache,
    )
    draft_cache = [
        ContextOnlyDraftKVCache(
            sink_size=draft_sink_size,
            window_size=draft_window_size,
        )
        for _ in range(len(draft_model.layers))
    ]
    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}

    try:
        start_ns = time.perf_counter_ns()
        prefill_start_ns = time.perf_counter_ns()
        prefill_step_size = 2048
        prefill_logits = None
        target_hidden_chunks: list[mx.array] = []
        for chunk_start in range(0, prompt_len, prefill_step_size):
            chunk_end = min(chunk_start + prefill_step_size, prompt_len)
            chunk_ids = prompt_array[:, chunk_start:chunk_end]
            prefill_logits, prefill_hidden_states = target_forward_with_hidden_states(
                target_model,
                input_ids=chunk_ids,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(prefill_logits, prefill_hidden_states)
            target_hidden_chunks.append(
                extract_context_feature_from_dict(
                    prefill_hidden_states,
                    list(draft_model.target_layer_ids),
                )
            )
            yield {
                "event": "prefill_progress",
                "tokens_processed": chunk_end,
                "tokens_total": prompt_len,
            }
        prefill_ns = time.perf_counter_ns() - prefill_start_ns

        suppress_token_mask = build_suppress_token_mask(int(prefill_logits.shape[-1]), suppress_token_ids)
        staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_token_mask).reshape(-1)
        target_hidden = (
            target_hidden_chunks[0]
            if len(target_hidden_chunks) == 1
            else mx.concatenate(target_hidden_chunks, axis=1)
        )

        yield {
            "event": "prefill",
            "prefill_us": prefill_ns / 1_000.0,
            "prompt_token_count": prompt_len,
        }

        draft_block_size = int(draft_model.block_size)
        requested_block_tokens = draft_block_size if block_tokens is None else int(block_tokens)
        effective_block_tokens = max(1, min(requested_block_tokens, draft_block_size))
        block_token_buffer = mx.full((effective_block_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
        generated_token_ids: list[int] = []
        accepted_from_draft = 0
        cycles_completed = 0
        verify_len_cap = _resolve_verify_len_cap(target_model, effective_block_tokens)
        start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        seen_draft_cycle = False
        profile_cycles = _profile_dflash_cycles_enabled()
        cycle_profiles: list[dict[str, Any]] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
        }

        while len(generated_token_ids) < max_new_tokens:
            cycle_start_ns = time.perf_counter_ns()
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            commit_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            remaining = max_new_tokens - len(generated_token_ids)
            block_len = max(1, min(effective_block_tokens, remaining))
            block_token_buffer[:block_len] = draft_model.mask_token_id
            block_token_buffer[:1] = staged_first
            block_token_ids = block_token_buffer[:block_len]

            if block_len > 1:
                draft_start_ns = time.perf_counter_ns()
                noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
                draft_hidden = draft_model(
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    cache=draft_cache,
                )
                draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])
                mx.async_eval(draft_logits)
                mx.eval(draft_logits)
                drafted = greedy_tokens_with_mask(draft_logits, suppress_token_mask).squeeze(0)
                block_token_ids[1:block_len] = drafted
                draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            verify_token_ids = block_token_ids[: min(block_len, verify_len_cap)]
            verify_ids = verify_token_ids[None]
            _arm_target_rollback_with_prefix(target_cache, prefix_len=start)
            verify_start_ns = time.perf_counter_ns()
            verify_logits, verify_hidden_states = _verify_target_block(
                target_model=target_model,
                verify_ids=verify_ids,
                target_cache=target_cache,
                verify_chunk_tokens=None,
                capture_layer_ids=capture_layer_ids,
            )
            if profile_cycles:
                _eval_logits_and_captured(verify_logits, verify_hidden_states)
            verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
            verify_ns_total += verify_cycle_ns

            acceptance_start_ns = time.perf_counter_ns()
            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
            acceptance_len = int(
                _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
            )
            acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            hidden_extract_start_ns = time.perf_counter_ns()
            committed_hidden = extract_context_feature_from_dict(
                verify_hidden_states,
                list(draft_model.target_layer_ids),
            )[:, : (1 + acceptance_len), :]
            mx.eval(committed_hidden, posterior)
            hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            commit_start_ns = time.perf_counter_ns()
            start += commit_count
            target_hidden = committed_hidden
            replay_cycle_ns = _restore_target_cache_after_acceptance(
                target_cache,
                target_len=start,
                acceptance_length=acceptance_len,
                drafted_tokens=block_len - 1,
            )
            replay_ns_total += replay_cycle_ns
            cycles_completed += 1
            commit_wall_ns = time.perf_counter_ns() - commit_start_ns
            commit_ns_total += commit_wall_ns
            commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

            accepted_from_draft += acceptance_len
            committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
            for token_id in committed_ids:
                if len(generated_token_ids) >= max_new_tokens:
                    break
                generated_token_ids.append(token_id)
                yield {
                    "event": "token",
                    "token_id": token_id,
                    "generated_tokens": len(generated_token_ids),
                    "acceptance_ratio": (
                        accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
                    ),
                    "cycles_completed": cycles_completed,
                }

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            if stop_hit:
                break

            staged_first = posterior[acceptance_len : acceptance_len + 1]

            if profile_cycles:
                cycle_total_ns = time.perf_counter_ns() - cycle_start_ns
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profiles.append(
                    {
                        "cycle": cycles_completed,
                        "block_len": int(block_len),
                        "commit_count": int(commit_count),
                        "acceptance_len": int(acceptance_len),
                        "draft_us": _ns_to_us(draft_cycle_ns),
                        "verify_us": _ns_to_us(verify_cycle_ns),
                        "acceptance_us": _ns_to_us(acceptance_cycle_ns),
                        "hidden_extraction_us": _ns_to_us(hidden_extract_cycle_ns),
                        "rollback_us": _ns_to_us(replay_cycle_ns),
                        "other_us": _ns_to_us(other_cycle_ns),
                        "cycle_total_us": _ns_to_us(cycle_total_ns),
                    }
                )
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns

        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        summary = {
            "event": "summary",
            "elapsed_us": elapsed_us,
            "prompt_token_count": prompt_len,
            "generated_token_ids": generated_token_ids,
            "generation_tokens": len(generated_token_ids),
            "accepted_from_draft": accepted_from_draft,
            "acceptance_ratio": (
                accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
            ),
            "block_tokens": effective_block_tokens,
            "cycles_completed": cycles_completed,
            "phase_timings_us": {
                "prefill": prefill_ns / 1_000.0,
                "draft": draft_ns_total / 1_000.0,
                "draft_prefill": draft_prefill_ns / 1_000.0,
                "draft_incremental": draft_incremental_ns / 1_000.0,
                "verify": verify_ns_total / 1_000.0,
                "replay": replay_ns_total / 1_000.0,
                "commit": commit_ns_total / 1_000.0,
            },
            "verify_len_cap": int(verify_len_cap),
        }
        if profile_cycles:
            summary["cycle_profile_us"] = cycle_profiles
            summary["cycle_profile_totals_us"] = {
                key: _ns_to_us(value) for key, value in profile_totals_ns.items()
            }
        yield summary
    finally:
        _cleanup_generation_caches(target_cache, draft_cache)
        del draft_cache
        del target_cache

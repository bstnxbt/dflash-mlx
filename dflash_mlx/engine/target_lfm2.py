# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import time
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models import cache as cache_mod
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
)

from dflash_mlx.engine.target_ops import TargetCapabilities
from dflash_mlx.short_conv_rollback_cache import ShortConvRollbackCache


def _install_short_conv_rollback_hook(conv_module: Any) -> None:
    cls = type(conv_module)
    if getattr(cls, "_dflash_short_conv_rollback_installed", False):
        return

    original_call = cls.__call__

    def patched_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not isinstance(cache, ShortConvRollbackCache):
            return original_call(self, x, mask=mask, cache=cache)

        BCx = self.in_proj(x)
        B_split, C, x_val = mx.split(BCx, 3, axis=-1)
        Bx = B_split * x_val
        if mask is not None:
            Bx = mx.where(mask[..., None], Bx, 0)

        if cache[0] is None:
            state = mx.zeros(
                (Bx.shape[0], self.L_cache - 1, self.args.hidden_size),
                dtype=Bx.dtype,
            )
        else:
            state = cache[0]
        Bx = mx.concatenate([state, Bx], axis=1)

        cache.record_tape_if_armed(Bx)

        n_keep = self.L_cache - 1
        t = x_val.shape[1]
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, t)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(Bx, positions, axis=1)
        else:
            cache[0] = mx.contiguous(Bx[:, -n_keep:, :])
        cache.advance(t)

        conv_out = self.conv(Bx)
        y = C * conv_out
        return self.out_proj(y)

    cls.__call__ = patched_call
    cls._dflash_short_conv_rollback_installed = True


class Lfm2TargetOps:
    backend_name = "lfm2"

    def model_type(self, target_model: Any) -> str:
        args = getattr(target_model, "args", None)
        if args is None and hasattr(target_model, "language_model"):
            args = getattr(target_model.language_model, "args", None)
        value = getattr(args, "model_type", None)
        if value is not None:
            return str(value).lower()
        config = getattr(target_model, "config", None)
        if isinstance(config, dict):
            return str(config.get("model_type", "")).lower()
        return ""

    def supports_model(self, target_model: Any) -> bool:
        if "lfm" not in self.model_type(target_model):
            return False
        return self._has_lfm_text_shape(target_model)

    def _has_lfm_text_shape(self, target_model: Any) -> bool:
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        if not (hasattr(inner, "layers") and hasattr(inner, "embed_tokens")):
            return False
        return any(
            hasattr(layer, "conv") and not getattr(layer, "is_attention_layer", True)
            for layer in inner.layers
        )

    def text_wrapper(self, target_model: Any) -> Any:
        if hasattr(target_model, "model"):
            return target_model
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        raise AttributeError(f"Unsupported target model wrapper: {type(target_model)!r}")

    def text_model(self, target_model: Any) -> Any:
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "model"):
            return wrapper.model
        raise AttributeError(f"Unsupported target text model: {type(wrapper)!r}")

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(self, target_model: Any, hidden_states: mx.array) -> mx.array:
        return self.text_model(target_model).embed_tokens.as_linear(hidden_states)

    def family(self, target_model: Any) -> str:
        return "hybrid_short_conv"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=True,
            supports_kv_trim=True,
            supports_prefix_snapshot=True,
            supports_rotating_cache_snapshot=False,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
        )

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array],
        target_layer_ids: list[int],
    ) -> mx.array:
        selected = [captured_dict[layer_id + 1] for layer_id in target_layer_ids]
        return mx.concatenate(selected, axis=-1)

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        inner = self.text_model(target_model)
        hidden_states = (
            input_embeddings if input_embeddings is not None else inner.embed_tokens(input_ids)
        )
        if cache is None:
            cache = [None] * len(inner.layers)
        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [hidden_states]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: hidden_states} if 0 in capture_layer_ids else {}
        h = hidden_states

        fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
        conv_mask = create_ssm_mask(hidden_states, cache[inner.conv_idx])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            mask = fa_mask if getattr(layer, "is_attention_layer", False) else conv_mask
            h = layer(h, mask=mask, cache=layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h
        normalized = inner.embedding_norm(h)
        logits = self.logits_from_hidden(target_model, normalized)
        return logits, captured

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        if int(verify_ids.shape[1]) <= 0:
            raise ValueError("verify block must contain at least one token")
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def install_speculative_hooks(self, target_model: Any) -> None:
        text_model = self.text_model(target_model)
        if getattr(text_model, "_dflash_speculative_hooks_installed", False):
            return
        for layer in text_model.layers:
            if hasattr(layer, "conv") and not getattr(layer, "is_attention_layer", True):
                _install_short_conv_rollback_hook(layer.conv)
        text_model._dflash_speculative_hooks_installed = True

    def configure_full_attention_split(
        self,
        target_model: Any,
        *,
        enabled: bool,
        chunk_size: int = 8,
    ) -> None:
        return

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
    ) -> list[Any]:
        fa_window = 0 if target_fa_window is None else int(target_fa_window)
        if fa_window < 0:
            raise ValueError("target_fa_window must be >= 0")
        if fa_window > 0 and quantize_kv_cache:
            raise ValueError(
                "target_fa_window does not support quantized target KV cache"
            )
        text_model = self.text_model(target_model)
        caches: list[Any] = []
        for layer in text_model.layers:
            if hasattr(layer, "conv") and not getattr(layer, "is_attention_layer", True):
                if enable_speculative_linear_cache:
                    self.install_speculative_hooks(target_model)
                    kernel_size = int(getattr(layer.conv, "L_cache", 4))
                    caches.append(ShortConvRollbackCache(kernel_size=kernel_size))
                else:
                    caches.append(cache_mod.ArraysCache(size=1))
            else:
                if fa_window > 0:
                    caches.append(cache_mod.RotatingKVCache(max_size=fa_window))
                elif quantize_kv_cache:
                    caches.append(cache_mod.QuantizedKVCache(group_size=64, bits=8))
                else:
                    caches.append(cache_mod.KVCache())
        return caches

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        for cache_entry in cache_entries:
            if hasattr(cache_entry, "arm_rollback"):
                cache_entry.arm_rollback(prefix_len=int(prefix_len))

    def clear_rollback_state(self, cache_entry: Any) -> None:
        if hasattr(cache_entry, "clear_transients"):
            cache_entry.clear_transients()
            return
        if hasattr(cache_entry, "_armed"):
            cache_entry._armed = False
        if hasattr(cache_entry, "_tape"):
            cache_entry._tape = None
        if hasattr(cache_entry, "_snapshot"):
            cache_entry._snapshot = None

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        replay_ns_total = 0
        fully_accepted = acceptance_length == drafted_tokens
        for cache_entry in cache_entries:
            if hasattr(cache_entry, "rollback"):
                if fully_accepted:
                    self.clear_rollback_state(cache_entry)
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

    def cleanup_generation_caches(
        self,
        target_cache: list[Any],
        draft_cache: list[Any],
    ) -> None:
        for cache_entry in target_cache:
            if hasattr(cache_entry, "clear_transients"):
                cache_entry.clear_transients()
        draft_cache.clear()
        target_cache.clear()

# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import time
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

from dflash_mlx.engine.gqa_sdpa import (
    async_per_head_gqa_sdpa,
    grouped_gqa_sdpa,
    per_head_gqa_sdpa,
    repeat_gqa_mask,
)
from dflash_mlx.engine.target_ops import TargetCapabilities
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

_HYBRID_SDPA_EXACT_KV_THRESHOLD = 1024
_TREE_POSITIONS_ATTR = "_dflash_tree_positions"
_TREE_PARENT_IDS_ATTR = "_dflash_tree_parent_ids"
_TREE_ATTENTION_MASK_ATTR = "_dflash_tree_attention_mask"
_TREE_PREFIX_LEN_ATTR = "_dflash_tree_prefix_len"
_TREE_SIZE_ATTR = "_dflash_tree_size"


def _int_attr(obj: Any, name: str) -> int:
    value = getattr(obj, name, 0)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid Qwen target config field {name}: {value!r}") from None


def _required_positive_int_attr(obj: Any, name: str) -> int:
    value = getattr(obj, name, None)
    if value is None:
        raise ValueError(f"Missing Qwen target config field {name}")
    parsed = _int_attr(obj, name)
    if parsed <= 0:
        raise ValueError(f"Invalid Qwen target config field {name}: {value!r}")
    return parsed

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

def _gqa_reshape_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any] = None,
) -> mx.array:
    _, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, kv_len, _ = keys.shape
    if kv_heads <= 0 or query_heads == kv_heads or query_heads % kv_heads != 0:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )
    gqa = query_heads // kv_heads
    has_quantized_cache = cache is not None and hasattr(cache, "bits")
    q_len_i = int(q_len)
    kv_len_i = int(kv_len)
    kv_heads_i = int(kv_heads)
    if (
        has_quantized_cache
        or int(head_dim) != 256
        or queries.dtype not in (mx.bfloat16, mx.float16)
    ):
        return grouped_gqa_sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        )

    use_native_gqa = (q_len_i == 1 and kv_len_i >= 4096) or (
        q_len_i == 4
        and (
            kv_len_i <= 8192
            or (int(gqa) <= 4 and kv_len_i <= 32768)
            or (kv_heads_i >= 4 and kv_len_i <= 16384)
        )
    )
    if use_native_gqa:
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        )

    if q_len_i not in (4, 16):
        return grouped_gqa_sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        )
    if kv_heads_i <= 2 and q_len_i == 16:
        if kv_len_i < 16384:
            return grouped_gqa_sdpa(
                queries,
                keys,
                values,
                cache=cache,
                scale=scale,
                mask=mask,
            )
        grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
        if kv_len_i < 65536:
            return async_per_head_gqa_sdpa(
                queries,
                keys,
                values,
                scale=scale,
                mask=grouped_mask,
                gqa=gqa,
            )
        return per_head_gqa_sdpa(
            queries,
            keys,
            values,
            scale=scale,
            mask=grouped_mask,
            gqa=gqa,
        )
    if kv_heads_i <= 2 and kv_len_i >= 8192:
        grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
        return per_head_gqa_sdpa(
            queries,
            keys,
            values,
            scale=scale,
            mask=grouped_mask,
            gqa=gqa,
        )
    if kv_heads_i >= 4 and kv_len_i >= 16384:
        grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
        return async_per_head_gqa_sdpa(
            queries,
            keys,
            values,
            scale=scale,
            mask=grouped_mask,
            gqa=gqa,
        )
    return grouped_gqa_sdpa(
        queries,
        keys,
        values,
        cache=cache,
        scale=scale,
        mask=mask,
    )


def _as_int_list(values: mx.array) -> list[int]:
    return [int(value) for value in values.tolist()]


def _apply_rope_positions(rope: Any, x: mx.array, positions: mx.array) -> mx.array:
    pos = _as_int_list(positions)
    if not pos:
        return x
    if len(pos) == 1:
        return rope(x, offset=pos[0])
    chunks = [rope(x[:, :, index : index + 1, :], offset=value) for index, value in enumerate(pos)]
    return mx.concatenate(chunks, axis=2)


def _tree_path_indices(parent_ids: list[int], slot_index: int) -> list[int]:
    path: list[int] = []
    cursor = int(slot_index)
    while cursor >= 0:
        path.append(cursor)
        cursor = int(parent_ids[cursor])
    return list(reversed(path))


def _tree_conv_window_indices(
    parent_ids: list[int],
    *,
    conv_kernel_size: int,
) -> mx.array:
    kernel = int(conv_kernel_size)
    keep = kernel - 1
    rows: list[list[int]] = []
    for slot_index in range(len(parent_ids)):
        path = _tree_path_indices(parent_ids, slot_index)
        path_tail = path[-kernel:]
        prefix_need = kernel - len(path_tail)
        prefix_start = keep - prefix_need
        rows.append(
            list(range(prefix_start, keep))
            + [keep + int(path_slot) for path_slot in path_tail]
        )
    return mx.array(rows, dtype=mx.int32)


def _tree_conv1d(
    linear_attn: Any,
    *,
    qkv: mx.array,
    cache: RecurrentRollbackCache,
    parent_ids: list[int],
) -> mx.array:
    batch_size = int(qkv.shape[0])
    conv_dim = int(qkv.shape[-1])
    kernel = int(linear_attn.conv_kernel_size)
    keep = kernel - 1
    prefix_state = cache[0]
    if keep <= 0:
        prefix_state = mx.zeros((batch_size, 0, conv_dim), dtype=qkv.dtype)
    elif prefix_state is None:
        prefix_state = mx.zeros((batch_size, keep, conv_dim), dtype=qkv.dtype)
    cache._dflash_tree_prefix_conv_state = prefix_state
    gather_indices = _tree_conv_window_indices(parent_ids, conv_kernel_size=kernel)
    from dflash_mlx.kernels import tree_depthwise_conv1d

    return tree_depthwise_conv1d(
        qkv,
        prefix_state,
        linear_attn.conv1d.weight,
        getattr(linear_attn.conv1d, "bias", None),
        gather_indices,
    )


def _linear_attn_projections(
    linear_attn: Any,
    inputs: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    if (
        hasattr(linear_attn, "in_proj_qkvz")
        and hasattr(linear_attn, "in_proj_ba")
        and hasattr(linear_attn, "fix_query_key_value_ordering")
    ):
        batch_size, seq_len, _ = inputs.shape
        q, k, v, z, b, a = linear_attn.fix_query_key_value_ordering(
            linear_attn.in_proj_qkvz(inputs),
            linear_attn.in_proj_ba(inputs),
        )
        qkv = mx.concatenate(
            [
                q.reshape(batch_size, seq_len, -1),
                k.reshape(batch_size, seq_len, -1),
                v.reshape(batch_size, seq_len, -1),
            ],
            axis=-1,
        )
        return qkv, z, b, a

    qkv = linear_attn.in_proj_qkv(inputs)
    z_proj = linear_attn.in_proj_z(inputs)
    z = z_proj.reshape(
        inputs.shape[0],
        inputs.shape[1],
        linear_attn.num_v_heads,
        linear_attn.head_v_dim,
    )
    b = linear_attn.in_proj_b(inputs)
    a = linear_attn.in_proj_a(inputs)
    return qkv, z, b, a


def _tree_recurrent_call(
    linear_attn: Any,
    inputs: mx.array,
    *,
    cache: RecurrentRollbackCache,
) -> mx.array:
    from mlx.nn.layers.distributed import sum_gradients

    parent_ids = _as_int_list(getattr(cache, _TREE_PARENT_IDS_ATTR))
    batch_size, tree_size, _ = inputs.shape
    sharding_group = getattr(linear_attn, "sharding_group", None)

    if sharding_group is not None:
        inputs = sum_gradients(sharding_group)(inputs)

    qkv, z, b, a = _linear_attn_projections(linear_attn, inputs)

    conv_out = _tree_conv1d(linear_attn, qkv=qkv, cache=cache, parent_ids=parent_ids)
    q, k, v = [
        tensor.reshape(batch_size, tree_size, heads, dim)
        for tensor, heads, dim in zip(
            mx.split(conv_out, [linear_attn.key_dim, 2 * linear_attn.key_dim], -1),
            [linear_attn.num_k_heads, linear_attn.num_k_heads, linear_attn.num_v_heads],
            [linear_attn.head_k_dim, linear_attn.head_k_dim, linear_attn.head_v_dim],
            strict=True,
        )
    ]

    state = cache[1]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    g = gated_delta_mod.compute_g(linear_attn.A_log, a, linear_attn.dt_bias)
    beta = mx.sigmoid(b)

    if state is None:
        _, _, h_k, d_k = q.shape
        h_v, d_v = v.shape[-2:]
        state = mx.zeros((batch_size, h_v, d_v, d_k), dtype=mx.float32)
    cache._dflash_tree_prefix_state = state

    from dflash_mlx.kernels import gated_delta_tree_tape_kernel

    out, tree_tape = gated_delta_tree_tape_kernel(
        q,
        k,
        v,
        g,
        beta,
        state,
        mx.array(parent_ids, dtype=mx.int32),
    )
    cache._dflash_tree_tape = mx.contiguous(tree_tape)
    cache._dflash_tree_k = mx.contiguous(k)
    cache._dflash_tree_g = mx.contiguous(g)
    cache._dflash_tree_qkv = mx.contiguous(qkv)
    cache.advance(tree_size)
    out = linear_attn.norm(out, z)
    out_flat = out.reshape(batch_size, tree_size, -1)
    out = linear_attn.out_proj(out_flat)

    if sharding_group is not None:
        out = mx.distributed.all_sum(out, group=sharding_group)

    return out


def _tree_attention_call(
    attn: Any,
    x: mx.array,
    *,
    mask: Optional[mx.array],
    cache: Any,
) -> Optional[mx.array]:
    if cache is None or not hasattr(cache, _TREE_POSITIONS_ATTR):
        return None

    batch_size, seq_len, _ = x.shape
    q_proj_output = attn.q_proj(x)
    num_attention_heads = _attention_num_heads(attn)
    num_key_value_heads = _attention_num_kv_heads(attn)
    gate = None
    if _attention_has_gated_q_proj(attn):
        queries, gate = mx.split(
            q_proj_output.reshape(batch_size, seq_len, num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)
    else:
        queries = q_proj_output.reshape(batch_size, seq_len, num_attention_heads, -1)

    keys = attn.k_proj(x)
    values = attn.v_proj(x)

    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    keys = attn.k_norm(keys.reshape(batch_size, seq_len, num_key_value_heads, -1)).transpose(
        0, 2, 1, 3
    )
    values = values.reshape(batch_size, seq_len, num_key_value_heads, -1).transpose(
        0, 2, 1, 3
    )

    positions = getattr(cache, _TREE_POSITIONS_ATTR)
    queries = _apply_rope_positions(attn.rope, queries, positions)
    keys = _apply_rope_positions(attn.rope, keys, positions)
    keys, values = cache.update_and_fetch(keys, values)
    tree_mask = getattr(cache, _TREE_ATTENTION_MASK_ATTR, mask)
    output = _gqa_reshape_sdpa(
        queries,
        keys,
        values,
        cache=cache,
        scale=attn.scale,
        mask=tree_mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    if gate is not None:
        output = output * mx.sigmoid(gate)
    return attn.o_proj(output)


def _set_tree_cache_context(cache_entries: list[Any], tree_inputs: Any) -> None:
    tree_size = int(tree_inputs.token_ids.shape[0])
    prefix_len = int(tree_inputs.attention_mask.shape[1]) - tree_size
    for cache_entry in cache_entries:
        if isinstance(cache_entry, cache_mod.QuantizedKVCache):
            raise NotImplementedError("DDTree target-tree verify does not support quantized KV cache")
        if isinstance(cache_entry, cache_mod.RotatingKVCache):
            raise NotImplementedError("DDTree target-tree verify does not support rotating KV cache")
        setattr(cache_entry, _TREE_POSITIONS_ATTR, tree_inputs.positions)
        setattr(cache_entry, _TREE_PARENT_IDS_ATTR, tree_inputs.parent_ids)
        setattr(cache_entry, _TREE_ATTENTION_MASK_ATTR, tree_inputs.attention_mask)
        setattr(cache_entry, _TREE_PREFIX_LEN_ATTR, prefix_len)
        setattr(cache_entry, _TREE_SIZE_ATTR, tree_size)


def _clear_tree_cache_context(cache_entry: Any) -> None:
    for attr_name in (
        _TREE_POSITIONS_ATTR,
        _TREE_PARENT_IDS_ATTR,
        _TREE_ATTENTION_MASK_ATTR,
        _TREE_PREFIX_LEN_ATTR,
        _TREE_SIZE_ATTR,
        "_dflash_tree_prefix_conv_state",
        "_dflash_tree_prefix_state",
        "_dflash_tree_states",
        "_dflash_tree_tape",
        "_dflash_tree_k",
        "_dflash_tree_g",
        "_dflash_tree_qkv",
    ):
        if hasattr(cache_entry, attr_name):
            delattr(cache_entry, attr_name)


def _commit_kv_tree_path(cache_entry: Any, accepted_tree_indices: list[int]) -> None:
    if getattr(cache_entry, "keys", None) is None:
        _clear_tree_cache_context(cache_entry)
        return
    prefix_len = int(getattr(cache_entry, _TREE_PREFIX_LEN_ATTR))
    tree_size = int(getattr(cache_entry, _TREE_SIZE_ATTR))
    if int(getattr(cache_entry, "offset", 0)) < prefix_len + tree_size:
        raise RuntimeError("DDTree KV cache did not append the full tree")
    gather_indices = list(range(prefix_len)) + [
        prefix_len + int(slot_index) for slot_index in accepted_tree_indices
    ]
    gather = mx.array(gather_indices, dtype=mx.int32)
    cache_entry.keys = mx.take(cache_entry.keys[..., : prefix_len + tree_size, :], gather, axis=2)
    cache_entry.values = mx.take(cache_entry.values[..., : prefix_len + tree_size, :], gather, axis=2)
    cache_entry.offset = len(gather_indices)
    _clear_tree_cache_context(cache_entry)


def _commit_recurrent_tree_path(
    cache_entry: RecurrentRollbackCache,
    accepted_tree_indices: list[int],
) -> None:
    tree_states = getattr(cache_entry, "_dflash_tree_states", None)
    tree_tape = getattr(cache_entry, "_dflash_tree_tape", None)
    tree_k = getattr(cache_entry, "_dflash_tree_k", None)
    tree_g = getattr(cache_entry, "_dflash_tree_g", None)
    tree_qkv = getattr(cache_entry, "_dflash_tree_qkv", None)
    if tree_qkv is None or (tree_states is None and (tree_tape is None or tree_k is None or tree_g is None)):
        _clear_tree_cache_context(cache_entry)
        raise RuntimeError("DDTree recurrent cache missing tree intermediates")

    last_slot = int(accepted_tree_indices[-1])
    if tree_states is not None:
        cache_entry[1] = mx.contiguous(tree_states[:, last_slot, ...])
    else:
        from dflash_mlx.kernels import tape_replay_kernel

        gather = mx.array([int(slot_index) for slot_index in accepted_tree_indices], dtype=mx.int32)
        prefix_state = getattr(cache_entry, "_dflash_tree_prefix_state", None)
        if prefix_state is None:
            prefix_state = cache_entry[1]
        if prefix_state is None:
            _clear_tree_cache_context(cache_entry)
            raise RuntimeError("DDTree recurrent cache missing prefix state")
        accepted_tape = mx.take(tree_tape, gather, axis=1)
        accepted_k = mx.take(tree_k, gather, axis=1)
        accepted_g = mx.take(tree_g, gather, axis=1)
        cache_entry[1] = tape_replay_kernel(
            accepted_tape,
            accepted_k,
            accepted_g,
            prefix_state,
            None,
        )

    keep = int(cache_entry.conv_kernel_size) - 1
    if keep > 0:
        prefix_state = getattr(cache_entry, "_dflash_tree_prefix_conv_state", None)
        if prefix_state is None:
            prefix_state = mx.zeros(
                (tree_qkv.shape[0], keep, tree_qkv.shape[-1]),
                dtype=tree_qkv.dtype,
            )
        accepted_qkv = mx.take(
            tree_qkv,
            mx.array([int(slot_index) for slot_index in accepted_tree_indices], dtype=mx.int32),
            axis=1,
        )
        conv_source = mx.concatenate([prefix_state, accepted_qkv], axis=1)
        cache_entry[0] = mx.contiguous(conv_source[:, -keep:, :])
    else:
        cache_entry[0] = None

    _clear_tree_cache_context(cache_entry)


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
        if isinstance(cache, RecurrentRollbackCache) and hasattr(cache, _TREE_PARENT_IDS_ATTR):
            return _tree_recurrent_call(self, inputs, cache=cache)
        if not isinstance(cache, RecurrentRollbackCache) or not getattr(cache, "_armed", False):
            return original_call(self, inputs, mask=mask, cache=cache)

        from mlx.nn.layers.distributed import sum_gradients

        B, S, _ = inputs.shape
        sharding_group = getattr(self, "sharding_group", None)

        if sharding_group is not None:
            inputs = sum_gradients(sharding_group)(inputs)

        qkv, z, b, a = _linear_attn_projections(self, inputs)

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
        cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :])
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

        cache[1] = mx.contiguous(state)
        cache.advance(S)
        out = self.norm(out, z)
        out_flat = out.reshape(B, S, -1)
        out = self.out_proj(out_flat)

        if sharding_group is not None:
            out = mx.distributed.all_sum(out, group=sharding_group)

        return out

    cls.__call__ = speculative_call
    cls._dflash_speculative_call_installed = True

def _install_full_attention_gqa_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_full_attention_gqa_installed", False):
        return

    original_call = cls.__call__

    def attention_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        tree_output = _tree_attention_call(self, x, mask=mask, cache=cache)
        if tree_output is not None:
            return tree_output
        cached_prefix_len = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0
        can_route_gqa = (
            cache is not None
            and not isinstance(cache, cache_mod.QuantizedKVCache)
            and cached_prefix_len >= _HYBRID_SDPA_EXACT_KV_THRESHOLD
            and (mask is None or mask == "causal" or isinstance(mask, mx.array))
            and 0 < int(x.shape[1]) <= 16
        )
        if not can_route_gqa:
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
        can_use_gqa_fast_path = (
            queries.dtype in (mx.bfloat16, mx.float16)
            and int(queries.shape[-1]) in (128, 256)
            and int(values.shape[-1]) in (128, 256)
        )
        if not can_use_gqa_fast_path:
            return original_call(self, x, mask=mask, cache=cache)

        queries = self.rope(queries, offset=cached_prefix_len)
        keys = self.rope(keys, offset=cached_prefix_len)
        keys, values = cache.update_and_fetch(keys, values)
        output = _gqa_reshape_sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gated_output = output * mx.sigmoid(gate)
        return self.o_proj(gated_output)

    cls.__call__ = attention_call
    cls._dflash_full_attention_gqa_installed = True

class QwenGdnTargetOps:
    backend_name = "qwen_gdn"

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
        model_type = self.model_type(target_model)
        if "qwen" in model_type:
            return self._has_qwen_text_shape(target_model)
        return False

    def _has_qwen_text_shape(self, target_model: Any) -> bool:
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        return hasattr(inner, "layers") and hasattr(inner, "embed_tokens")

    def text_wrapper(self, target_model: Any) -> Any:
        language_model = getattr(target_model, "language_model", None)
        if language_model is not None and hasattr(language_model, "model"):
            return language_model
        if hasattr(target_model, "model"):
            return target_model
        raise AttributeError(f"Unsupported target model wrapper: {type(target_model)!r}")

    def text_model(self, target_model: Any) -> Any:
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "model"):
            return wrapper.model
        raise AttributeError(f"Unsupported target text model: {type(wrapper)!r}")

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(self, target_model: Any, hidden_states: mx.array) -> mx.array:
        wrapper = self.text_wrapper(target_model)
        if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True):
            return wrapper.model.embed_tokens.as_linear(hidden_states)
        return wrapper.lm_head(hidden_states)

    def family(self, target_model: Any) -> str:
        inner = self.text_model(target_model)
        has_linear = any(
            hasattr(layer, "linear_attn") or getattr(layer, "is_linear", False)
            for layer in inner.layers
        )
        return "hybrid_gdn" if has_linear else "pure_attention"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        has_recurrent = self.family(target_model) == "hybrid_gdn"
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=has_recurrent,
            supports_kv_trim=True,
            supports_prefix_snapshot=True,
            supports_rotating_cache_snapshot=False,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
            supports_verify_linear=self._supports_verify_linear(target_model),
            supports_tree_verify=True,
            # The DFlash draft is trained with full attention over context
            # features; the engine gates engagement on draft_full_context_min_ctx.
            supports_full_context_draft_layers=True,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        return all(
            not isinstance(
                cache_entry,
                (cache_mod.QuantizedKVCache, cache_mod.RotatingKVCache),
            )
            for cache_entry in cache_entries
        )

    def _supports_verify_linear(self, target_model: Any) -> bool:
        wrapper = self.text_wrapper(target_model)
        args = getattr(wrapper, "args", None)
        num_experts = _int_attr(args, "num_experts")
        num_layers = _int_attr(args, "num_hidden_layers")
        if num_layers <= 0:
            num_layers = len(getattr(self.text_model(target_model), "layers", []))
        if num_experts > 0:
            hidden_size = _required_positive_int_attr(args, "hidden_size")
            num_heads = _required_positive_int_attr(args, "num_attention_heads")
            num_kv_heads = _required_positive_int_attr(args, "num_key_value_heads")
            return not (
                num_layers == 40
                and hidden_size == 2048
                and num_heads == 16
                and num_kv_heads == 2
            )
        return num_layers >= 40

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
        logits_last_only: bool = False,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        inner = self.text_model(target_model)
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
        if logits_last_only and isinstance(captured, dict):
            captured[-1] = normalized
        logits_hidden = normalized[:, -1:, :] if logits_last_only else normalized
        logits = self.logits_from_hidden(target_model, logits_hidden)
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

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        tree_size = int(tree_inputs.token_ids.shape[0])
        if tree_size <= 0:
            raise ValueError("DDTree target tree must contain at least the root slot")
        self.install_speculative_hooks(target_model)
        _set_tree_cache_context(target_cache, tree_inputs)
        try:
            return self.forward_with_hidden_capture(
                target_model,
                input_ids=tree_inputs.token_ids[None],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
        except Exception as exc:
            for cache_entry in target_cache:
                _clear_tree_cache_context(cache_entry)
            raise RuntimeError("DDTree target-tree verify failed") from exc

    def restore_after_tree_acceptance(
        self,
        cache_entries: list[Any],
        *,
        accepted_tree_indices: list[int],
    ) -> int:
        if not accepted_tree_indices:
            raise ValueError("accepted_tree_indices must not be empty")
        replay_start_ns = time.perf_counter_ns()
        for cache_entry in cache_entries:
            if isinstance(cache_entry, RecurrentRollbackCache):
                _commit_recurrent_tree_path(cache_entry, accepted_tree_indices)
            elif hasattr(cache_entry, "keys") and hasattr(cache_entry, "values"):
                _commit_kv_tree_path(cache_entry, accepted_tree_indices)
            else:
                _clear_tree_cache_context(cache_entry)
                raise NotImplementedError(
                    f"DDTree cache commit unsupported for {type(cache_entry).__name__}"
                )
        return time.perf_counter_ns() - replay_start_ns

    def install_speculative_hooks(self, target_model: Any) -> None:
        text_model = self.text_model(target_model)
        if getattr(text_model, "_dflash_speculative_hooks_installed", False):
            return
        if self.family(target_model) == "pure_attention":
            for layer in text_model.layers:
                if hasattr(layer, "self_attn"):
                    _install_full_attention_gqa_hook(layer.self_attn)
            text_model._dflash_speculative_hooks_installed = True
            return
        for layer in text_model.layers:
            if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
                _install_speculative_linear_cache_hook(layer.linear_attn)
            elif not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
                _install_full_attention_gqa_hook(layer.self_attn)
        text_model._dflash_speculative_hooks_installed = True

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
            if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
                if enable_speculative_linear_cache:
                    self.install_speculative_hooks(target_model)
                    conv_kernel_size = int(getattr(layer.linear_attn, "conv_kernel_size", 4))
                    caches.append(
                        RecurrentRollbackCache(size=2, conv_kernel_size=conv_kernel_size)
                    )
                else:
                    caches.append(cache_mod.ArraysCache(size=2))
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
        if hasattr(cache_entry, "_tape_k"):
            cache_entry._tape_k = None
        if hasattr(cache_entry, "_tape_g"):
            cache_entry._tape_g = None
        if hasattr(cache_entry, "_tape_qkv"):
            cache_entry._tape_qkv = None
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

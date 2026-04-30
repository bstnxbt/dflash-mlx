# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from dflash_mlx.internal_debug import (
    verify_include as _debug_verify_include,
    verify_max_n as _debug_verify_max_n,
    verify_qmm_enabled as _debug_verify_qmm_enabled,
)
from dflash_mlx.verify_qmm import (
    verify_matmul,
    _auto_variant,
    _build_kernel_mma2big,
    _build_kernel_mma2big_8bit,
    _build_kernel_mma2big_pipe,
    _build_kernel_mma2big_pipe_8bit,
)

_VERIFY_MAX_N_DEFAULT = 100_000

_PROJ_TAGS = {
    "mlp.gate_proj":        "mlp_gate",
    "mlp.up_proj":          "mlp_up",
    "mlp.down_proj":        "mlp_down",
    "self_attn.q_proj":     "attn_q",
    "self_attn.k_proj":     "attn_k",
    "self_attn.v_proj":     "attn_v",
    "self_attn.o_proj":     "attn_o",
    "linear_attn.in_proj_qkv": "gdn_qkv",
    "linear_attn.in_proj_z":   "gdn_z",
    "linear_attn.out_proj":    "gdn_o",
}

def _path_tag(path: str) -> str:
    for suffix, tag in _PROJ_TAGS.items():
        if path.endswith(suffix):
            return tag
    return "other"

def is_verify_eligible(ql: nn.QuantizedLinear, path: str = "") -> bool:
    if not isinstance(ql, nn.QuantizedLinear):
        return False
    if getattr(ql, "bits", None) not in (4, 8):
        return False
    if getattr(ql, "group_size", None) not in (32, 64, 128):
        return False
    if getattr(ql, "mode", "affine") != "affine":
        return False
    w = ql.weight
    N = int(w.shape[0])
    K = int(w.shape[1]) * (32 // ql.bits)
    if N % 32 != 0 or K % 32 != 0:
        return False
    if N >= _debug_verify_max_n(_VERIFY_MAX_N_DEFAULT):
        return False
    include = _debug_verify_include()
    if include not in ("", "all"):
        tag = _path_tag(path)
        allowed = {s.strip() for s in include.split(",") if s.strip()}
        if "mlp" in allowed:
            allowed.update({"mlp_gate", "mlp_up", "mlp_down"})
        if "attn" in allowed:
            allowed.update({"attn_q", "attn_k", "attn_v", "attn_o"})
        if "gdn" in allowed:
            allowed.update({"gdn_qkv", "gdn_z", "gdn_o"})
        if tag not in allowed:
            return False
    return True

class VerifyQuantizedLinear(nn.QuantizedLinear):

    @classmethod
    def from_quantized(
        cls,
        ql: nn.QuantizedLinear,
        *,
        enable_qmm: Optional[bool] = None,
    ) -> "VerifyQuantizedLinear":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.group_size = ql.group_size
        obj.bits = ql.bits
        obj.mode = getattr(ql, "mode", "affine")
        obj.weight = ql.weight
        obj.scales = ql.scales
        if getattr(ql, "biases", None) is not None:
            obj.biases = ql.biases
        if "bias" in ql:
            obj.bias = ql.bias

        object.__setattr__(obj, "_call_fn", _build_dispatch(obj, enable_qmm=enable_qmm))

        obj.freeze()
        return obj

    def __call__(self, x: mx.array) -> mx.array:
        return self._call_fn(x)

def _build_dispatch(
    obj: "VerifyQuantizedLinear",
    *,
    enable_qmm: Optional[bool] = None,
):
    w = obj.weight
    s = obj.scales
    b = getattr(obj, "biases", None)
    gs = obj.group_size
    bits = obj.bits
    mode = obj.mode
    has_bias = "bias" in obj
    bias = obj.bias if has_bias else None

    N = int(w.shape[0])
    K = int(w.shape[1]) * (32 // bits)

    qmm_enabled = (
        _debug_verify_qmm_enabled()
        if enable_qmm is None
        else bool(enable_qmm)
    )

    if qmm_enabled and N % 32 == 0 and K % 32 == 0:
        variant, auto_kp = _auto_variant(K, N)
        K_PARTS = auto_kp
        if variant == "mma2big_pipe" and K % (32 * K_PARTS) != 0:
            variant = "mma2big"
            K_PARTS = 1

        w_c = mx.contiguous(w)
        s_c = mx.contiguous(s)
        b_c = mx.contiguous(b) if b is not None else b
        mx.eval(w_c, s_c)
        if b_c is not None:
            mx.eval(b_c)

        _kern_fn = (
            (_build_kernel_mma2big_pipe_8bit if bits == 8 else _build_kernel_mma2big_pipe)
            if variant == "mma2big_pipe" else
            (_build_kernel_mma2big_8bit if bits == 8 else _build_kernel_mma2big)
        )
        kern_bf16 = _kern_fn(gs, mx.bfloat16)
        kern_fp16 = _kern_fn(gs, mx.float16)

        if variant == "mma2big_pipe":
            def _run(x2: mx.array, kern) -> mx.array:
                (partials,) = kern(
                    inputs=[x2, w_c, s_c, b_c, 16, K, N, K_PARTS],
                    template=[("T", x2.dtype)],
                    grid=(64, N // 32, K_PARTS),
                    threadgroup=(64, 1, 1),
                    output_shapes=[(K_PARTS, 16, N)],
                    output_dtypes=[mx.float32],
                )
                return partials.sum(axis=0).astype(x2.dtype)
        else:
            def _run(x2: mx.array, kern) -> mx.array:
                (y,) = kern(
                    inputs=[x2, w_c, s_c, b_c, 16, K, N],
                    template=[("T", x2.dtype)],
                    grid=(64, N // 32, 1),
                    threadgroup=(64, 1, 1),
                    output_shapes=[(16, N)],
                    output_dtypes=[x2.dtype],
                )
                return y

        if has_bias:
            def call(x: mx.array) -> mx.array:
                orig = x.shape
                m = 1
                for d in orig[:-1]:
                    m *= d
                if m == 16:
                    x2 = mx.contiguous(x.reshape(16, orig[-1]))
                    dtype = x2.dtype
                    if dtype == mx.bfloat16:
                        y = _run(x2, kern_bf16)
                    elif dtype == mx.float16:
                        y = _run(x2, kern_fp16)
                    else:
                        y = mx.quantized_matmul(x, w_c, scales=s_c, biases=b_c,
                                                transpose=True, group_size=gs, bits=bits, mode=mode)
                    return y.reshape(*orig[:-1], N) + bias
                y = mx.quantized_matmul(x, w_c, scales=s_c, biases=b_c,
                                        transpose=True, group_size=gs, bits=bits, mode=mode)
                return y + bias
        else:
            def call(x: mx.array) -> mx.array:
                orig = x.shape
                m = 1
                for d in orig[:-1]:
                    m *= d
                if m == 16:
                    x2 = mx.contiguous(x.reshape(16, orig[-1]))
                    dtype = x2.dtype
                    if dtype == mx.bfloat16:
                        return _run(x2, kern_bf16).reshape(*orig[:-1], N)
                    if dtype == mx.float16:
                        return _run(x2, kern_fp16).reshape(*orig[:-1], N)
                return mx.quantized_matmul(x, w_c, scales=s_c, biases=b_c,
                                           transpose=True, group_size=gs, bits=bits, mode=mode)
        return call

    if has_bias:
        def call(x):
            m = 1
            for d in x.shape[:-1]:
                m *= d
            if m == 16:
                y = verify_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)
            else:
                y = mx.quantized_matmul(x, w, scales=s, biases=b,
                                        transpose=True, group_size=gs, bits=bits, mode=mode)
            return y + bias
    else:
        def call(x):
            m = 1
            for d in x.shape[:-1]:
                m *= d
            if m == 16:
                return verify_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)
            return mx.quantized_matmul(x, w, scales=s, biases=b,
                                       transpose=True, group_size=gs, bits=bits, mode=mode)
    return call

def prewarm_verify_kernels(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    seen: set[tuple] = set()
    warmed = 0
    for _, m in tree_flatten(model.leaf_modules()):
        if not isinstance(m, VerifyQuantizedLinear):
            continue
        K = int(m.weight.shape[1]) * (32 // m.bits)
        N = int(m.weight.shape[0])
        key = (K, N, m.bits, m.group_size)
        if key in seen:
            continue
        seen.add(key)
        dummy = mx.zeros((1, 16, K), dtype=mx.bfloat16)
        mx.eval(m(dummy))
        warmed += 1
    return warmed

def install_verify_linears(
    model: nn.Module,
    *,
    predicate: Optional[Callable[[str, nn.QuantizedLinear], bool]] = None,
    enable_qmm: Optional[bool] = None,
) -> int:
    if predicate is None:
        predicate = lambda path, m: is_verify_eligible(m, path=path)

    count = 0

    def _maybe_swap(path, m):
        nonlocal count
        if isinstance(m, VerifyQuantizedLinear):
            return m
        if isinstance(m, nn.QuantizedLinear) and predicate(path, m):
            count += 1
            return VerifyQuantizedLinear.from_quantized(m, enable_qmm=enable_qmm)
        return m

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_maybe_swap, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return count

def uninstall_verify_linears(model: nn.Module) -> int:
    count = 0

    def _maybe_unswap(path, m):
        nonlocal count
        if isinstance(m, VerifyQuantizedLinear):
            ql = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
            nn.Module.__init__(ql)
            ql.group_size = m.group_size
            ql.bits = m.bits
            ql.mode = m.mode
            ql.weight = m.weight
            ql.scales = m.scales
            if getattr(m, "biases", None) is not None:
                ql.biases = m.biases
            if "bias" in m:
                ql.bias = m.bias
            ql.freeze()
            count += 1
            return ql
        return m

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_maybe_unswap, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return count

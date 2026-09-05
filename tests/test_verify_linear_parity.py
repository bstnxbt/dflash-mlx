# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import pytest

os.environ.setdefault("DFLASH_VERIFY_QMM", "1")

from dflash_mlx.verify_qmm import verify_matmul
from dflash_mlx.verify_linear import (
    VerifyQuantizedLinear,
    is_verify_eligible,
    install_verify_linears,
)

@pytest.fixture(scope="module")
def small_ql():
    gs, bits = 64, 4
    in_dims, out_dims = 512, 1024
    lin = nn.Linear(in_dims, out_dims, bias=False)
    lin.weight = mx.random.normal((out_dims, in_dims)).astype(mx.bfloat16) * 0.1
    ql = nn.QuantizedLinear.from_linear(lin, group_size=gs, bits=bits)
    return ql

def test_eligibility_basic(small_ql):
    assert is_verify_eligible(small_ql)

def test_eligibility_rejects_large_N():
    gs, bits = 64, 4
    lin = nn.Linear(512, 150_000, bias=False)
    lin.weight = mx.random.normal((150_000, 512)).astype(mx.bfloat16) * 0.01
    ql = nn.QuantizedLinear.from_linear(lin, group_size=gs, bits=bits)
    assert not is_verify_eligible(ql)

@pytest.mark.parametrize("M", [1, 8, 32])
def test_parity_non_verify(small_ql, M):
    verify = VerifyQuantizedLinear.from_quantized(small_ql)
    x = mx.random.normal((M, 512)).astype(mx.bfloat16) * 0.5
    y_ref = small_ql(x)
    y_verify = verify(x)
    mx.eval(y_ref, y_verify)
    assert mx.allclose(y_ref, y_verify, atol=0, rtol=0).item(), \
        "Non-verify path must be bit-identical (both route through stock qmm)"

def test_parity_verify_M16(small_ql):
    verify = VerifyQuantizedLinear.from_quantized(small_ql)
    x = mx.random.normal((16, 512)).astype(mx.bfloat16) * 0.5
    y_direct = verify_matmul(
        x, small_ql.weight, small_ql.scales, small_ql.biases,
        transpose=True, group_size=small_ql.group_size, bits=small_ql.bits,
    )
    y_verify = verify(x)
    mx.eval(y_direct, y_verify)
    assert mx.allclose(y_direct, y_verify, atol=0, rtol=0).item()

def test_m16_wrapper_accepts_n16_not_n32():
    ql = nn.QuantizedLinear.from_linear(_mk_linear(512, 16), group_size=64, bits=4)
    verify = VerifyQuantizedLinear.from_quantized(ql)
    x = mx.random.normal((16, 512)).astype(mx.bfloat16) * 0.5
    y_direct = verify_matmul(
        x, ql.weight, ql.scales, ql.biases,
        transpose=True, group_size=ql.group_size, bits=ql.bits,
    )
    y_verify = verify(x)
    mx.eval(y_direct, y_verify)
    assert mx.allclose(y_direct, y_verify, atol=0, rtol=0).item()

@pytest.mark.parametrize("variant", ["combo_ktmpl", "super_tree_fp16_ktmpl"])
def test_m16_wrapper_honors_forced_variant(monkeypatch, variant):
    monkeypatch.setenv("DFLASH_VERIFY_VARIANT", variant)
    ql = nn.QuantizedLinear.from_linear(_mk_linear(512, 1024), group_size=64, bits=4)
    verify = VerifyQuantizedLinear.from_quantized(ql)
    x = mx.random.normal((16, 512)).astype(mx.bfloat16) * 0.5
    y_direct = verify_matmul(
        x, ql.weight, ql.scales, ql.biases,
        transpose=True, group_size=ql.group_size, bits=ql.bits,
    )
    y_verify = verify(x)
    mx.eval(y_direct, y_verify)
    assert mx.allclose(y_direct, y_verify, atol=0, rtol=0).item()

@pytest.mark.parametrize("variant", ["combo_ktmpl", "super_tree_fp16_ktmpl"])
def test_m16_forced_variant_matches_stock(monkeypatch, variant):
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")
    monkeypatch.setenv("DFLASH_VERIFY_VARIANT", variant)
    ql = nn.QuantizedLinear.from_linear(_mk_linear(512, 1024), group_size=64, bits=4)
    verify = VerifyQuantizedLinear.from_quantized(ql)
    x = mx.random.normal((16, 512)).astype(mx.bfloat16) * 0.5
    y_stock = mx.quantized_matmul(
        x, ql.weight, scales=ql.scales, biases=ql.biases,
        transpose=True, group_size=ql.group_size, bits=ql.bits,
    )
    y_direct = verify_matmul(
        x, ql.weight, ql.scales, ql.biases,
        transpose=True, group_size=ql.group_size, bits=ql.bits,
    )
    y_verify = verify(x)
    mx.eval(y_stock, y_direct, y_verify)
    y_stock_f = y_stock.astype(mx.float32)
    y_direct_f = y_direct.astype(mx.float32)
    y_verify_f = y_verify.astype(mx.float32)
    direct_abs = float(mx.max(mx.abs(y_stock_f - y_direct_f)).item())
    wrapper_abs = float(mx.max(mx.abs(y_stock_f - y_verify_f)).item())
    ref_max = float(mx.max(mx.abs(y_stock_f)).item())
    assert direct_abs <= 6e-2
    assert wrapper_abs <= 6e-2
    assert direct_abs / (ref_max + 1e-3) <= 5e-2
    assert wrapper_abs / (ref_max + 1e-3) <= 5e-2

def test_parity_verify_M4_product_path(small_ql):
    verify = VerifyQuantizedLinear.from_quantized(small_ql)
    x = mx.random.normal((4, 512)).astype(mx.bfloat16) * 0.5
    y_direct = verify_matmul(
        x, small_ql.weight, small_ql.scales, small_ql.biases,
        transpose=True, group_size=small_ql.group_size, bits=small_ql.bits,
    )
    y_verify = verify(x)
    mx.eval(y_direct, y_verify)
    assert mx.allclose(y_direct, y_verify, atol=0, rtol=0).item()

def test_m4_wrapper_accepts_n4_not_n32():
    ql = nn.QuantizedLinear.from_linear(_mk_linear(64, 12), group_size=64, bits=4)
    assert is_verify_eligible(ql)
    verify = VerifyQuantizedLinear.from_quantized(ql)
    x = mx.random.normal((4, 64)).astype(mx.bfloat16) * 0.5
    y_direct = verify_matmul(
        x, ql.weight, ql.scales, ql.biases,
        transpose=True, group_size=ql.group_size, bits=ql.bits,
    )
    y_verify = verify(x)
    mx.eval(y_direct, y_verify)
    assert mx.allclose(y_direct, y_verify, atol=0, rtol=0).item()

def test_install_verify_linears_swaps_eligible_modules(small_ql):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_a = nn.QuantizedLinear.from_linear(
                _mk_linear(512, 1024), group_size=64, bits=4)
            self.proj_b = nn.QuantizedLinear.from_linear(
                _mk_linear(512, 1024), group_size=64, bits=4)
            self.proj_bad = nn.QuantizedLinear.from_linear(
                _mk_linear(512, 150_000), group_size=64, bits=4)

    m = Tiny()
    n = install_verify_linears(m)
    assert n == 2, f"expected 2 swaps, got {n}"
    assert isinstance(m.proj_a, VerifyQuantizedLinear)
    assert isinstance(m.proj_b, VerifyQuantizedLinear)
    assert not isinstance(m.proj_bad, VerifyQuantizedLinear)

    x = mx.random.normal((16, 512)).astype(mx.bfloat16) * 0.5
    y = m.proj_a(x); mx.eval(y); assert y.shape == (16, 1024)

def _mk_linear(in_dims, out_dims):
    lin = nn.Linear(in_dims, out_dims, bias=False)
    lin.weight = mx.random.normal((out_dims, in_dims)).astype(mx.bfloat16) * 0.1
    return lin


def test_prewarm_visits_modules_and_deduplicates_shapes(monkeypatch):
    from dflash_mlx.verify_linear import prewarm_verify_kernels

    layers = [VerifyQuantizedLinear.from_quantized(
        nn.QuantizedLinear(64, 32, bias=False), enable_qmm=False) for _ in range(2)]
    model = nn.Sequential(*layers)
    calls = []

    def call(self, x):
        calls.append(x.shape)
        return mx.zeros((*x.shape[:-1], 32), dtype=x.dtype)

    monkeypatch.setattr(VerifyQuantizedLinear, "__call__", call)
    count = prewarm_verify_kernels(model)
    assert count == 2
    assert calls == [(1, 16, 64), (1, 4, 64)]

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
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

def test_swap_and_unswap(small_ql):
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
    from dflash_mlx.verify_linear import uninstall_verify_linears
    n = install_verify_linears(m)
    assert n == 2, f"expected 2 swaps, got {n}"
    assert isinstance(m.proj_a, VerifyQuantizedLinear)
    assert isinstance(m.proj_b, VerifyQuantizedLinear)
    assert not isinstance(m.proj_bad, VerifyQuantizedLinear)

    x = mx.random.normal((16, 512)).astype(mx.bfloat16) * 0.5
    y = m.proj_a(x); mx.eval(y); assert y.shape == (16, 1024)
    n2 = uninstall_verify_linears(m)
    assert n2 == 2

def _mk_linear(in_dims, out_dims):
    lin = nn.Linear(in_dims, out_dims, bias=False)
    lin.weight = mx.random.normal((out_dims, in_dims)).astype(mx.bfloat16) * 0.1
    return lin

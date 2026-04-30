# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import itertools
import os

import mlx.core as mx
import numpy as np
import pytest

from dflash_mlx.verify_qmm import is_enabled, verify_matmul

GROUP_SIZE = 64
BITS = 4

REAL_MLP_SHAPES = [

    ("gate_proj", 16, 5120, 17408),
    ("up_proj",   16, 5120, 17408),
    ("down_proj", 16, 17408, 5120),
]

def _quantize_ref(w_fp, gs, bits):
    return mx.quantize(w_fp, group_size=gs, bits=bits)

def _gen_random_shapes(n=30, seed=0xDF1A):
    rng = np.random.default_rng(seed)
    shapes = []
    M_levels = [1, 4, 8, 12, 15, 16]
    for i in range(n):
        M = M_levels[i % len(M_levels)]

        K = int(rng.integers(2, 40)) * GROUP_SIZE
        N = int(rng.integers(1, 64)) * 8
        shapes.append((f"rnd{i:02d}", M, K, N))
    return shapes

def _run_case(name, M, K, N, dtype, gs):
    rng = np.random.default_rng(hash((name, M, K, N, gs)) & 0xFFFF)
    scale = 1.0 / np.sqrt(max(K, 1))
    x_np = (rng.standard_normal((M, K)) * scale).astype(np.float32)
    w_np = (rng.standard_normal((N, K)) * scale).astype(np.float32)

    x = mx.array(x_np).astype(dtype)
    w_fp = mx.array(w_np).astype(dtype)
    w_q, scales, biases = _quantize_ref(w_fp, gs, BITS)

    y_ref = mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases,
        transpose=True, group_size=gs, bits=BITS,
    )
    mx.eval(y_ref)

    y_verify = verify_matmul(
        x, w_q, scales, biases,
        transpose=True, group_size=gs, bits=BITS,
    )
    mx.eval(y_verify)

    y_ref_np = np.array(y_ref.astype(mx.float32))
    y_verify_np = np.array(y_verify.astype(mx.float32))

    max_abs = float(np.max(np.abs(y_verify_np - y_ref_np)))
    max_rel = float(max_abs / (np.max(np.abs(y_ref_np)) + 1e-3))

    return max_abs, max_rel, y_ref_np.shape

@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("name,M,K,N", REAL_MLP_SHAPES)
def test_mlp_real_shapes(name, M, K, N, dtype):
    abs_tol = 8e-3 if dtype == mx.bfloat16 else 4e-3
    rel_tol = 2e-2
    max_abs, max_rel, shape = _run_case(name, M, K, N, dtype, GROUP_SIZE)
    assert max_abs <= abs_tol, f"{name}[{dtype}] max_abs={max_abs:.4g} > {abs_tol}"
    assert max_rel <= rel_tol, f"{name}[{dtype}] max_rel={max_rel:.4g} > {rel_tol}"

@pytest.mark.parametrize("name,M,K,N", _gen_random_shapes(30))
def test_random_shapes(name, M, K, N):
    abs_tol = 8e-3
    rel_tol = 2e-2
    max_abs, max_rel, _ = _run_case(name, M, K, N, mx.bfloat16, GROUP_SIZE)
    assert max_abs <= abs_tol, f"{name} M={M} K={K} N={N} abs={max_abs:.4g}"
    assert max_rel <= rel_tol, f"{name} M={M} K={K} N={N} rel={max_rel:.4g}"

def test_stub_mode_reports_identity_when_disabled():
    if is_enabled():
        pytest.skip("DFLASH_VERIFY_QMM=1, skip stub sanity")
    max_abs, max_rel, _ = _run_case("stub_sanity", 16, 5120, 8, mx.bfloat16, GROUP_SIZE)
    assert max_abs == 0.0, "Stub path must return exact stock output, got delta"

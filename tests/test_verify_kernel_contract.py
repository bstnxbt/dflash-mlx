# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Numeric contract gate for the default-on verify kernels.

The product contract is byte-identical greedy output vs the stock 4-bit
target. The verify kernels are not bitwise-equal to stock qmm (different
reduction order), so this gate pins the properties that keep the contract
safe and turns silent numeric drift — an MLX/Metal upgrade or a kernel
edit — into a loud failure:

1. determinism: repeated calls return identical bits;
2. deviation bound: max |kernel - stock| stays under a pinned per-shape
   ceiling (measured headroom over the current value);
3. margin-certified argmax: every row whose stock top1-top2 margin exceeds
   twice the deviation must keep the stock argmax (a flip there means the
   deviation measurement itself is wrong);
4. golden fingerprints: per-GPU-profile hashes of the kernel output bytes.

A fingerprint mismatch does not necessarily mean broken output — it means
the kernel's numeric behavior changed and end-to-end GPU parity must be
re-validated before shipping. After re-validating, regenerate with
DFLASH_REGEN_KERNEL_GOLDENS=1 and update _GOLDENS.
"""

from __future__ import annotations

import hashlib
import os

import mlx.core as mx
import numpy as np
import pytest

from dflash_mlx.verify_qmm import verify_matmul

GROUP_SIZE = 64
BITS = 4
DTYPE = mx.bfloat16
SEEDS = (0xA1, 0xB2, 0xC3)

PROD_SHAPES = [
    ("m16_gate_up", 16, 5120, 17408),
    ("m16_down", 16, 17408, 5120),
    ("m4_gate_up", 4, 5120, 17408),
    ("m4_down", 4, 17408, 5120),
]

DEV_BOUNDS = {
    "m16_gate_up": 1e-3,
    "m16_down": 2e-2,
    "m4_gate_up": 2e-3,
    "m4_down": 1e-3,
}

_GOLDENS: dict[str, dict[str, str]] = {
    "applegpu_g17s": {
        "m16_gate_up": "dbb3c35f93a9e0678240675e6c7893bceaf9a5df549edc29963036869e009236",
        "m16_down": "969cb9327d6964d7b05506a36eba312365cd24962a78ee8116ac50736fde68af",
        "m4_gate_up": "e3fee4161dbcfe0938230dc140732210abd0b6a3022b37ef3b898ece443feea8",
        "m4_down": "731fe6500783948cf17333352d5c22cc9b59ddc506c10033fd0ee7ba45c82c15",
    },
}


def _arch() -> str:
    return str(mx.device_info().get("architecture", "unknown"))


def _case(name: str, M: int, K: int, N: int, seed: int):
    # Each activation row is planted on a target column (verify rows are
    # peaked at real tokens in production); i.i.d. Gaussian rows have
    # top1-top2 margins below the kernel deviation and certify nothing.
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(K)
    w_np = (rng.standard_normal((N, K)) * scale).astype(np.float32)
    targets = rng.integers(0, N, size=M)
    x_np = w_np[targets] + (rng.standard_normal((M, K)) * scale * 0.3).astype(
        np.float32
    )
    x = mx.array(x_np).astype(DTYPE)
    w_fp = mx.array(w_np).astype(DTYPE)
    w_q, scales, biases = mx.quantize(w_fp, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(x, w_q, scales, biases)
    return x, w_q, scales, biases


def _kernel(x, w_q, scales, biases):
    y = verify_matmul(
        x, w_q, scales, biases, transpose=True, group_size=GROUP_SIZE, bits=BITS
    )
    mx.eval(y)
    return y


def _stock(x, w_q, scales, biases):
    y = mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases,
        transpose=True, group_size=GROUP_SIZE, bits=BITS,
    )
    mx.eval(y)
    return y


@pytest.fixture(autouse=True)
def _enable_verify_qmm(monkeypatch):
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")


@pytest.mark.parametrize("name,M,K,N", PROD_SHAPES)
def test_kernel_is_deterministic(name, M, K, N):
    x, w_q, scales, biases = _case(name, M, K, N, SEEDS[0])
    outs = [_kernel(x, w_q, scales, biases) for _ in range(3)]
    for i, y in enumerate(outs[1:], start=2):
        assert mx.array_equal(outs[0], y).item(), (
            f"{name}: call {i} differs from call 1 — non-deterministic kernel"
        )


@pytest.mark.parametrize("name,M,K,N", PROD_SHAPES)
def test_deviation_bound_and_certified_argmax(name, M, K, N):
    bound = DEV_BOUNDS[name]
    for seed in SEEDS:
        x, w_q, scales, biases = _case(name, M, K, N, seed)
        y_k = np.array(_kernel(x, w_q, scales, biases).astype(mx.float32))
        y_s = np.array(_stock(x, w_q, scales, biases).astype(mx.float32))
        dev = float(np.max(np.abs(y_k - y_s)))
        assert dev <= bound, (
            f"{name} seed={seed:#x}: deviation {dev:.4g} exceeds pinned bound "
            f"{bound} — verify-kernel numerics drifted; re-validate e2e parity"
        )
        top2 = np.partition(y_s, -2, axis=-1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        certified = margin > 2.0 * dev
        flips = (np.argmax(y_k, axis=-1) != np.argmax(y_s, axis=-1)) & certified
        assert not flips.any(), (
            f"{name} seed={seed:#x}: argmax flipped on margin-certified rows "
            f"{np.nonzero(flips)[0].tolist()} — deviation bound is unsound"
        )
        assert certified.mean() >= 0.9, (
            f"{name} seed={seed:#x}: only {certified.mean():.0%} rows certified; "
            "margins collapsed relative to kernel deviation"
        )


@pytest.mark.parametrize("name,M,K,N", PROD_SHAPES)
def test_golden_fingerprint(name, M, K, N):
    arch = _arch()
    x, w_q, scales, biases = _case(name, M, K, N, SEEDS[0])
    y = np.array(_kernel(x, w_q, scales, biases).astype(mx.float32))
    digest = hashlib.sha256(y.tobytes()).hexdigest()
    if os.environ.get("DFLASH_REGEN_KERNEL_GOLDENS") == "1":
        print(f'GOLDEN "{arch}" "{name}": "{digest}"')
        pytest.skip("golden regeneration mode")
    profile = _GOLDENS.get(arch)
    if profile is None:
        pytest.skip(
            f"no golden fingerprints for GPU profile {arch!r}; run with "
            "DFLASH_REGEN_KERNEL_GOLDENS=1 to generate them"
        )
    assert profile[name] == digest, (
        f"{name} on {arch}: kernel output bytes changed "
        f"(expected {profile[name][:12]}…, got {digest[:12]}…) — numeric "
        "drift from an MLX/Metal/kernel change; re-validate e2e GPU parity, "
        "then regenerate with DFLASH_REGEN_KERNEL_GOLDENS=1"
    )

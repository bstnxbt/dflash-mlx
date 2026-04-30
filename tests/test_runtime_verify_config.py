# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os

from dflash_mlx import runtime
from dflash_mlx.engine.config import resolve_verify_len_cap, verify_token_count_for_block
from dflash_mlx.runtime import VerifyConfig
from dflash_mlx.runtime_context import runtime_config_from_profile

def _large_dense_config() -> dict:
    return {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "num_experts": 0,
    }

def test_verify_config_off_disables_target_verify_linears(monkeypatch):
    calls: list[bool | None] = []

    monkeypatch.setenv("DFLASH_VERIFY_LINEAR", "1")
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")
    monkeypatch.setattr(
        runtime,
        "load",
        lambda *args, **kwargs: (object(), object(), _large_dense_config()),
    )
    monkeypatch.setattr(runtime, "detect_target_family", lambda model: "pure_attention")

    import dflash_mlx.verify_linear as verify_linear

    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm=None, predicate=None: calls.append(enable_qmm) or 1,
    )

    _, _, meta = runtime.load_target_bundle(
        "model",
        verify_config=VerifyConfig.from_mode("off"),
    )

    assert meta["verify_linear_enabled"] is False
    assert calls == []
    assert os.environ["DFLASH_VERIFY_LINEAR"] == "1"
    assert os.environ["DFLASH_VERIFY_QMM"] == "1"

def test_verify_config_auto_threads_qmm_without_env_transport(monkeypatch):
    calls: list[bool | None] = []

    monkeypatch.setenv("DFLASH_VERIFY_QMM", "0")
    monkeypatch.setattr(
        runtime,
        "load",
        lambda *args, **kwargs: (object(), object(), _large_dense_config()),
    )
    monkeypatch.setattr(runtime, "detect_target_family", lambda model: "pure_attention")

    import dflash_mlx.verify_linear as verify_linear

    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm=None, predicate=None: calls.append(enable_qmm) or 3,
    )

    _, _, meta = runtime.load_target_bundle(
        "model",
        verify_config=VerifyConfig.from_mode("auto"),
    )

    assert meta["verify_linear_enabled"] is True
    assert meta["verify_linear_swapped"] == 3
    assert calls == [True]
    assert os.environ["DFLASH_VERIFY_QMM"] == "0"

def test_verify_config_none_preserves_internal_env_override(monkeypatch):
    calls: list[bool | None] = []

    monkeypatch.setenv("DFLASH_VERIFY_LINEAR", "0")
    monkeypatch.setattr(
        runtime,
        "load",
        lambda *args, **kwargs: (object(), object(), _large_dense_config()),
    )
    monkeypatch.setattr(runtime, "detect_target_family", lambda model: "pure_attention")

    import dflash_mlx.verify_linear as verify_linear

    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm=None, predicate=None: calls.append(enable_qmm) or 1,
    )

    _, _, meta = runtime.load_target_bundle("model", verify_config=None)

    assert meta["verify_linear_enabled"] is False
    assert calls == []

def test_runtime_verify_len_cap_limits_verify_token_count():
    cfg = runtime_config_from_profile(profile="balanced", verify_len_cap=4)
    cap = resolve_verify_len_cap(cfg, block_tokens=16)

    assert cap == 4
    assert verify_token_count_for_block(block_len=16, verify_len_cap=cap) == 4
    assert verify_token_count_for_block(block_len=2, verify_len_cap=cap) == 2

def test_runtime_verify_len_cap_zero_uses_block_size():
    cfg = runtime_config_from_profile(profile="balanced", verify_len_cap=0)
    cap = resolve_verify_len_cap(cfg, block_tokens=16)

    assert cap == 16
    assert verify_token_count_for_block(block_len=16, verify_len_cap=cap) == 16

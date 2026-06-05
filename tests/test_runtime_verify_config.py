# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import builtins
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

import dflash_mlx.runtime.config as runtime_config
from dflash_mlx.runtime import loading as runtime_loading
from dflash_mlx.engine.config import (
    resolve_speculative_cycle_config,
    resolve_verify_len_cap,
    verify_token_count_for_block,
)
from dflash_mlx.runtime import VerifyConfig
from dflash_mlx.runtime.config import (
    BENCHMARK_RUNTIME_FIELDS,
    add_offline_runtime_arguments,
    offline_runtime_error_message,
    offline_runtime_kwargs,
    runtime_config_from_defaults,
)
from dflash_mlx.runtime.context import build_offline_runtime_config

def _large_dense_config() -> dict:
    return {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "num_experts": 0,
    }


def _qwen_moe_verify_excluded_config() -> dict:
    return {
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "num_experts": 128,
    }


def _pure_attention_ops(*, supports_verify_linear: bool = True):
    return SimpleNamespace(
        family=lambda model: "pure_attention",
        capabilities_for=lambda model: SimpleNamespace(
            supports_verify_linear=supports_verify_linear,
        ),
        install_speculative_hooks=lambda model: None,
    )


def test_local_model_path_missing_huggingface_hub_reports_file_not_found(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(FileNotFoundError, match="huggingface_hub is unavailable"):
        runtime_loading._resolve_local_model_path(tmp_path / "missing-model")


def test_local_model_path_propagates_unexpected_hub_import_errors(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise RuntimeError("import side effect failed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="import side effect failed"):
        runtime_loading._resolve_local_model_path(tmp_path / "missing-model")


def test_verify_config_off_disables_target_verify_linears(monkeypatch):
    calls: list[bool | None] = []

    monkeypatch.setenv("DFLASH_VERIFY_LINEAR", "1")
    monkeypatch.setenv("DFLASH_VERIFY_QMM", "1")
    monkeypatch.setattr(
        runtime_loading,
        "load",
        lambda *args, **kwargs: (object(), object(), _large_dense_config()),
    )
    monkeypatch.setattr(runtime_loading, "resolve_target_ops", lambda model: _pure_attention_ops())

    import dflash_mlx.verify_linear as verify_linear

    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm=None, predicate=None: calls.append(enable_qmm) or 1,
    )

    target_bundle = runtime_loading.load_target_bundle(
        "model",
        verify_config=VerifyConfig.from_mode("off"),
    )
    meta = target_bundle.meta

    assert meta["verify_linear_enabled"] is False
    assert calls == []
    assert os.environ["DFLASH_VERIFY_LINEAR"] == "1"
    assert os.environ["DFLASH_VERIFY_QMM"] == "1"

def test_verify_config_dflash_threads_qmm_without_env_transport(monkeypatch):
    calls: list[bool | None] = []

    monkeypatch.setenv("DFLASH_VERIFY_QMM", "0")
    monkeypatch.setattr(
        runtime_loading,
        "load",
        lambda *args, **kwargs: (object(), object(), _large_dense_config()),
    )
    monkeypatch.setattr(runtime_loading, "resolve_target_ops", lambda model: _pure_attention_ops())

    import dflash_mlx.verify_linear as verify_linear

    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm=None, predicate=None: calls.append(enable_qmm) or 3,
    )

    target_bundle = runtime_loading.load_target_bundle(
        "model",
        verify_config=VerifyConfig.from_mode("dflash"),
    )
    meta = target_bundle.meta

    assert meta["verify_linear_enabled"] is True
    assert meta["verify_linear_swapped"] == 3
    assert calls == [True]
    assert os.environ["DFLASH_VERIFY_QMM"] == "0"


def test_verify_config_rejects_old_auto_mode_name():
    with pytest.raises(ValueError, match="dflash, adaptive, ddtree, or off"):
        VerifyConfig.from_mode("auto")


def test_runtime_loader_uses_capability_not_config_fingerprint(monkeypatch):
    calls: list[bool | None] = []

    monkeypatch.setattr(
        runtime_loading,
        "load",
        lambda *args, **kwargs: (object(), object(), _qwen_moe_verify_excluded_config()),
    )
    monkeypatch.setattr(runtime_loading, "resolve_target_ops", lambda model: _pure_attention_ops())

    import dflash_mlx.verify_linear as verify_linear

    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm=None, predicate=None: calls.append(enable_qmm) or 2,
    )

    target_bundle = runtime_loading.load_target_bundle(
        "model",
        verify_config=VerifyConfig.from_mode("dflash"),
    )
    meta = target_bundle.meta

    assert meta["verify_linear_enabled"] is True
    assert meta["verify_linear_swapped"] == 2
    assert calls == [True]


def test_verify_config_none_preserves_internal_env_override(monkeypatch):
    calls: list[bool | None] = []

    monkeypatch.setenv("DFLASH_VERIFY_LINEAR", "0")
    monkeypatch.setattr(
        runtime_loading,
        "load",
        lambda *args, **kwargs: (object(), object(), _large_dense_config()),
    )
    monkeypatch.setattr(runtime_loading, "resolve_target_ops", lambda model: _pure_attention_ops())

    import dflash_mlx.verify_linear as verify_linear

    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm=None, predicate=None: calls.append(enable_qmm) or 1,
    )

    target_bundle = runtime_loading.load_target_bundle("model", verify_config=None)
    meta = target_bundle.meta

    assert meta["verify_linear_enabled"] is False
    assert calls == []

def test_target_capability_can_disable_verify_linear_before_parity(monkeypatch):
    calls: list[bool | None] = []

    monkeypatch.setenv("DFLASH_VERIFY_LINEAR", "1")
    monkeypatch.setattr(
        runtime_loading,
        "load",
        lambda *args, **kwargs: (object(), object(), _large_dense_config()),
    )
    monkeypatch.setattr(
        runtime_loading,
        "resolve_target_ops",
        lambda model: _pure_attention_ops(supports_verify_linear=False),
    )

    import dflash_mlx.verify_linear as verify_linear

    monkeypatch.setattr(
        verify_linear,
        "install_verify_linears",
        lambda model, *, enable_qmm=None, predicate=None: calls.append(enable_qmm) or 1,
    )

    target_bundle = runtime_loading.load_target_bundle(
        "model",
        verify_config=VerifyConfig.from_mode("dflash"),
    )
    meta = target_bundle.meta

    assert meta["verify_linear_enabled"] is False
    assert calls == []

def test_runtime_config_from_defaults_uses_canonical_l2_default():
    cfg = runtime_config_from_defaults()

    assert cfg.prefix_cache_l2 is True
    assert cfg.prefix_cache_l2_dir == os.path.expanduser("~/.cache/dflash/prefix_l2")


def test_runtime_config_from_defaults_ignores_runtime_env(monkeypatch):
    monkeypatch.setenv("DFLASH_PREFIX_CACHE_L2_DIR", "/tmp/dflash-other-l2")

    cfg = runtime_config_from_defaults()

    assert cfg.prefix_cache_l2_dir == os.path.expanduser("~/.cache/dflash/prefix_l2")


def test_runtime_config_rejects_old_auto_verify_mode():
    with pytest.raises(ValueError, match="verify_mode must be dflash"):
        runtime_config_from_defaults(verify_mode="auto")


def test_offline_runtime_config_uses_default_runtime_without_cache(monkeypatch):
    monkeypatch.setenv("DFLASH_DRAFT_WINDOW_SIZE", "2048")

    cfg = build_offline_runtime_config(target_fa_window=0)

    assert cfg.prefill_step_size == 2048
    assert cfg.draft_sink_size == 64
    assert cfg.draft_window_size == 1024
    assert cfg.verify_len_cap == 0
    assert cfg.prefix_cache is False
    assert cfg.prefix_cache_l2 is False


def test_offline_runtime_arguments_project_from_shared_runtime_schema():
    parser = argparse.ArgumentParser()
    add_offline_runtime_arguments(parser, BENCHMARK_RUNTIME_FIELDS)

    args = parser.parse_args(
        [
            "--target-fa-window",
            "2048",
            "--draft-window-size",
            "512",
            "--verify-mode",
            "ddtree",
        ]
    )

    assert offline_runtime_kwargs(args, BENCHMARK_RUNTIME_FIELDS) == {
        "prefill_step_size": 2048,
        "target_fa_window": 2048,
        "quantize_kv_cache": False,
        "draft_sink_size": 64,
        "draft_window_size": 512,
        "verify_len_cap": 0,
        "verify_mode": "ddtree",
    }


def test_offline_runtime_arguments_reject_old_auto_verify_mode():
    parser = argparse.ArgumentParser()
    add_offline_runtime_arguments(parser, BENCHMARK_RUNTIME_FIELDS)

    with pytest.raises(SystemExit):
        parser.parse_args(["--verify-mode", "auto"])


def test_offline_runtime_argument_defaults_follow_single_runtime_default(monkeypatch):
    monkeypatch.setattr(
        runtime_config,
        "DEFAULT_RUNTIME_CONFIG",
        replace(runtime_config.DEFAULT_RUNTIME_CONFIG, draft_window_size=1536),
    )
    parser = argparse.ArgumentParser()
    add_offline_runtime_arguments(parser, BENCHMARK_RUNTIME_FIELDS)

    args = parser.parse_args([])

    assert args.draft_window_size == 1536


def test_offline_runtime_error_message_removes_internal_field_labels():
    assert (
        offline_runtime_error_message("--draft-window-size / draft_window_size must be > 0")
        == "--draft-window-size must be > 0"
    )


def test_runtime_loader_delegates_hooks_to_target_ops_without_family_gate(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class _Ops:
        def family(self, model):
            return "gemma4_swa"

        def capabilities_for(self, model):
            return SimpleNamespace(
                supports_verify_linear=False,
            )

        def install_speculative_hooks(self, model):
            calls.append(("install", {}))

    monkeypatch.setattr(
        runtime_loading,
        "load",
        lambda *args, **kwargs: (object(), object(), _large_dense_config()),
    )
    monkeypatch.setattr(runtime_loading, "resolve_target_ops", lambda model: _Ops())

    target_bundle = runtime_loading.load_target_bundle(
        "model",
        quantize_kv_cache=False,
        verify_config=VerifyConfig.from_mode("dflash"),
    )
    meta = target_bundle.meta

    assert meta["target_family"] == "gemma4_swa"
    assert calls == [("install", {})]


def test_runtime_verify_len_cap_limits_verify_token_count():
    cfg = runtime_config_from_defaults(verify_len_cap=4)
    cap = resolve_verify_len_cap(cfg, block_tokens=16)

    assert cap == 4
    assert verify_token_count_for_block(block_len=16, verify_len_cap=cap) == 4
    assert verify_token_count_for_block(block_len=2, verify_len_cap=cap) == 2

def test_runtime_verify_len_cap_zero_uses_block_size():
    cfg = runtime_config_from_defaults(verify_len_cap=0)
    cap = resolve_verify_len_cap(cfg, block_tokens=16)

    assert cap == 16
    assert verify_token_count_for_block(block_len=16, verify_len_cap=cap) == 16


def test_speculative_cycle_config_defaults_to_draft_block_size():
    cfg = runtime_config_from_defaults(verify_len_cap=0)
    draft = SimpleNamespace(block_size=16)

    cycle = resolve_speculative_cycle_config(cfg, draft, block_tokens=None)

    assert cycle.draft_block_size == 16
    assert cycle.requested_block_tokens == 16
    assert cycle.effective_block_tokens == 16
    assert cycle.verify_len_cap == 16


def test_speculative_cycle_config_clamps_requested_block_size():
    cfg = runtime_config_from_defaults(verify_len_cap=0)
    draft = SimpleNamespace(block_size=16)

    high = resolve_speculative_cycle_config(cfg, draft, block_tokens=64)
    low = resolve_speculative_cycle_config(cfg, draft, block_tokens=0)

    assert high.requested_block_tokens == 64
    assert high.effective_block_tokens == 16
    assert low.requested_block_tokens == 0
    assert low.effective_block_tokens == 1


def test_speculative_cycle_config_threads_verify_cap_after_block_clamp():
    cfg = runtime_config_from_defaults(verify_len_cap=4)
    draft = SimpleNamespace(block_size=16)

    cycle = resolve_speculative_cycle_config(cfg, draft, block_tokens=12)

    assert cycle.effective_block_tokens == 12
    assert cycle.verify_len_cap == 4

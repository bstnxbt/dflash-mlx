# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Registry semantics for the runtime cache manager.

Two DFlash models loaded in one process (e.g. an embedding host's engine
pool) must each keep their own prefix-cache manager alive concurrently.
Previously a single process-global slot meant loading the second model
retired the first's cache. These tests pin the registry contract:

  * distinct cache identities coexist (neither retired),
  * the same identity + same config returns the same manager,
  * the same identity reconfigured replaces (and retires) the old one,
  * shutdown can target one identity or clear all.
"""
from __future__ import annotations

import pytest

import dflash_mlx.cache.manager as cm
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.runtime.config import runtime_config_from_defaults
from dflash_mlx.runtime.context import build_runtime_context


def _ctx(**overrides):
    values = dict(
        prefix_cache=True,
        prefix_cache_l2=False,
        prefix_cache_l2_dir="/tmp/dflash-prefix-l2-registry-test",
    )
    values.update(overrides)
    return build_runtime_context(runtime_config_from_defaults(**values))


def _key(*, target: str = "target-A", window: int = 1024) -> DFlashPrefixKey:
    return DFlashPrefixKey(
        target_model_id=target,
        draft_model_id="draft-A",
        capture_layer_ids=(1, 2),
        draft_sink_size=64,
        draft_window_size=window,
        template_hash="a" * 64,
        prompt_policy_hash="b" * 64,
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    # Start and end each test with an empty registry so order can't leak state.
    cm.shutdown_runtime_cache_manager()
    yield
    cm.shutdown_runtime_cache_manager()


def test_distinct_identities_coexist():
    ctx = _ctx()
    mgr_a = cm.get_runtime_cache_manager(ctx, cache_identity="model-A")
    mgr_b = cm.get_runtime_cache_manager(ctx, cache_identity="model-B")

    assert mgr_a is not None and mgr_b is not None
    assert mgr_a is not mgr_b
    # The crux of #1892: loading B must NOT retire A.
    assert mgr_a.active
    assert mgr_b.active


def test_same_identity_same_config_returns_same_manager():
    ctx = _ctx()
    first = cm.get_runtime_cache_manager(ctx, cache_identity="model-A")
    second = cm.get_runtime_cache_manager(ctx, cache_identity="model-A")
    assert first is second


def test_same_identity_reconfig_replaces_and_retires_old():
    first = cm.get_runtime_cache_manager(
        _ctx(prefix_cache_max_entries=2), cache_identity="model-A"
    )
    second = cm.get_runtime_cache_manager(
        _ctx(prefix_cache_max_entries=7), cache_identity="model-A"
    )
    assert first is not second
    assert not first.active  # reconfigured -> old retired
    assert second.active
    assert second.stats()["max_entries"] == 7


def test_same_model_fingerprint_change_replaces_old_manager():
    ctx = _ctx()
    first = cm.get_runtime_cache_manager(ctx, cache_identity=_key(window=1024))
    second = cm.get_runtime_cache_manager(ctx, cache_identity=_key(window=2048))

    assert first is not None and second is not None
    assert first is not second
    assert not first.active
    assert second.active


def test_disabled_same_model_runtime_retires_existing_manager():
    first = cm.get_runtime_cache_manager(_ctx(), cache_identity=_key())

    disabled = cm.get_runtime_cache_manager(
        _ctx(prefix_cache=False),
        cache_identity=_key(),
    )

    assert disabled is None
    assert first is not None and not first.active


def test_reconfig_of_one_identity_leaves_other_untouched():
    ctx = _ctx()
    mgr_a = cm.get_runtime_cache_manager(ctx, cache_identity="model-A")
    mgr_b = cm.get_runtime_cache_manager(ctx, cache_identity="model-B")
    # Reconfigure only A.
    mgr_a2 = cm.get_runtime_cache_manager(
        _ctx(prefix_cache_max_entries=9), cache_identity="model-A"
    )
    assert mgr_a2 is not mgr_a
    assert not mgr_a.active
    assert mgr_a2.active
    assert mgr_b.active  # B is unaffected


def test_sync_resolves_per_identity():
    ctx = _ctx()
    mgr_a = cm.get_runtime_cache_manager(ctx, cache_identity="model-A")
    mgr_b = cm.get_runtime_cache_manager(ctx, cache_identity="model-B")
    assert cm.sync_runtime_cache_manager(ctx, cache_identity="model-A") is mgr_a
    assert cm.sync_runtime_cache_manager(ctx, cache_identity="model-B") is mgr_b


def test_keyed_shutdown_only_retires_that_identity():
    ctx = _ctx()
    mgr_a = cm.get_runtime_cache_manager(ctx, cache_identity="model-A")
    mgr_b = cm.get_runtime_cache_manager(ctx, cache_identity="model-B")
    cm.shutdown_runtime_cache_manager(ctx, cache_identity="model-A")
    assert not mgr_a.active
    assert mgr_b.active  # B survives A's unload
    # A's slot is gone; B is still resolvable.
    assert cm.sync_runtime_cache_manager(ctx, cache_identity="model-B") is mgr_b


def test_shutdown_all_retires_everything():
    ctx = _ctx()
    mgr_a = cm.get_runtime_cache_manager(ctx, cache_identity="model-A")
    mgr_b = cm.get_runtime_cache_manager(ctx, cache_identity="model-B")
    cm.shutdown_runtime_cache_manager()  # no args == teardown all
    assert not mgr_a.active
    assert not mgr_b.active

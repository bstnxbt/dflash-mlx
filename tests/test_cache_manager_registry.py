# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import pytest

import dflash_mlx.cache.manager as cache_managers
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.runtime.config import runtime_config_from_defaults
from dflash_mlx.runtime.context import build_runtime_context


def _context(**overrides):
    values = {
        "prefix_cache": True,
        "prefix_cache_l2": False,
        "prefix_cache_l2_dir": "/tmp/dflash-prefix-l2-registry-test",
    }
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
    cache_managers.shutdown_runtime_cache_manager()
    yield
    cache_managers.shutdown_runtime_cache_manager()


def test_distinct_model_managers_coexist_and_resolve_independently():
    context = _context()
    first = cache_managers.get_runtime_cache_manager(
        context, cache_identity="model-A"
    )
    second = cache_managers.get_runtime_cache_manager(
        context, cache_identity="model-B"
    )

    assert first is not None and second is not None and first is not second
    assert first.active and second.active
    assert (
        cache_managers.sync_runtime_cache_manager(
            context, cache_identity="model-A"
        )
        is first
    )
    assert (
        cache_managers.sync_runtime_cache_manager(
            context, cache_identity="model-B"
        )
        is second
    )


def test_same_model_reuses_or_replaces_its_manager_when_config_changes():
    context = _context(prefix_cache_max_entries=2)
    first = cache_managers.get_runtime_cache_manager(
        context, cache_identity=_key(window=1024)
    )
    assert (
        cache_managers.get_runtime_cache_manager(
            context, cache_identity=_key(window=1024)
        )
        is first
    )

    replacement = cache_managers.get_runtime_cache_manager(
        _context(prefix_cache_max_entries=7),
        cache_identity=_key(window=2048),
    )

    assert first is not None and replacement is not None
    assert replacement is not first
    assert not first.active and replacement.active
    assert replacement.stats()["max_entries"] == 7


def test_disabling_one_model_retires_only_its_manager():
    context = _context()
    first = cache_managers.get_runtime_cache_manager(
        context, cache_identity=_key(target="target-A")
    )
    second = cache_managers.get_runtime_cache_manager(
        context, cache_identity=_key(target="target-B")
    )

    disabled = cache_managers.get_runtime_cache_manager(
        _context(prefix_cache=False),
        cache_identity=_key(target="target-A"),
    )

    assert disabled is None
    assert first is not None and not first.active
    assert second is not None and second.active


def test_scoped_and_global_shutdown_retire_the_expected_managers():
    context = _context()
    first = cache_managers.get_runtime_cache_manager(
        context, cache_identity="model-A"
    )
    second = cache_managers.get_runtime_cache_manager(
        context, cache_identity="model-B"
    )

    cache_managers.shutdown_runtime_cache_manager(
        context, cache_identity="model-A"
    )
    assert first is not None and not first.active
    assert second is not None and second.active

    cache_managers.shutdown_runtime_cache_manager()
    assert not second.active

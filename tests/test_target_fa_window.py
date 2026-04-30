# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache, RotatingKVCache

from dflash_mlx.cache.codecs import target_cache_is_serializable
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.engine.rollback import restore_target_cache_after_acceptance
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

class _FakeLinearAttn:
    conv_kernel_size = 4

class _FakeGdnLayer:
    is_linear = True
    linear_attn = _FakeLinearAttn()

class _FakeFaLayer:
    is_linear = False
    self_attn = object()

class _FakeTarget:
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            layers=[
                _FakeFaLayer(),
                _FakeGdnLayer(),
                _FakeFaLayer(),
            ]
        )

def _make_prefix_key(target_fa_window: int) -> DFlashPrefixKey:
    return DFlashPrefixKey(
        target_model_id="target",
        draft_model_id="draft",
        capture_layer_ids=(3, 7),
        draft_sink_size=64,
        draft_window_size=1024,
        target_fa_window=target_fa_window,
    )

def _runtime_context(*, target_fa_window: int = 0, prefix_cache: bool = True):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            target_fa_window=target_fa_window,
            prefix_cache=prefix_cache,
        ),
        diagnostics=SimpleNamespace(trace=None),
    )

def _keys(n_tokens: int, *, offset: int = 0) -> mx.array:
    return (
        mx.arange(offset, offset + n_tokens * 4, dtype=mx.float32)
        .reshape(1, 1, n_tokens, 4)
    )

def test_target_fa_window_startup_env_parser(monkeypatch):
    from dflash_mlx.server.config import build_parser, normalize_cli_args

    monkeypatch.delenv("DFLASH_TARGET_FA_WINDOW", raising=False)
    args = build_parser().parse_args(["--model", "target"])
    normalize_cli_args(args)
    assert args.runtime_config.target_fa_window == 0

    monkeypatch.setenv("DFLASH_TARGET_FA_WINDOW", "2048")
    args = build_parser().parse_args(["--model", "target"])
    normalize_cli_args(args)
    assert args.runtime_config.target_fa_window == 2048

    monkeypatch.setenv("DFLASH_TARGET_FA_WINDOW", "")
    args = build_parser().parse_args(["--model", "target"])
    normalize_cli_args(args)
    assert args.runtime_config.target_fa_window == 0

    monkeypatch.setenv("DFLASH_TARGET_FA_WINDOW", "-1")
    args = build_parser().parse_args(["--model", "target"])
    with pytest.raises(SystemExit, match="target_fa_window"):
        normalize_cli_args(args)

    monkeypatch.setenv("DFLASH_TARGET_FA_WINDOW", "not-int")
    args = build_parser().parse_args(["--model", "target"])
    with pytest.raises(SystemExit):
        normalize_cli_args(args)

def test_target_cache_default_keeps_fa_kvcache_and_gdn_rollback(monkeypatch):
    import dflash_mlx.runtime as runtime

    monkeypatch.delenv("DFLASH_TARGET_FA_WINDOW", raising=False)
    monkeypatch.setattr(runtime, "_install_target_speculative_hooks", lambda _model: None)

    caches = runtime.make_target_cache(
        _FakeTarget(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
    )

    assert isinstance(caches[0], KVCache)
    assert isinstance(caches[1], RecurrentRollbackCache)
    assert isinstance(caches[2], KVCache)

def test_target_cache_window_rotates_fa_only_and_leaves_gdn_unchanged(monkeypatch):
    import dflash_mlx.runtime as runtime

    monkeypatch.setattr(runtime, "_install_target_speculative_hooks", lambda _model: None)

    caches = runtime.make_target_cache(
        _FakeTarget(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=2048,
    )

    assert isinstance(caches[0], RotatingKVCache)
    assert caches[0].max_size == 2048
    assert isinstance(caches[1], RecurrentRollbackCache)
    assert isinstance(caches[2], RotatingKVCache)
    assert caches[2].max_size == 2048
    assert target_cache_is_serializable(caches) is False

def test_target_cache_window_rejects_quantized_target_kv(monkeypatch):
    import dflash_mlx.runtime as runtime

    with pytest.raises(ValueError, match="target_fa_window"):
        runtime.make_target_cache(
            _FakeTarget(),
            enable_speculative_linear_cache=False,
            quantize_kv_cache=True,
            target_fa_window=2048,
        )

def test_prefix_cache_fingerprint_separates_target_fa_window():
    prompt = [1, 2, 3, 4]
    key_full = _make_prefix_key(0)
    key_windowed = _make_prefix_key(2048)
    assert key_full != key_windowed

    cache = DFlashPrefixCache(max_entries=4)
    snap = DFlashPrefixSnapshot.from_full_target_hidden(
        token_ids=tuple(prompt),
        fa_states=(),
        gdn_states=(),
        target_hidden=mx.zeros((1, len(prompt), 1), dtype=mx.float32),
        last_logits=None,
        key=key_full,
    )
    cache.insert(snap)
    matched, found = cache.lookup(prompt, key_windowed)
    assert matched == 0
    assert found is None

def test_prefix_cache_flow_disabled_for_windowed_target(monkeypatch):
    import dflash_mlx.server.prefix_cache_flow as flow_mod

    class FakeProvider:
        model_key = ("target", None, "draft")

    class FakeDraft:
        target_layer_ids = [3, 7]

    class FakeTokenizer:
        unk_token_id = -1

        def convert_tokens_to_ids(self, tokens):
            return [-1 for _ in tokens]

    monkeypatch.setattr(flow_mod, "_DFLASH_PREFIX_CACHE_SINGLETON", DFlashPrefixCache())

    flow = flow_mod.PrefixCacheFlow.for_request(
        model_provider=FakeProvider(),
        draft_model=FakeDraft(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        runtime_context=_runtime_context(target_fa_window=2048),
    )

    assert flow.cache is None
    assert flow.key is None
    assert flow.hit_tokens == 0
    assert flow.snapshot is None

def test_build_prefix_key_records_target_fa_window(monkeypatch):
    from dflash_mlx.server.prefix_cache_manager import build_prefix_key

    class FakeProvider:
        model_key = ("target", None, "draft")

    class FakeDraft:
        target_layer_ids = [3, 7]

    key = build_prefix_key(
        FakeProvider(),
        FakeDraft(),
        _runtime_context(target_fa_window=4096),
    )
    assert key.target_fa_window == 4096

def test_serve_cli_target_fa_window_sets_runtime_config(monkeypatch):
    from dflash_mlx.server.config import build_parser, normalize_cli_args

    monkeypatch.delenv("DFLASH_TARGET_FA_WINDOW", raising=False)
    args = build_parser().parse_args(
        ["--model", "target", "--target-fa-window", "4096"]
    )
    normalize_cli_args(args)
    assert args.runtime_config.target_fa_window == 4096
    assert "DFLASH_TARGET_FA_WINDOW" not in os.environ

def test_serve_cli_target_fa_window_rejects_negative():
    from dflash_mlx.server.config import build_parser, normalize_cli_args

    args = build_parser().parse_args(
        ["--model", "target", "--target-fa-window", "-1"]
    )
    with pytest.raises(SystemExit, match="--target-fa-window"):
        normalize_cli_args(args)

def test_fallback_ar_forces_target_fa_window_zero(monkeypatch):
    import dflash_mlx.engine.fallback as fallback
    import dflash_mlx.runtime as runtime

    calls = []

    def fake_make_target_cache(target_model, **kwargs):
        calls.append(kwargs)
        return ["cache"]

    monkeypatch.setattr(runtime, "make_target_cache", fake_make_target_cache)

    cache = fallback._make_fallback_target_cache(
        object(),
        quantize_kv_cache=False,
    )

    assert cache == ["cache"]
    assert calls == [
        {
            "enable_speculative_linear_cache": False,
            "quantize_kv_cache": False,
            "target_fa_window": 0,
        }
    ]

def test_rotating_kv_trim_survives_logical_positions_past_cache_length():
    cache = RotatingKVCache(max_size=8)
    cache.update_and_fetch(_keys(12), _keys(12, offset=1000))
    cache.update_and_fetch(_keys(4, offset=100), _keys(4, offset=1100))
    mx.eval(cache.keys, cache.values)

    assert cache.offset == 16
    assert int(cache.keys.shape[2]) == 11

    restore_target_cache_after_acceptance(
        [cache],
        target_len=14,
        acceptance_length=1,
        drafted_tokens=3,
    )

    assert cache.offset == 14
    mask = cache.make_mask(4, return_array=True)
    assert mask.shape[-2:] == (4, 11)

    cache.update_and_fetch(_keys(1, offset=200), _keys(1, offset=1200))
    mx.eval(cache.keys, cache.values)
    assert cache.offset == 15

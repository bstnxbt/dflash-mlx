# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from dflash_mlx import model as model_mod
from dflash_mlx.runtime import loading as runtime_loading
from dflash_mlx.draft_backend import EagerDraftBackend
from dflash_mlx.engine import spec_epoch
from dflash_mlx.engine.config import resolve_draft_window
from dflash_mlx.engine.events import SnapshotPublishedEvent, SummaryEvent, TokenEvent
from dflash_mlx.engine.target_gemma4 import Gemma4TargetOps
from dflash_mlx.engine.target_ops import bind_draft_to_target
from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashAttention,
    DFlashDraftModel,
    DFlashDraftModelArgs,
    FullContextDraftKVCache,
)
from dflash_mlx.runtime.config import runtime_config_from_defaults
from dflash_mlx.runtime.context import build_runtime_context


def _args(**overrides):
    base = {
        "model_type": "gemma4",
        "hidden_size": 32,
        "num_hidden_layers": 5,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "rope_theta": 1_000_000.0,
        "head_dim": 8,
        "tie_word_embeddings": False,
        "num_target_layers": 8,
        "block_size": 16,
        "layer_types": (
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ),
        "sliding_window": 2048,
    }
    base.update(overrides)
    return DFlashDraftModelArgs(**base)


class _TinyTargetOps:
    def __init__(self, *, token_id: int, supports_prefix_snapshot: bool = True) -> None:
        self.token_id = int(token_id)
        self.supports_prefix_snapshot = bool(supports_prefix_snapshot)

    def capabilities_for(self, _target_model):
        return SimpleNamespace(supports_prefix_snapshot=self.supports_prefix_snapshot)

    def make_cache(self, *_args, **_kwargs):
        return []

    def forward_with_hidden_capture(
        self,
        _target_model,
        *,
        input_ids,
        cache,
        capture_layer_ids,
        logits_last_only=False,
    ):
        del cache, capture_layer_ids
        batch, seq_len = input_ids.shape
        logits_len = 1 if logits_last_only else seq_len
        logits = mx.zeros((batch, logits_len, 8), dtype=mx.float32)
        logits[:, :, self.token_id] = 1.0
        hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
        return logits, hidden

    def verify_block(
        self,
        *,
        target_model,
        verify_ids,
        target_cache,
        capture_layer_ids,
    ):
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def extract_context_feature(self, hidden_states, target_layer_id_list):
        return hidden_states[int(target_layer_id_list[0]) + 1]

    def arm_rollback(self, *_args, **_kwargs):
        return None

    def restore_after_acceptance(self, *_args, **_kwargs):
        return 0

    def cleanup_generation_caches(self, *_args):
        return None


class _EmptyDraftBackend:
    def make_cache(self, **_kwargs):
        return []


def _tiny_runtime_context():
    return build_runtime_context(
        runtime_config_from_defaults(
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )


def _tiny_draft_model(*, mask_token_id: int = 5):
    return SimpleNamespace(
        target_layer_ids=[0],
        block_size=1,
        mask_token_id=mask_token_id,
        project_target_hidden=lambda value: value,
    )


def test_gemma4_draft_args_default_missing_layer_types_to_official_swa_pattern():
    args = DFlashDraftModelArgs.from_dict(
        {
            "model_type": "gemma4",
            "hidden_size": 32,
            "num_hidden_layers": 6,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 128,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "rope_theta": 1_000_000.0,
            "head_dim": 8,
            "tie_word_embeddings": False,
            "num_target_layers": 8,
            "block_size": 16,
            "sliding_window": 2048,
            "sliding_window_pattern": 5,
        }
    )

    assert args.layer_types == (
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    )


def test_non_gemma_draft_args_keep_missing_layer_types_for_historical_cache_policy():
    args = DFlashDraftModelArgs.from_dict(
        {
            "model_type": "qwen3",
            "hidden_size": 32,
            "num_hidden_layers": 3,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 128,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "rope_theta": 1_000_000.0,
            "head_dim": 8,
            "tie_word_embeddings": False,
            "num_target_layers": 8,
            "block_size": 16,
        }
    )

    assert args.layer_types == ()

    caches = EagerDraftBackend().make_cache(
        draft_model=SimpleNamespace(args=args, layers=[object(), object(), object()]),
        sink_size=2,
        window_size=3,
    )

    assert [type(cache) for cache in caches] == [
        ContextOnlyDraftKVCache,
        ContextOnlyDraftKVCache,
        ContextOnlyDraftKVCache,
    ]
    assert not any(isinstance(cache, FullContextDraftKVCache) for cache in caches)
    assert [cache.sink_size for cache in caches] == [2, 2, 2]
    assert [cache.window_size for cache in caches] == [3, 3, 3]


def test_qwen_explicit_full_attention_layer_types_keep_bounded_cache_policy():
    args = DFlashDraftModelArgs.from_dict(
        {
            "model_type": "qwen3",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 128,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "rope_theta": 1_000_000.0,
            "head_dim": 8,
            "tie_word_embeddings": False,
            "num_target_layers": 8,
            "block_size": 16,
            "layer_types": ("full_attention", "full_attention"),
        }
    )

    caches = EagerDraftBackend().make_cache(
        draft_model=SimpleNamespace(args=args, layers=[object(), object()]),
        sink_size=2,
        window_size=3,
    )

    assert args.layer_types == ("full_attention", "full_attention")
    assert [type(cache) for cache in caches] == [
        ContextOnlyDraftKVCache,
        ContextOnlyDraftKVCache,
    ]
    assert not any(isinstance(cache, FullContextDraftKVCache) for cache in caches)
    assert [cache.sink_size for cache in caches] == [2, 2]
    assert [cache.window_size for cache in caches] == [3, 3]


def test_qwen_explicit_full_attention_resolve_draft_window_stays_bounded():
    draft_model = DFlashDraftModel(
        _args(
            model_type="qwen3",
            num_hidden_layers=2,
            layer_types=("full_attention", "full_attention"),
            sliding_window=None,
        )
    )
    runtime_config = SimpleNamespace(draft_sink_size=2, draft_window_size=3)

    assert (
        resolve_draft_window(
            runtime_config,
            draft_model,
            context_len=100,
            allow_full_attention_context=False,
        )
        == (2, 3)
    )


def test_full_context_capability_can_expand_explicit_full_attention_draft_window():
    draft_model = DFlashDraftModel(
        _args(
            model_type="qwen3",
            num_hidden_layers=2,
            layer_types=("full_attention", "full_attention"),
            sliding_window=None,
        )
    )
    runtime_config = SimpleNamespace(draft_sink_size=2, draft_window_size=3)

    assert (
        resolve_draft_window(
            runtime_config,
            draft_model,
            context_len=100,
            allow_full_attention_context=True,
        )
        == (2, 100)
    )


def test_real_zlab_gemma4_draft_shape_allocates_full_context_from_target_capability():
    target_model = SimpleNamespace(
        args=SimpleNamespace(
            model_type="gemma4",
            layer_types=("sliding_attention", "full_attention"),
            num_kv_shared_layers=0,
        ),
        model=SimpleNamespace(layers=[object()], embed_tokens=object()),
    )
    target_ops = Gemma4TargetOps()
    assert target_ops.supports_model(target_model)
    allow_full_context = target_ops.capabilities_for(
        target_model
    ).supports_full_context_draft_layers
    draft_model = DFlashDraftModel(_args(model_type="qwen3"))

    caches = EagerDraftBackend().make_cache(
        draft_model=draft_model,
        sink_size=2,
        window_size=3,
        allow_full_context_layers=allow_full_context,
    )

    assert draft_model.args.model_type == "qwen3"
    assert allow_full_context is True
    assert [type(cache) for cache in caches] == [
        ContextOnlyDraftKVCache,
        ContextOnlyDraftKVCache,
        ContextOnlyDraftKVCache,
        ContextOnlyDraftKVCache,
        FullContextDraftKVCache,
    ]


def test_full_context_draft_cache_keeps_all_positions_without_windowing():
    sliding_cache = ContextOnlyDraftKVCache(sink_size=2, window_size=3)
    full_cache = FullContextDraftKVCache()
    keys = mx.zeros((1, 1, 8, 4))
    values = mx.zeros((1, 1, 8, 4))

    sliding_cache.append_context(keys, values, num_positions=8)
    full_cache.append_context(keys, values, num_positions=8)

    mx.eval(sliding_cache.position_indices(), full_cache.position_indices())
    assert sliding_cache.cache_length() == 5
    assert sliding_cache.position_indices().tolist() == [0, 1, 5, 6, 7]
    assert full_cache.cache_length() == 8
    assert full_cache.position_indices().tolist() == list(range(8))


def test_full_context_draft_cache_never_concatenates(monkeypatch):
    concat_calls = 0
    original_concatenate = model_mod.mx.concatenate

    def tracked_concatenate(*args, **kwargs):
        nonlocal concat_calls
        concat_calls += 1
        return original_concatenate(*args, **kwargs)

    cache = FullContextDraftKVCache()
    first_keys = mx.zeros((1, 1, 8, 4))
    first_values = mx.zeros((1, 1, 8, 4))
    next_keys = mx.ones((1, 1, 2, 4))
    next_values = mx.ones((1, 1, 2, 4))

    monkeypatch.setattr(model_mod.mx, "concatenate", tracked_concatenate)
    cache.append_context(first_keys, first_values, num_positions=8)
    cache.append_context(next_keys, next_values, num_positions=2)
    keys, values = cache.fetch()
    positions = cache.position_indices()
    mx.eval(keys, values, positions)

    assert concat_calls == 0
    assert cache.cache_length() == 10
    assert cache.offset == 10
    assert keys.shape[2] == 10
    assert values.shape[2] == 10
    assert positions.tolist() == list(range(10))
    assert bool(mx.all(keys[:, :, :8, :] == 0).item())
    assert bool(mx.all(keys[:, :, 8:, :] == 1).item())


def test_full_context_draft_cache_grow_preserves_content_across_step():
    cache = FullContextDraftKVCache()
    first = mx.broadcast_to(
        mx.arange(300, dtype=mx.float32).reshape(1, 1, 300, 1), (1, 2, 300, 4)
    )
    cache.append_context(mx.array(first), mx.array(first) * 2.0, num_positions=300)
    tail = mx.full((1, 2, 10, 4), 7.0)
    cache.append_context(tail, tail, num_positions=10)

    keys, values = cache.fetch()
    mx.eval(keys, values)
    assert cache.cache_length() == 310
    assert int(cache.keys.shape[2]) == 512
    assert bool(mx.all(keys[:, :, :300, :] == first).item())
    assert bool(mx.all(values[:, :, :300, :] == first * 2.0).item())
    assert bool(mx.all(keys[:, :, 300:, :] == 7.0).item())


def test_full_context_draft_cache_fetch_with_block_is_transient():
    cache = FullContextDraftKVCache()
    ctx = mx.zeros((1, 1, 8, 4))
    cache.append_context(ctx, ctx, num_positions=8)

    noise = mx.full((1, 1, 3, 4), 5.0)
    keys, values = cache.fetch_with_block(noise, noise)
    mx.eval(keys, values)
    assert keys.shape[2] == 11
    assert bool(mx.all(keys[:, :, 8:, :] == 5.0).item())
    assert cache.cache_length() == 8
    assert cache.offset == 8

    committed = mx.full((1, 1, 3, 4), 9.0)
    cache.append_context(committed, committed, num_positions=3)
    keys2, _ = cache.fetch()
    mx.eval(keys2)
    assert cache.cache_length() == 11
    assert bool(mx.all(keys2[:, :, 8:, :] == 9.0).item())


def test_full_context_draft_cache_matches_concat_reference():
    mx.random.seed(7)
    chunks = [
        mx.random.normal((1, 2, n, 4), dtype=mx.float32) for n in (5, 1, 3, 2)
    ]
    cache = FullContextDraftKVCache()
    for chunk in chunks:
        cache.append_context(
            mx.array(chunk), mx.array(chunk) + 1.0, num_positions=int(chunk.shape[2])
        )
    keys, values = cache.fetch()
    ref_keys = mx.concatenate(chunks, axis=2)
    mx.eval(keys, values, ref_keys)
    assert bool(mx.array_equal(keys, ref_keys).item())
    assert bool(mx.array_equal(values, ref_keys + 1.0).item())


def test_full_attention_draft_cache_attention_path_keeps_full_context():
    attn = DFlashAttention(
        _args(
            num_hidden_layers=1,
            layer_types=("full_attention",),
            sliding_window=None,
        ),
        layer_idx=0,
    )
    cache = FullContextDraftKVCache()
    hidden = mx.zeros((1, 2, 32))
    target_hidden = mx.zeros((1, 8, 32))

    out = attn(hidden, target_hidden=target_hidden, cache=cache)
    mx.eval(out, cache.position_indices())

    assert out.shape == hidden.shape
    assert cache.offset == 8
    assert cache.cache_length() == 8
    assert cache.position_indices().tolist() == list(range(8))


def test_bind_target_model_propagates_non_default_embed_scale():
    draft_model = DFlashDraftModel(_args())
    draft_model.layers = []
    draft_model.norm = lambda value: value

    class _TargetOps:
        def text_model(self, target_model):
            class _TextModel:
                embed_scale = 3.0

            return _TextModel()

    bind_draft_to_target(draft_model, object(), target_ops=_TargetOps())
    noise_embedding = mx.ones((1, 2, 32))
    target_hidden = mx.zeros((1, 2, len(draft_model.target_layer_ids) * 32))

    out = draft_model(
        noise_embedding=noise_embedding,
        target_hidden=target_hidden,
        cache=[],
    )
    expected = noise_embedding * 3.0
    mx.eval(out, expected)

    assert bool(mx.all(out == expected).item())


def test_projected_context_forward_matches_raw_wrapper():
    draft_model = DFlashDraftModel(_args())
    noise_embedding = mx.zeros((1, 3, 32), dtype=mx.float32)
    raw_features = mx.arange(
        1 * 5 * len(draft_model.target_layer_ids) * 32,
        dtype=mx.float32,
    ).reshape(1, 5, len(draft_model.target_layer_ids) * 32)

    projected = draft_model.project_target_hidden(raw_features)
    raw_out = draft_model(
        noise_embedding=noise_embedding,
        target_hidden=raw_features,
        cache=None,
    )
    projected_out = draft_model.forward_projected_context(
        noise_embedding=noise_embedding,
        draft_context=projected,
        cache=None,
    )
    mx.eval(raw_out, projected_out)

    assert bool(mx.allclose(raw_out, projected_out, rtol=1e-5, atol=1e-5).item())


def test_project_target_hidden_is_chunk_equivalent():
    draft_model = DFlashDraftModel(_args())
    raw_features = mx.arange(
        1 * 5 * len(draft_model.target_layer_ids) * 32,
        dtype=mx.float32,
    ).reshape(1, 5, len(draft_model.target_layer_ids) * 32)

    full = draft_model.project_target_hidden(raw_features)
    chunked = mx.concatenate(
        [
            draft_model.project_target_hidden(raw_features[:, :2, :]),
            draft_model.project_target_hidden(raw_features[:, 2:, :]),
        ],
        axis=1,
    )
    mx.eval(full, chunked)

    assert bool(mx.allclose(full, chunked, rtol=1e-5, atol=1e-5).item())


def test_load_draft_bundle_rejects_future_draft_owned_lm_head(tmp_path):
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {"lm_head.weight": mx.zeros((2, 2))},
    )

    try:
        runtime_loading.load_draft_bundle(tmp_path)
    except ValueError as exc:
        message = str(exc)
        assert "contains draft-owned lm_head weights" in message
        assert "TargetOps.logits_from_hidden" in message
    else:
        raise AssertionError("draft checkpoints with lm_head weights must fail fast")


def test_load_draft_bundle_model_class_callback_accepts_mlx_lm_config_keyword(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_load_model(model_path, *, lazy, get_model_classes):
        model_classes = get_model_classes(config={"model_type": "qwen3"})
        calls.append((model_path, lazy, model_classes))
        return object(), {"model_type": "qwen3"}

    monkeypatch.setattr(runtime_loading, "load_model", fake_load_model)

    model, meta = runtime_loading.load_draft_bundle(tmp_path, lazy=False)

    assert model is not None
    assert meta["config"] == {"model_type": "qwen3"}
    assert calls == [
        (
            tmp_path,
            False,
            (DFlashDraftModel, DFlashDraftModelArgs),
        )
    ]


def test_load_draft_bundle_uses_real_mlx_lm_model_class_callback(tmp_path):
    config = {
        "model_type": "qwen3",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 32,
        "num_key_value_heads": 1,
        "max_position_embeddings": 64,
        "rope_theta": 1_000_000.0,
        "head_dim": 8,
        "tie_word_embeddings": False,
        "num_target_layers": 2,
        "block_size": 16,
    }
    draft_model = DFlashDraftModel(DFlashDraftModelArgs.from_dict(config))
    weights = dict(tree_flatten(draft_model.parameters()))

    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    loaded_model, meta = runtime_loading.load_draft_bundle(tmp_path, lazy=False)

    assert isinstance(loaded_model, DFlashDraftModel)
    assert meta["config"]["model_type"] == "qwen3"
    assert meta["config"]["hidden_size"] == 16


def test_load_draft_bundle_requires_safetensors_for_weight_inspection(
    tmp_path, monkeypatch
):
    (tmp_path / "model.safetensors").write_bytes(b"placeholder")
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setattr(
        runtime_loading,
        "load_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("load_model should not run before lm_head inspection")
        ),
    )

    with pytest.raises(
        ValueError,
        match="Cannot inspect draft safetensors weights for unsupported lm_head",
    ):
        runtime_loading.load_draft_bundle(tmp_path)


def test_load_draft_bundle_rejects_malformed_safetensors_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text("{not-json")

    with pytest.raises(ValueError, match="Invalid draft safetensors index JSON"):
        runtime_loading.load_draft_bundle(tmp_path)


def test_load_draft_bundle_rejects_non_object_safetensors_index(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps([]))

    with pytest.raises(ValueError, match="Invalid draft safetensors index payload"):
        runtime_loading.load_draft_bundle(tmp_path)


def test_load_draft_bundle_rejects_non_object_safetensors_weight_map(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": []})
    )

    with pytest.raises(ValueError, match="Invalid draft safetensors index weight_map"):
        runtime_loading.load_draft_bundle(tmp_path)


def test_load_draft_bundle_rejects_index_declared_draft_owned_lm_head(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}})
    )

    with pytest.raises(ValueError, match="contains draft-owned lm_head weights"):
        runtime_loading.load_draft_bundle(tmp_path)


def test_mask_token_id_can_be_generated():
    mask_token_id = 5

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_TinyTargetOps(token_id=mask_token_id),
            tokenizer=object(),
            draft_model=_tiny_draft_model(mask_token_id=mask_token_id),
            draft_backend=_EmptyDraftBackend(),
            prompt="unused",
            max_new_tokens=1,
            prompt_tokens_override=[1],
            runtime_context=_tiny_runtime_context(),
        )
    )

    token_events = [event for event in events if isinstance(event, TokenEvent)]
    summary = next(event for event in events if isinstance(event, SummaryEvent))
    assert [event.token_id for event in token_events] == [mask_token_id]
    assert summary.generated_token_ids == (mask_token_id,)


def test_prefix_snapshot_capability_disables_snapshot_events(monkeypatch):
    generation_concat_calls = 0
    original_concatenate = mx.concatenate

    def tracked_concatenate(values, *args, **kwargs):
        nonlocal generation_concat_calls
        values = list(values)
        if len(values) == 2 and all(
            getattr(value, "shape", None) == (1, 1, 2) for value in values
        ):
            generation_concat_calls += 1
        return original_concatenate(values, *args, **kwargs)

    monkeypatch.setattr(spec_epoch.mx, "concatenate", tracked_concatenate)
    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_TinyTargetOps(token_id=2, supports_prefix_snapshot=False),
            tokenizer=object(),
            draft_model=_tiny_draft_model(),
            draft_backend=_EmptyDraftBackend(),
            prompt="unused",
            max_new_tokens=1,
            prompt_tokens_override=[1],
            runtime_context=_tiny_runtime_context(),
        )
    )

    assert not any(isinstance(event, SnapshotPublishedEvent) for event in events)
    assert generation_concat_calls == 0


def test_draft_args_reject_layer_type_length_mismatch():
    try:
        _args(layer_types=("full_attention",), num_hidden_layers=2)
    except ValueError as exc:
        assert "layer_types length must match num_hidden_layers" in str(exc)
    else:
        raise AssertionError("layer_types length mismatch must be rejected")


def test_draft_args_reject_unknown_layer_type():
    try:
        _args(
            num_hidden_layers=2,
            layer_types=("sliding_attention", "mystery_attention"),
        )
    except ValueError as exc:
        assert "Unknown DFlash draft layer type" in str(exc)
        assert "mystery_attention" in str(exc)
    else:
        raise AssertionError("unknown layer type must be rejected")


def test_draft_args_reject_sliding_attention_without_positive_window():
    try:
        _args(
            num_hidden_layers=2,
            layer_types=("sliding_attention", "full_attention"),
            sliding_window=0,
        )
    except ValueError as exc:
        assert "sliding_attention draft layers require a positive sliding_window" in str(exc)
    else:
        raise AssertionError("sliding attention without positive window must be rejected")


def test_gemma4_unified_target_supports_model():
    # Current mlx-community Gemma 4 exports declare gemma4_unified; mlx-lm's
    # MODEL_REMAPPING loads them through the gemma4 module, so the wrapper
    # this backend receives is structurally identical and only the
    # args.model_type string differs.
    target_model = SimpleNamespace(
        args=SimpleNamespace(
            model_type="gemma4_unified",
            layer_types=("sliding_attention", "full_attention"),
            num_kv_shared_layers=0,
        ),
        model=SimpleNamespace(layers=[object()], embed_tokens=object()),
    )
    assert Gemma4TargetOps().supports_model(target_model)


def test_draft_args_hoist_rope_theta_from_rope_parameters():
    # transformers >= 5.x drafter exports (z-lab gemma4 drafters) nest rope
    # fields under rope_parameters with no top-level rope_theta.
    base = {
        "model_type": "qwen3",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "head_dim": 8,
        "tie_word_embeddings": True,
        "num_target_layers": 8,
        "block_size": 16,
        "layer_types": ["full_attention", "full_attention"],
    }
    args = DFlashDraftModelArgs.from_dict(
        {**base, "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"}}
    )
    assert args.rope_theta == 1_000_000
    assert args.rope_scaling is None

    yarn = {"rope_theta": 1_000_000, "rope_type": "yarn", "factor": 4.0}
    args = DFlashDraftModelArgs.from_dict({**base, "rope_parameters": yarn})
    assert args.rope_theta == 1_000_000
    assert args.rope_scaling == yarn

    # explicit top-level values always win over the nested form
    args = DFlashDraftModelArgs.from_dict(
        {
            **base,
            "rope_theta": 10_000.0,
            "rope_parameters": {"rope_theta": 1_000_000, "rope_type": "default"},
        }
    )
    assert args.rope_theta == 10_000.0

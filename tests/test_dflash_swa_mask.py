import mlx.core as mx

from dflash_mlx import runtime
from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashAttention,
    DFlashDraftModel,
    DFlashDraftModelArgs,
)


def _args(**overrides):
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
        "rope_theta": 1_000_000.0,
        "head_dim": 8,
        "tie_word_embeddings": False,
        "num_target_layers": 8,
        "block_size": 16,
        "layer_types": ("sliding_attention", "full_attention"),
        "sliding_window": 3,
    }
    base.update(overrides)
    return DFlashDraftModelArgs(**base)


def test_dflash_attention_marks_only_sliding_layers():
    model = DFlashDraftModel(_args())

    assert model.layers[0].self_attn.sliding_window == 3
    assert model.layers[1].self_attn.sliding_window is None


def test_sliding_attention_mask_matches_full_causal_window():
    attn = DFlashAttention(_args(), layer_idx=0)
    mask = attn._attention_mask(block_len=3, query_offset=5, key_len=8)
    expected = mx.array(
        [
            [False, False, False, True, True, True, False, False],
            [False, False, False, False, True, True, True, False],
            [False, False, False, False, False, True, True, True],
        ],
        dtype=mx.bool_,
    )
    mx.eval(mask, expected)

    assert mask.shape == expected.shape
    assert bool(mx.all(mask == expected).item())


def test_sliding_attention_mask_respects_truncated_context_cache_positions():
    cache = ContextOnlyDraftKVCache(sink_size=2, window_size=3)
    keys = mx.zeros((1, 1, 6, 4))
    values = mx.zeros((1, 1, 6, 4))
    cache.append_context(keys, values, num_positions=6)

    positions = cache.position_indices()
    noise_positions = mx.array([6, 7], dtype=mx.int32)
    key_positions = mx.concatenate([positions, noise_positions], axis=0)
    attn = DFlashAttention(_args(), layer_idx=0)
    mask = attn._attention_mask(
        block_len=2,
        query_offset=6,
        key_len=7,
        key_positions=key_positions,
    )
    expected = mx.array(
        [
            [False, False, False, True, True, True, False],
            [False, False, False, False, True, True, True],
        ],
        dtype=mx.bool_,
    )
    mx.eval(positions, mask, expected)

    assert positions.tolist() == [0, 1, 3, 4, 5]
    assert bool(mx.all(mask == expected).item())


def test_context_cache_selects_only_retained_spans_for_long_initial_context():
    cache = ContextOnlyDraftKVCache(sink_size=2, window_size=3)

    assert cache.context_spans_to_append(8) == [(0, 2), (5, 8)]


def test_context_cache_sparse_append_preserves_logical_positions_and_offset():
    cache = ContextOnlyDraftKVCache(sink_size=2, window_size=3)
    keys = mx.zeros((1, 1, 5, 4))
    values = mx.zeros((1, 1, 5, 4))
    positions = mx.array([0, 1, 5, 6, 7], dtype=mx.int32)

    cache.append_context(
        keys,
        values,
        num_positions=8,
        positions=positions,
        advance_positions=8,
    )

    mx.eval(cache.position_indices())
    assert cache.offset == 8
    assert cache.cache_length() == 5
    assert cache.position_indices().tolist() == [0, 1, 5, 6, 7]


def test_context_cache_later_long_append_keeps_new_tail_only():
    cache = ContextOnlyDraftKVCache(sink_size=2, window_size=3)
    keys = mx.zeros((1, 1, 5, 4))
    values = mx.zeros((1, 1, 5, 4))
    cache.append_context(
        keys,
        values,
        num_positions=8,
        positions=mx.array([0, 1, 5, 6, 7], dtype=mx.int32),
        advance_positions=8,
    )

    assert cache.context_spans_to_append(6) == [(3, 6)]

    new_keys = mx.zeros((1, 1, 3, 4))
    new_values = mx.zeros((1, 1, 3, 4))
    cache.append_context(
        new_keys,
        new_values,
        num_positions=6,
        positions=mx.array([11, 12, 13], dtype=mx.int32),
        advance_positions=6,
    )

    mx.eval(cache.position_indices())
    assert cache.offset == 14
    assert cache.cache_length() == 5
    assert cache.position_indices().tolist() == [0, 1, 11, 12, 13]


def test_full_attention_layer_keeps_existing_unmasked_behavior():
    attn = DFlashAttention(_args(), layer_idx=1)

    assert attn._attention_mask(block_len=3, query_offset=5, key_len=8) is None


def test_sliding_attention_disables_unmasked_fast_cross_attention(monkeypatch):
    attn = DFlashAttention(_args(), layer_idx=0)

    def fail_fast_path(*args, **kwargs):
        raise AssertionError("sliding attention must not use unmasked fast cross-attention")

    monkeypatch.setattr(mx.fast, "dflash_cross_attention", fail_fast_path, raising=False)
    hidden = mx.zeros((1, 2, 32))
    target_hidden = mx.zeros((1, 4, 32))
    out = attn(hidden, target_hidden=target_hidden, cache=None)
    mx.eval(out)

    assert out.shape == hidden.shape


def test_effective_draft_window_never_under_model_swa_window():
    draft_model = DFlashDraftModel(_args(sliding_window=2048))

    assert runtime._effective_draft_window_size(draft_model, 1024) == 2048
    assert runtime._effective_draft_window_size(draft_model, 4096) == 4096


def test_effective_draft_window_expands_unwindowed_full_attention_to_context():
    draft_model = DFlashDraftModel(
        _args(layer_types=("full_attention", "full_attention"), sliding_window=None)
    )

    assert (
        runtime._effective_draft_window_size(
            draft_model,
            1024,
            context_len=4096,
        )
        == 4096
    )


def test_effective_draft_window_can_keep_explicit_user_window():
    draft_model = DFlashDraftModel(
        _args(layer_types=("full_attention", "full_attention"), sliding_window=None)
    )

    assert (
        runtime._effective_draft_window_size(
            draft_model,
            1024,
            context_len=4096,
            allow_full_attention_context=False,
        )
        == 1024
    )

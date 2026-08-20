# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import json

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from dflash_mlx.draft.checkpoint import get_draft_model_classes
from dflash_mlx.draft.dflash2 import (
    CandidateSelector,
    DFlash2Attention,
    DFlash2DraftModel,
    DFlash2DraftModelArgs,
    DraftProposal,
    _grouped_dynamic_convolve,
    normalize_dflash2_config,
    remap_dflash2_codebook_weights,
)
from dflash_mlx.draft_backend import EagerDraftBackend
from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs
from dflash_mlx.runtime import loading as runtime_loading


def _dflash2_config(**overrides):
    config = {
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "is_causal": False,
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 5,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [1],
        },
        "head_dim": 8,
        "hidden_size": 32,
        "intermediate_size": 64,
        "layer_types": ["sliding_attention"],
        "max_position_embeddings": 4096,
        "model_type": "qwen3",
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "num_target_layers": 8,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1_000_000.0, "rope_type": "default"},
        "sliding_window": 4,
        "tie_word_embeddings": False,
        "vocab_size": 32,
    }
    for key, value in overrides.items():
        if key == "dflash_config":
            nested = dict(config["dflash_config"])
            nested.update(value)
            config["dflash_config"] = nested
        else:
            config[key] = value
    return config


def test_dflash2_config_normalizes_nested_checkpoint_contract():
    normalized = normalize_dflash2_config(_dflash2_config())

    assert normalized["block_size"] == 8
    assert normalized["conv_kernel_size"] == 2
    assert normalized["conv_group_size"] == 16
    assert normalized["selector_top_k"] == 16
    assert normalized["selector_rank"] == 256
    assert normalized["target_layer_ids"] == [1]
    assert normalized["rope_theta"] == 1_000_000.0
    assert normalized["rope_scaling"] is None

    scaled = normalize_dflash2_config(
        _dflash2_config(
            rope_parameters={
                "factor": 4.0,
                "rope_theta": 1_000_000.0,
                "rope_type": "yarn",
            }
        )
    )

    assert scaled["rope_theta"] == 1_000_000.0
    assert scaled["rope_scaling"] == {"factor": 4.0, "rope_type": "yarn"}


def test_dflash2_config_fails_fast_on_malformed_schema():
    missing_selector = _dflash2_config()
    del missing_selector["dflash_config"]["selector_top_k"]
    invalid_configs = (
        _dflash2_config(is_causal=True),
        missing_selector,
        _dflash2_config(dflash_config={"conv_kernel_size": 3}),
    )

    for config in invalid_configs:
        with pytest.raises(ValueError):
            DFlash2DraftModelArgs.from_dict(config)


def test_checkpoint_dispatch_selects_dflash2_without_affecting_prior_dflash():
    assert get_draft_model_classes(_dflash2_config()) == (
        DFlash2DraftModel,
        DFlash2DraftModelArgs,
    )
    capabilities = DFlash2DraftModel(
        DFlash2DraftModelArgs.from_dict(_dflash2_config())
    ).capabilities
    assert capabilities.default_block_tokens == 5
    assert capabilities.max_block_tokens == 5
    assert not capabilities.supports_copyspec
    assert not capabilities.supports_ddtree
    assert capabilities.supports_early_rollback_launch

    prior_config = {
        "model_type": "qwen3",
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 32,
        "num_key_value_heads": 2,
        "max_position_embeddings": 4096,
        "rope_theta": 1_000_000.0,
        "head_dim": 8,
        "tie_word_embeddings": False,
        "num_target_layers": 8,
        "block_size": 16,
    }

    assert get_draft_model_classes(prior_config) == (
        DFlashDraftModel,
        DFlashDraftModelArgs,
    )


def test_dflash2_non_causal_block_attention_allows_future_block_tokens():
    args = DFlash2DraftModelArgs.from_dict(_dflash2_config())
    attn = DFlash2Attention(args, layer_idx=0)

    mask = attn._attention_mask(
        block_len=2,
        query_offset=4,
        key_len=6,
        key_positions=mx.array([0, 1, 2, 3, 4, 5], dtype=mx.int32),
    )
    expected = mx.array(
        [
            [False, True, True, True, True, True],
            [False, False, True, True, True, True],
        ],
        dtype=mx.bool_,
    )
    mx.eval(mask, expected)

    assert bool(mx.all(mask == expected).item())


def test_dflash2_grouped_dynamic_conv_matches_reference():
    hidden = mx.arange(1 * 4 * 32, dtype=mx.float32).reshape(1, 4, 32) / 32.0
    dynamic = mx.arange(1 * 4 * 2 * 2, dtype=mx.float32).reshape(1, 4, 2, 2) / 10.0
    base = mx.arange(2 * 32, dtype=mx.float32).reshape(2, 32) / 100.0

    out = _grouped_dynamic_convolve(hidden, dynamic, base, group_size=16)
    mx.eval(out)

    h = hidden.tolist()
    d = dynamic.tolist()
    b = base.tolist()
    expected = []
    for pos in range(4):
        row = []
        for channel in range(32):
            group = channel // 16
            value = 0.0
            for offset in range(2):
                src = pos - offset
                if src < 0:
                    continue
                source = h[0][src][channel]
                value += b[offset][channel] * source
                value += d[0][pos][offset][group] * source
            row.append(value)
        expected.append(row)

    assert out.shape == hidden.shape
    assert mx.allclose(
        out,
        mx.array([expected], dtype=mx.float32),
        rtol=1e-5,
        atol=1e-5,
    ).item()


def test_dflash2_candidate_selector_uses_predecessor_path_edges():
    args = DFlash2DraftModelArgs.from_dict(_dflash2_config())
    selector = CandidateSelector(args)
    selector.hidden_projection.weight = mx.zeros_like(selector.hidden_projection.weight)
    selector.hidden_projection.weight[0, 0] = 1.0
    selector.hidden_projection.weight[1, 1] = 1.0
    selector.predecessor_codebook.weight = mx.zeros_like(
        selector.predecessor_codebook.weight
    )
    selector.successor_codebook.weight = mx.zeros_like(selector.successor_codebook.weight)
    selector.predecessor_codebook.weight[7, 0] = 1.0
    selector.successor_codebook.weight[20, 0] = 10.0
    selector.predecessor_codebook.weight[20, 1] = 1.0
    selector.successor_codebook.weight[21, 1] = 10.0

    hidden = mx.array([[[1.0, 0.0] + [0.0] * 30, [0.0, 1.0] + [0.0] * 30]])
    logits = mx.broadcast_to(
        mx.arange(32, dtype=mx.float32).reshape(1, 1, 32) / 100.0,
        (1, 2, 32),
    )

    proposal = selector.select(
        hidden,
        logits,
        mx.array([7], dtype=mx.uint32),
        temperature=0.0,
    )
    mx.eval(proposal.token_ids, proposal.candidate_ids)

    assert proposal.token_ids.tolist() == [[20, 21]]
    assert proposal.candidate_ids.shape == (1, 2, 16)
    assert proposal.probabilities is None


def test_dflash2_backend_proposal_uses_selector_contract():
    args = DFlash2DraftModelArgs.from_dict(_dflash2_config())
    draft_model = DFlash2DraftModel(args)
    draft_model.forward_projected_context = lambda **_kwargs: mx.zeros(
        (1, 8, 32), dtype=mx.float32
    )

    class _TargetOps:
        def embed_tokens(self, _target_model):
            return lambda token_ids: mx.zeros((*token_ids.shape, 32), dtype=mx.float32)

        def logits_from_hidden(self, _target_model, hidden):
            logits = mx.arange(32, dtype=mx.float32).reshape(1, 1, 32) / 100.0
            return mx.broadcast_to(logits, (1, int(hidden.shape[1]), 32))

    proposal = EagerDraftBackend().propose_block(
        target_model=object(),
        target_ops=_TargetOps(),
        draft_model=draft_model,
        draft_cache=[],
        staged_first=mx.array([7], dtype=mx.uint32),
        draft_context=mx.zeros((1, 2, 32), dtype=mx.float32),
        block_len=8,
        mask_token_tail=mx.full((7,), 5, dtype=mx.uint32),
        suppress_token_mask=None,
        capture_q=True,
    )
    mx.eval(proposal.token_ids, proposal.q_token_ids, proposal.q_probs)

    assert proposal.token_ids.shape == (7,)
    assert proposal.q_token_ids.shape == (7, 16)
    assert proposal.q_probs.shape == (7, 16)
    assert mx.allclose(mx.sum(proposal.q_probs, axis=-1), mx.ones((7,))).item()

    drafted, top_ids, top_logprobs = EagerDraftBackend().draft_greedy_capture(
        target_model=object(),
        target_ops=_TargetOps(),
        draft_model=draft_model,
        draft_cache=[],
        staged_first=mx.array([7], dtype=mx.uint32),
        draft_context=mx.zeros((1, 2, 32), dtype=mx.float32),
        block_len=8,
        mask_token_tail=mx.full((7,), 5, dtype=mx.uint32),
        suppress_token_mask=None,
        async_launch=False,
        top_width=4,
    )
    mx.eval(drafted, top_ids, top_logprobs)

    assert drafted.shape == (7,)
    assert top_ids.shape == top_logprobs.shape == (7, 4)
    assert mx.all(top_logprobs[:, :-1] >= top_logprobs[:, 1:]).item()


def test_dflash2_backend_rejects_underfilled_proposal():
    args = DFlash2DraftModelArgs.from_dict(_dflash2_config())
    draft_model = DFlash2DraftModel(args)
    draft_model.forward_projected_context = lambda **_kwargs: mx.zeros(
        (1, 5, 32), dtype=mx.float32
    )
    draft_model.select_proposal = lambda **_kwargs: DraftProposal(
        token_ids=mx.zeros((3,), dtype=mx.uint32)
    )

    class _TargetOps:
        def embed_tokens(self, _target_model):
            return lambda token_ids: mx.zeros(
                (*token_ids.shape, 32), dtype=mx.float32
            )

        def logits_from_hidden(self, _target_model, hidden):
            return mx.zeros((1, int(hidden.shape[1]), 32), dtype=mx.float32)

    with pytest.raises(ValueError):
        EagerDraftBackend().propose_block(
            target_model=object(),
            target_ops=_TargetOps(),
            draft_model=draft_model,
            draft_cache=[],
            staged_first=mx.array([7], dtype=mx.uint32),
            draft_context=mx.zeros((1, 2, 32), dtype=mx.float32),
            block_len=5,
            mask_token_tail=mx.full((4,), 5, dtype=mx.uint32),
            suppress_token_mask=None,
        )


def test_dflash2_safetensor_codebook_remap_is_exact():
    pred = mx.zeros((4, 2))
    succ = mx.ones((4, 2))

    remapped = remap_dflash2_codebook_weights(
        {
            "candidate_selector.predecessor_codebook": pred,
            "candidate_selector.successor_codebook": succ,
        }
    )

    assert "candidate_selector.predecessor_codebook" not in remapped
    assert "candidate_selector.successor_codebook" not in remapped
    assert remapped["candidate_selector.predecessor_codebook.weight"] is pred
    assert remapped["candidate_selector.successor_codebook.weight"] is succ

    with pytest.raises(ValueError):
        remap_dflash2_codebook_weights(
            {
                "candidate_selector.predecessor_codebook": pred,
                "candidate_selector.predecessor_codebook.weight": pred,
            }
        )


def test_load_draft_bundle_loads_dflash2_checkpoint(tmp_path):
    config = _dflash2_config()
    draft_model = DFlash2DraftModel(DFlash2DraftModelArgs.from_dict(config))
    weights = dict(tree_flatten(draft_model.parameters()))
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    loaded_model, meta = runtime_loading.load_draft_bundle(tmp_path, lazy=False)

    assert isinstance(loaded_model, DFlash2DraftModel)
    assert meta["config"]["architectures"] == ["DFlash2DraftModel"]

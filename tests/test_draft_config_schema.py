# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import pytest

from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs


def _schema(*, nested: bool) -> dict:
    data = {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 6 if nested else 8,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "num_key_value_heads": 8 if nested else 4,
        "max_position_embeddings": 4096,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "num_target_layers": 41,
        "dflash_config": {
            "mask_token_id": 248077,
            "target_layer_ids": (
                [1, 6, 11, 16, 22, 27, 32, 37]
                if nested
                else [1, 10, 19, 28, 37]
            ),
        },
        "layer_types": (
            (
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            )
            if nested
            else ("full_attention",) * 8
        ),
    }
    if nested:
        data["sliding_window"] = 4096
        data["dflash_config"]["block_size"] = 16
        data["rope_parameters"] = {
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
        }
    else:
        data["block_size"] = 16
        data["rope_theta"] = 10_000_000.0
    return data


@pytest.mark.parametrize(
    "nested,expected_layers,expected_kv_heads,expected_anchors",
    [(False, 8, 4, 5), (True, 6, 8, 8)],
)
def test_draft_schema_builds_legacy_and_nested_checkpoints(
    nested: bool,
    expected_layers: int,
    expected_kv_heads: int,
    expected_anchors: int,
):
    args = DFlashDraftModelArgs.from_dict(_schema(nested=nested))
    model = DFlashDraftModel(args)

    assert args.block_size == 16
    assert args.rope_theta == 10_000_000.0
    assert args.num_key_value_heads == expected_kv_heads
    assert len(model.target_layer_ids) == expected_anchors
    assert len(model.layers) == expected_layers
    assert model.fc.weight.shape == (2048, expected_anchors * 2048)


def test_top_level_schema_fields_take_precedence_over_nested_values():
    data = _schema(nested=False)
    data["dflash_config"]["block_size"] = 99
    data["rope_parameters"] = {"rope_theta": 1.0}

    args = DFlashDraftModelArgs.from_dict(data)

    assert args.block_size == 16
    assert args.rope_theta == 10_000_000.0

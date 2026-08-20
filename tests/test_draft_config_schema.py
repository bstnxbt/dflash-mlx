# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Regression tests for DFlashDraftModelArgs.from_dict schema tolerance.

z-lab iterated the DFlash draft config.json layout multiple times. This test
pins from_dict behavior across every known schema variant so consumers do not
silently break when HuggingFace publishes a new revision.

Covered schemas:
    * legacy (HF revisions through ~May 2026): block_size and rope_theta top-level.
    * nested-dflash-config (z-lab HF revision c69b185, Jun 18 2026):
      block_size moved into dflash_config; rope_theta into rope_parameters.
    * rope-scaling-fallback: rope_theta nested inside rope_scaling (defensive).
"""

import pytest

from dflash_mlx.model import DFlashDraftModelArgs


def _base_legacy_schema() -> dict:
    return {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 8,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
        "rope_theta": 10_000_000.0,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "num_target_layers": 41,
        "block_size": 16,
        "dflash_config": {
            "mask_token_id": 248070,
            "target_layer_ids": [1, 10, 19, 28, 37],
        },
        "layer_types": ("full_attention",) * 8,
    }


def _base_nested_schema() -> dict:
    """z-lab HF revision c69b185 / f181eece (Jun 18-19 2026).

    block_size moved into dflash_config; rope_theta into a new rope_parameters
    dict; layer topology switched to mostly-sliding; layer count 8 -> 6;
    num_key_value_heads 4 -> 8; target_layer_ids 5 -> 8 anchors.
    """
    return {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 6,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "num_key_value_heads": 8,
        "max_position_embeddings": 4096,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "num_target_layers": 41,
        "sliding_window": 4096,
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37],
        },
        "rope_parameters": {
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
        },
        "layer_types": (
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ),
    }


@pytest.mark.parametrize(
    "schema_factory,expected_layers,expected_kv_heads,expected_anchor_count",
    [
        ("legacy", 8, 4, 5),
        ("nested", 6, 8, 8),
    ],
)
def test_from_dict_resolves_block_size_and_rope_theta_across_schemas(
    schema_factory: str,
    expected_layers: int,
    expected_kv_heads: int,
    expected_anchor_count: int,
):
    factory = _base_legacy_schema if schema_factory == "legacy" else _base_nested_schema
    data = factory()

    args = DFlashDraftModelArgs.from_dict(data)

    assert args.block_size == 16, f"[{schema_factory}] block_size must resolve to 16"
    assert args.rope_theta == 10_000_000.0, f"[{schema_factory}] rope_theta must resolve to 1e7"
    assert args.num_hidden_layers == expected_layers
    assert args.num_key_value_heads == expected_kv_heads
    assert tuple(args.layer_types)[0:1] in (("full_attention",), ("sliding_attention",))


def test_from_dict_unwraps_block_size_from_dflash_config_when_top_level_absent():
    data = _base_nested_schema()
    data.pop("block_size", None)
    data.pop("rope_theta", None)

    args = DFlashDraftModelArgs.from_dict(data)

    assert args.block_size == 16
    assert args.rope_theta == 10_000_000.0


def test_from_dict_does_not_invent_missing_block_size():
    data = _base_nested_schema()
    data.pop("block_size", None)
    data["dflash_config"].pop("block_size", None)

    with pytest.raises(TypeError, match="block_size"):
        DFlashDraftModelArgs.from_dict(data)


def test_from_dict_unwraps_rope_theta_from_rope_scaling_as_fallback():
    data = _base_legacy_schema()
    data.pop("rope_theta", None)
    data["rope_scaling"] = {"rope_theta": 10_000_000.0, "rope_type": "default"}

    args = DFlashDraftModelArgs.from_dict(data)

    assert args.rope_theta == 10_000_000.0


def test_from_dict_preserves_top_level_block_size_and_rope_theta_when_present():
    data = _base_legacy_schema()
    data["block_size"] = 16
    data["rope_theta"] = 10_000_000.0
    data["dflash_config"]["block_size"] = 99
    data["rope_parameters"] = {"rope_theta": 1.0}

    args = DFlashDraftModelArgs.from_dict(data)

    assert args.block_size == 16
    assert args.rope_theta == 10_000_000.0


def test_nested_schema_constructs_into_a_model():
    """The Jun 2026 nested schema must round-trip into a real DFlashDraftModel.

    Guards against regressions in the model class (fc width, anchor count,
    sliding-window setup) that could be introduced by future refactors.
    """
    from dflash_mlx.model import DFlashDraftModel

    args = DFlashDraftModelArgs.from_dict(_base_nested_schema())
    model = DFlashDraftModel(args)

    assert len(model.layers) == 6
    assert model.block_size == 16
    assert model.mask_token_id == 248077
    assert len(model.target_layer_ids) == 8
    assert model.fc.weight.shape == (2048, 8 * 2048)

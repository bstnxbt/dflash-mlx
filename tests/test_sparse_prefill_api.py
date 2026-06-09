# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file

"""API-surface tests for positional sparse prefill (prompt_token_positions).

Phase 1 covers only the public-API threading and validation of the new
``prompt_token_positions`` argument; the prefill/decode mechanics land in later
phases. These tests are model-free and fast.
"""

from __future__ import annotations

import inspect

import pytest

from dflash_mlx.engine.spec_epoch import _SessionRequest, stream_dflash_generate_impl
from dflash_mlx.runtime import stream_dflash_generate


def _make_request(prompt_tokens, positions, *, prefix_cache_active=False):
    return _SessionRequest.from_tokens(
        prompt_tokens=prompt_tokens,
        max_new_tokens=8,
        block_tokens=None,
        stop_token_ids=None,
        suppress_token_ids=None,
        prefix_snapshot=None,
        snapshot_service=None,
        stable_prefix_len=None,
        prefix_cache_active=prefix_cache_active,
        prompt_token_positions=positions,
    )


class TestSessionRequestPositions:
    def test_none_positions_is_default_dense_behavior(self):
        req = _make_request([10, 11, 12, 13], None)
        assert req.prompt_token_positions is None
        assert req.prompt_len == 4

    def test_valid_positions_stored_as_tuple(self):
        req = _make_request([10, 11, 12], [0, 5, 9])
        assert req.prompt_token_positions == (0, 5, 9)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            _make_request([10, 11, 12], [0, 5])

    def test_non_increasing_positions_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            _make_request([10, 11, 12], [0, 5, 5])

    def test_descending_positions_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            _make_request([10, 11, 12], [9, 5, 0])

    def test_negative_position_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            _make_request([10, 11], [-1, 3])

    def test_positions_with_active_prefix_cache_rejected(self):
        with pytest.raises(ValueError, match="prefix cache"):
            _make_request([10, 11, 12], [0, 5, 9], prefix_cache_active=True)

    def test_select_all_positions_allowed(self):
        # The correctness anchor: selecting every token at its own index is valid.
        req = _make_request([10, 11, 12, 13], [0, 1, 2, 3])
        assert req.prompt_token_positions == (0, 1, 2, 3)


class TestPublicApiSignature:
    def test_stream_dflash_generate_accepts_prompt_token_positions(self):
        sig = inspect.signature(stream_dflash_generate)
        assert "prompt_token_positions" in sig.parameters
        assert sig.parameters["prompt_token_positions"].default is None

    def test_impl_accepts_prompt_token_positions(self):
        sig = inspect.signature(stream_dflash_generate_impl)
        assert "prompt_token_positions" in sig.parameters
        assert sig.parameters["prompt_token_positions"].default is None

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import mlx.core as mx

def match_acceptance_length(
    drafted_tokens: mx.array,
    posterior_tokens: mx.array,
) -> mx.array:
    if int(drafted_tokens.shape[0]) == 0:
        return mx.array(0, dtype=mx.int32)
    matches = mx.equal(drafted_tokens, posterior_tokens).astype(mx.int32)
    return mx.sum(mx.cumprod(matches, axis=0))

# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file

import math

import mlx.core as mx

from dflash_mlx.engine.sampling import greedy_tokens_with_mask, masked_topk_arrays


def test_masked_topk_arrays_orders_masks_and_matches_greedy():
    logits = mx.array(
        [
            [0.0, 3.0, 1.0, 2.0, -1.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    suppress = mx.array([False, False, False, True, False])

    top_ids, top_logprobs = masked_topk_arrays(logits, suppress, width=2)

    # id 3 is suppressed: row 0 ranks id1 (3.0) then id2 (1.0).
    assert top_ids.tolist() == [[1, 2], [0, 1]]
    greedy = greedy_tokens_with_mask(logits, suppress)
    assert [int(t) for t in greedy.tolist()] == [row[0] for row in top_ids.tolist()]

    # Values are the log-softmax of the masked row (suppressed mass ~ 0).
    kept = [0.0, 3.0, 1.0, -1.0]
    z = math.log(sum(math.exp(v) for v in kept))
    row0 = top_logprobs.tolist()[0]
    assert abs(row0[0] - (3.0 - z)) < 1e-4
    assert abs(row0[1] - (1.0 - z)) < 1e-4


def test_masked_topk_arrays_no_mask_full_width():
    logits = mx.array([[1.0, 2.0, 0.5]])
    top_ids, top_logprobs = masked_topk_arrays(logits, None, width=3)
    assert top_ids.tolist() == [[1, 0, 2]]
    z = math.log(sum(math.exp(v) for v in [1.0, 2.0, 0.5]))
    vals = top_logprobs.tolist()[0]
    assert abs(vals[0] - (2.0 - z)) < 1e-4
    assert abs(vals[2] - (0.5 - z)) < 1e-4

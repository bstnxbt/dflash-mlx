# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""Real-model parity for positional sparse prefill.

Gated behind DFLASH_RUN_REAL_MODEL_TESTS=1 (and a locally-cached model) like the
other real-model suites. The key correctness anchor: with select-all positions
(range(L)), PositionMappedRoPE must reproduce the model's native RoPE exactly, so
a sparse-RoPE forward equals a dense forward to numerical tolerance.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx
import pytest

from dflash_mlx.engine.sparse_rope import (
    install_position_mapped_rope,
    restore_ropes,
)
from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.engine.events import SummaryEvent, TokenEvent

pytestmark = pytest.mark.skipif(
    os.environ.get("DFLASH_RUN_REAL_MODEL_TESTS") != "1",
    reason="set DFLASH_RUN_REAL_MODEL_TESTS=1 to run local real-model parity tests",
)


def _local_model_path() -> str:
    repo_id = os.environ.get(
        "DFLASH_REAL_QWEN_MODEL",
        "Qwen/Qwen3-0.6B",
    )
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"huggingface_hub unavailable: {exc}")
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"local Qwen model not present for {repo_id}: {exc}")


@lru_cache(maxsize=1)
def _load_model():
    from mlx_lm.utils import load

    return load(_local_model_path(), lazy=True)


def _prompt_ids(tokenizer, n=12):
    ids = list(tokenizer.encode("Write one short sentence about the sea and the sky."))
    if len(ids) < n:
        ids = (ids * n)[:n]
    return ids[:n]


def _forward_logits(ops, model, ids):
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    logits, _ = ops.forward_with_hidden_capture(
        model,
        input_ids=mx.array(ids, dtype=mx.uint32)[None],
        cache=cache,
        logits_last_only=True,
    )
    mx.eval(logits)
    return logits


def test_select_all_forward_matches_dense():
    model, tokenizer = _load_model()
    ops = resolve_target_ops(model)
    ids = _prompt_ids(tokenizer)

    dense = _forward_logits(ops, model, ids)

    text_model = ops.text_model(model)
    positions = mx.arange(0, len(ids), dtype=mx.int32)
    saved = install_position_mapped_rope(text_model, positions, cache_start=0)
    try:
        sparse = _forward_logits(ops, model, ids)
    finally:
        restore_ropes(saved)

    mx.eval(dense, sparse)
    max_abs = float(mx.abs(dense.astype(mx.float32) - sparse.astype(mx.float32)).max())
    # select-all => PositionMappedRoPE is the identity over native RoPE.
    assert max_abs <= 5e-3, f"select-all sparse RoPE diverged from dense: {max_abs}"


def test_noncontiguous_sparse_prefill_runs_and_is_finite():
    # Exercises the manual_rope path on real weights: select a non-contiguous
    # subset placed at its original positions. We can't assert an exact target
    # (dropped tokens are an approximation), but it must run, stay finite, and
    # decode to a valid token id.
    model, tokenizer = _load_model()
    ops = resolve_target_ops(model)
    full_ids = _prompt_ids(tokenizer, n=16)
    keep = [0, 1, 2, 5, 9, 12, 14, 15]  # non-contiguous, keeps the final token
    sel_ids = [full_ids[i] for i in keep]

    text_model = ops.text_model(model)
    saved = install_position_mapped_rope(
        text_model, mx.array(keep, dtype=mx.int32), cache_start=0
    )
    try:
        logits = _forward_logits(ops, model, sel_ids)
    finally:
        restore_ropes(saved)

    mx.eval(logits)
    assert bool(mx.all(mx.isfinite(logits.astype(mx.float32))))
    tok_id = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    vocab = int(logits.shape[-1])
    assert 0 <= tok_id < vocab


def test_restore_returns_native_trajectory():
    model, tokenizer = _load_model()
    ops = resolve_target_ops(model)
    ids = _prompt_ids(tokenizer)

    before = _forward_logits(ops, model, ids)
    text_model = ops.text_model(model)
    saved = install_position_mapped_rope(
        text_model, mx.arange(0, len(ids), dtype=mx.int32)
    )
    restore_ropes(saved)
    after = _forward_logits(ops, model, ids)

    mx.eval(before, after)
    assert float(mx.abs(before.astype(mx.float32) - after.astype(mx.float32)).max()) == 0.0


# --- End-to-end generate parity (full target + draft DFlash pipeline) ---

_E2E_TARGET = os.environ.get("DFLASH_E2E_TARGET", "Qwen/Qwen3-4B")
_E2E_DRAFT = os.environ.get("DFLASH_E2E_DRAFT", "z-lab/Qwen3-4B-DFlash-b16")


def _load_bundle():
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context

    rc = build_offline_runtime_context(
        prefill_step_size=2048, target_fa_window=0, draft_sink_size=64,
        draft_window_size=1024, verify_len_cap=0, verify_mode="dflash",
        copyspec_mode="off",
    )
    try:
        bundle = load_runtime_bundle(
            model_ref=_E2E_TARGET, draft_ref=_E2E_DRAFT, verify_config=rc.verify
        )
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"DFlash bundle unavailable ({_E2E_TARGET} + {_E2E_DRAFT}): {exc}")
    return bundle, rc


def _generate_ids(bundle, rc, prompt_tokens, positions, max_new=24):
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate

    kwargs = dict(
        target_model=bundle.target_model, target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer, draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend, prompt="", max_new_tokens=max_new,
        stop_token_ids=get_stop_token_ids(bundle.tokenizer),
        prompt_tokens_override=prompt_tokens, runtime_context=rc,
    )
    if positions is not None:
        kwargs["prompt_token_positions"] = positions
    out = []
    stream = stream_dflash_generate(**kwargs)
    try:
        for ev in stream:
            if isinstance(ev, TokenEvent):
                out.append(int(ev.token_id))
            elif isinstance(ev, SummaryEvent):
                break
    finally:
        stream.close()
    return out


def test_e2e_select_all_generate_matches_dense():
    # The whole run_events lifecycle (install -> switch -> restore) must be a
    # no-op for select-all positions: byte-identical generated token ids.
    bundle, rc = _load_bundle()
    ids = list(bundle.tokenizer.encode(
        "Explain in one short paragraph why the sky appears blue during the day."
    ))
    dense = _generate_ids(bundle, rc, ids, None)
    sparse = _generate_ids(bundle, rc, ids, list(range(len(ids))))
    assert sparse == dense, f"select-all diverged from dense:\n{dense}\n{sparse}"


def test_e2e_chunk_sparse_generate_runs():
    # A genuine chunk selection runs end-to-end and yields valid tokens.
    bundle, rc = _load_bundle()
    ids = list(bundle.tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. " * 8
    ))
    n = len(ids)
    keep = sorted(set(list(range(0, n // 3)) + list(range(2 * n // 3, n))))
    sel = [ids[i] for i in keep]
    out = _generate_ids(bundle, rc, sel, keep)
    assert len(out) > 0
    vocab = int(bundle.target_ops.text_model(bundle.target_model).embed_tokens.weight.shape[0])
    assert all(0 <= t < vocab for t in out)

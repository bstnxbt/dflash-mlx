# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from dflash_mlx.cache.codecs import _build_target_hidden_chunks
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.engine.prefill import init_target_hidden_from_snapshot

def _make_dummy_key() -> DFlashPrefixKey:
    return DFlashPrefixKey(
        target_model_id="test-target",
        draft_model_id="test-draft",
        capture_layer_ids=(0, 1),
        draft_sink_size=4,
        draft_window_size=2048,
        target_fa_window=0,
    )

def _make_snap(chunks, spans, total_len) -> DFlashPrefixSnapshot:
    return DFlashPrefixSnapshot(
        token_ids=tuple(range(total_len)),
        fa_states=tuple(),
        gdn_states=tuple(),
        target_hidden_chunks=chunks,
        target_hidden_chunk_spans=spans,
        target_hidden_total_len=total_len,
        last_logits=None,
        key=_make_dummy_key(),
        kind="prefill",
    )

def _build_full_snap(target_hidden: mx.array) -> DFlashPrefixSnapshot:
    chunks, spans, total_len = _build_target_hidden_chunks(
        target_hidden,
        trim_target_hidden=False,
    )
    return _make_snap(chunks, spans, total_len)

def _build_trim_snap(target_hidden: mx.array) -> DFlashPrefixSnapshot:
    chunks, spans, total_len = _build_target_hidden_chunks(target_hidden)
    return _make_snap(chunks, spans, total_len)

def _slice_window(arr: mx.array, sink: int, window: int) -> tuple[mx.array, mx.array]:
    n = int(arr.shape[1])
    sink_part = arr[:, :sink, :]
    tail_start = max(sink, n - window)
    tail_part = arr[:, tail_start:n, :]
    return sink_part, tail_part

def _check(case: str, total_len: int, sink: int, window: int) -> tuple[bool, str]:

    pos = mx.arange(total_len, dtype=mx.float32).reshape(1, total_len, 1)
    feat = mx.arange(64, dtype=mx.float32).reshape(1, 1, 64)
    full = pos * 1000.0 + feat
    mx.eval(full)

    full_snap = _build_full_snap(full)
    trim_snap = _build_trim_snap(full)

    full_bytes = sum(int(c.nbytes) for c in full_snap.target_hidden_chunks)
    trim_bytes = sum(int(c.nbytes) for c in trim_snap.target_hidden_chunks)

    full_hyd = init_target_hidden_from_snapshot(full_snap, total_len, total_len)
    trim_hyd = init_target_hidden_from_snapshot(trim_snap, total_len, total_len)
    mx.eval(full_hyd, trim_hyd)

    if full_hyd.shape != trim_hyd.shape:
        return False, f"shape mismatch full={full_hyd.shape} trim={trim_hyd.shape}"

    fs, ft = _slice_window(full_hyd, sink, window)
    ts, tt = _slice_window(trim_hyd, sink, window)
    sink_max = float(mx.abs(fs - ts).max())
    tail_max = float(mx.abs(ft - tt).max())
    if sink_max > 0.0 or tail_max > 0.0:
        return False, f"readable region differs sink_max={sink_max} tail_max={tail_max}"

    if total_len > sink + window:
        gap_start, gap_end = sink, total_len - window
        full_gap = full_hyd[:, gap_start:gap_end, :]
        trim_gap = trim_hyd[:, gap_start:gap_end, :]
        full_gap_max = float(mx.abs(full_gap).max())
        trim_gap_max = float(mx.abs(trim_gap).max())
        if trim_gap_max != 0.0:
            return False, f"trim gap not zero: max={trim_gap_max}"
        if full_gap_max == 0.0:
            return False, "full gap unexpectedly all zero (test data degenerate?)"

    expected_no_trim = total_len <= sink + window
    if expected_no_trim:

        if full_bytes != trim_bytes:
            return False, f"no-trim regime size mismatch full={full_bytes} trim={trim_bytes}"
    else:

        ratio = full_bytes / trim_bytes
        ratio_floor = max(1.0, total_len / (sink + window) - 0.05)
        if ratio < ratio_floor:
            return False, f"trim ratio={ratio:.2f} below floor {ratio_floor:.2f}"

    saved = full_bytes - trim_bytes
    return True, (
        f"OK | full={full_bytes/1e6:.2f}MB trim={trim_bytes/1e6:.2f}MB "
        f"saved={saved/1e6:.2f}MB ({100*saved/full_bytes:.0f}%) "
        f"sink_diff=0 tail_diff=0"
    )

def main() -> int:
    from dflash_mlx.runtime_context import runtime_config_from_profile

    cfg = runtime_config_from_profile(profile="balanced")
    sink, window = cfg.draft_sink_size, cfg.draft_window_size
    print(f"resolved draft window: sink={sink} window={window}")
    cases = [
        ("tiny (no trim)", 1024, sink, window),
        ("just below threshold", sink + window - 1, sink, window),
        ("exactly threshold", sink + window, sink, window),
        ("just above threshold", sink + window + 1, sink, window),
        ("4k", 4096, sink, window),
        ("8k", 8192, sink, window),
        ("16k", 16384, sink, window),
        ("32k", 32000, sink, window),
        ("60k", 60000, sink, window),
    ]
    all_ok = True
    for name, n, s, w in cases:
        ok, msg = _check(name, n, s, w)
        sym = "✓" if ok else "✗"
        print(f"  {sym} {name:25s} N={n:>6d} | {msg}")
        if not ok:
            all_ok = False
    if not all_ok:
        print("FAIL")
        return 1
    print("ALL OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

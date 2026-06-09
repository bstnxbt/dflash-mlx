# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dflash_mlx.benchmark import _sustained_summary, build_parser


def _gen(idx: int, t_start: float, tok_s: float, wall: float = 30.0) -> dict:
    return {
        "gen_index": idx,
        "t_start_s": t_start,
        "wall_s": wall,
        "generation_tokens": int(tok_s * wall),
        "decode_tok_s": tok_s,
        "tokens_per_cycle": 4.0,
    }


def test_sustained_summary_reports_fresh_plateau_and_cliff() -> None:
    # 107 fresh, decay past the cliff at ~190s, plateau ~60 for the rest
    rates = [107.0, 96.0, 88.0, 80.0, 74.0, 60.0, 52.0] + [60.0] * 17
    gens = [_gen(i, i * 30.0, r) for i, r in enumerate(rates)]

    s = _sustained_summary(gens)

    assert s["generations"] == 24
    assert s["fresh_tok_s"] == 107.0
    # first gen below 0.8 * 107 = 85.6 is index 3 (80.0) at t=90s
    assert s["cliff_s"] == 90.0
    # plateau = gens starting in the last 5 minutes (t >= 420-300=300... t>=max(300, 720-300=420))
    plateau_rows = [g for g in gens if g["t_start_s"] >= max(300.0, gens[-1]["t_start_s"] + 30.0 - 300.0)]
    assert s["plateau_tok_s"] == round(
        sum(g["decode_tok_s"] for g in plateau_rows) / len(plateau_rows), 2
    )
    assert s["throttle_factor"] == round(107.0 / s["plateau_tok_s"], 3)


def test_sustained_summary_without_plateau_window_is_partial() -> None:
    gens = [_gen(0, 0.0, 100.0), _gen(1, 30.0, 95.0)]

    s = _sustained_summary(gens)

    assert s["generations"] == 2
    assert s["fresh_tok_s"] == 100.0
    assert s["cliff_s"] is None
    assert "plateau_tok_s" not in s


def test_sustained_summary_empty() -> None:
    assert _sustained_summary([]) == {"generations": 0}


def test_benchmark_parser_accepts_sustained_minutes() -> None:
    args = build_parser().parse_args(
        ["--model", "m", "--sustained-minutes", "12", "--max-tokens", "2048"]
    )
    assert args.sustained_minutes == 12.0

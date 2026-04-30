# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json

from tools.benchmarks._agentic_trace import (
    _aggregate,
    _render_peer_comparison,
    derive_post_landmarks,
    summarize_cycles,
)

def test_summarize_cycles_reports_tokens_per_cycle():
    summary = summarize_cycles(
        [
            {
                "commit_count": 3,
                "verify_us": 1000,
                "block_len": 16,
                "acceptance_len": 2,
            },
            {
                "commit_count": 5,
                "verify_us": 2000,
                "block_len": 16,
                "acceptance_len": 4,
            },
        ]
    )

    assert summary is not None
    assert summary["n_cycles"] == 2
    assert summary["total_commits"] == 8
    assert summary["tokens_per_cycle"] == 4.0

def test_derive_post_landmarks_counts_unique_streamed_tool_calls(tmp_path):
    sse_path = tmp_path / "001.jsonl"
    events = [
        {"type": "first_byte", "t_ms": 10.0},
        {
            "type": "event",
            "t_ms": 20.0,
            "payload": {
                "data": {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": "{\"path\""},
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        },
        {
            "type": "event",
            "t_ms": 25.0,
            "payload": {
                "data": {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": ": \"a.py\"}"},
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        },
        {
            "type": "event",
            "t_ms": 30.0,
            "payload": {
                "data": {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 1,
                                        "function": {"arguments": "{\"cmd\": \"pytest\"}"},
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        },
    ]
    sse_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    landmarks = derive_post_landmarks(sse_path)

    assert landmarks["first_tool_call_sent_ms"] == 20.0
    assert landmarks["tool_call_complete_ms"] == 30.0
    assert landmarks["tool_call_delta_count"] == 3
    assert landmarks["tool_call_count"] == 2
    assert landmarks["finish_reason"] == "tool_calls"

def test_aggregate_reports_cycle_and_tool_totals():
    posts = [
        {
            "landmarks": {"tool_call_count": 2, "first_tool_call_sent_ms": 100.0},
            "server_metric": {
                "prompt_tokens": 10,
                "tokens": 6,
                "wall_s": 0.5,
                "accept": 0.75,
                "cache_hit_tokens": 8,
                "cycles_summary": {"n_cycles": 2, "total_commits": 6},
            },
        }
    ]

    totals = _aggregate(posts, wall_s=1.0)

    assert totals["total_cycles"] == 2
    assert totals["total_cycle_commits"] == 6
    assert totals["avg_tokens_per_cycle"] == 3.0
    assert totals["total_tool_calls"] == 2

def _comparison_summary(label: str, backend: str, post_count: int, decode_tokens: int) -> dict:
    posts = [
        {"landmarks": {"first_tool_call_sent_ms": 100.0 + idx}}
        for idx in range(post_count)
    ]
    return {
        "metadata": {"label": label, "backend": backend},
        "posts": posts,
        "totals": {
            "wall_s": 10.0,
            "total_decode_tokens": decode_tokens,
            "total_decode_wall_s": decode_tokens / 100.0,
            "decode_tps_avg": 100.0,
            "total_prompt_tokens": 1000,
            "total_tool_calls": post_count,
            "total_cache_hit_tokens": 128 if backend == "dflash" else 0,
            "prefill_tokens_saved_cumulative": 512 if backend == "dflash" else None,
            "weighted_acceptance": 0.75 if backend == "dflash" else None,
            "first_tool_call_ms_sum": sum(100.0 + idx for idx in range(post_count)),
        },
    }

def test_render_peer_comparison_keeps_per_post_table_when_trajectories_align():
    rendered = _render_peer_comparison(
        _comparison_summary("dflash", "dflash", 4, 100),
        _comparison_summary("mlxlm", "mlxlm", 4, 104),
    )

    assert "Trajectory-invariant metrics" in rendered
    assert "Trajectories aligned" in rendered
    assert "| # | this | peer | gap_ms |" in rendered

def test_render_peer_comparison_skips_per_post_table_when_trajectories_diverge():
    rendered = _render_peer_comparison(
        _comparison_summary("dflash", "dflash", 9, 220),
        _comparison_summary("mlxlm", "mlxlm", 4, 100),
    )

    assert "Trajectory-invariant metrics" in rendered
    assert "TRAJECTORY DIVERGED" in rendered
    assert "Skipped — trajectories diverged" in rendered
    assert "| # | this | peer | gap_ms |" not in rendered

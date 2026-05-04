# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import pytest

from dflash_mlx import generate
from dflash_mlx.runtime import parse_draft_quant_spec

def test_generate_cli_passes_verify_mode(monkeypatch):
    calls = []
    metal_calls = []

    monkeypatch.setattr(generate, "apply_metal_limits", lambda: metal_calls.append(True))
    monkeypatch.setattr(generate, "run_generate", lambda **kwargs: calls.append(kwargs) or 0)

    with pytest.raises(SystemExit) as exc:
        generate.main(
            [
                "--model",
                "m",
                "--prompt",
                "p",
                "--verify-mode",
                "off",
                "--prefill-step-size",
                "8192",
                "--draft-sink-size",
                "32",
                "--draft-window-size",
                "512",
                "--verify-len-cap",
                "8",
                "--draft-quant",
                "w4",
            ]
        )

    assert exc.value.code == 0
    assert metal_calls == [True]
    assert calls == [
        {
            "model_ref": "m",
            "prompt": "p",
            "max_tokens": 2048,
            "use_chat_template": True,
            "draft_ref": None,
            "target_fa_window": 0,
            "prefill_step_size": 8192,
            "draft_sink_size": 32,
            "draft_window_size": 512,
            "verify_len_cap": 8,
            "verify_mode": "off",
            "draft_quant": "w4",
            "bod_enabled": False,
            "bod_mode": "chain",
            "bod_min_bet": 2,
            "bod_max_bet": 16,
            "bod_default_scale_cost": 8.0,
            "bod_default_fixed_cost": 47.0,
        }
    ]

def test_generate_cli_rejects_invalid_prefill_step_size(monkeypatch):
    monkeypatch.setattr(generate, "apply_metal_limits", lambda: None)

    with pytest.raises(SystemExit) as exc:
        generate.main(
            [
                "--model",
                "m",
                "--prompt",
                "p",
                "--prefill-step-size",
                "0",
            ]
        )

    assert exc.value.code == "--prefill-step-size must be > 0"

def test_draft_quant_parser_no_env_fallback(monkeypatch):
    monkeypatch.setenv("DFLASH_DRAFT_QUANT", "w4")
    from dflash_mlx.runtime import _resolve_draft_quant

    assert _resolve_draft_quant(None) is None
    assert parse_draft_quant_spec("w4").weight_bits == 4

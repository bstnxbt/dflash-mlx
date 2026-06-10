# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from dflash_mlx import generate
import dflash_mlx.runtime.config as runtime_config
from dflash_mlx.engine.events import PrefillCompleteEvent, SummaryEvent
from dflash_mlx.runtime.loading import parse_draft_quant_spec


def test_decode_token_prefers_list_decode():
    calls = []

    class _Tokenizer:
        def decode(self, token):
            calls.append(token)
            return "list" if isinstance(token, list) else "scalar"

    assert generate.decode_token(_Tokenizer(), 7) == "list"
    assert calls == [[7]]


def test_decode_token_falls_back_to_scalar_decode_on_type_error():
    calls = []

    class _Tokenizer:
        def decode(self, token):
            calls.append(token)
            if isinstance(token, list):
                raise TypeError("scalar required")
            return f"token-{token}"

    assert generate.decode_token(_Tokenizer(), 7) == "token-7"
    assert calls == [[7], 7]


def test_decode_token_propagates_unexpected_decode_errors():
    calls = []

    class _Tokenizer:
        def decode(self, token):
            calls.append(token)
            raise RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        generate.decode_token(_Tokenizer(), 7)

    assert calls == [[7]]


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
            "quantize_kv_cache": False,
            "prefill_step_size": 8192,
            "draft_sink_size": 32,
            "draft_window_size": 512,
            "draft_full_context_min_ctx": 16384,
            "verify_len_cap": 8,
            "verify_mode": "off",
            "copyspec_mode": "conservative",
            "draft_quant": "w4",
        }
    ]


def test_run_generate_defaults_follow_single_runtime_default(monkeypatch):
    monkeypatch.setattr(
        runtime_config,
        "DEFAULT_RUNTIME_CONFIG",
        replace(runtime_config.DEFAULT_RUNTIME_CONFIG, draft_window_size=1536),
    )
    captured_runtime = []
    captured_load = []

    class _Tokenizer:
        def decode(self, token):
            return str(token)

    def _load_runtime_bundle(**kwargs):
        captured_load.append(kwargs)
        return SimpleNamespace(
            target_model=object(),
            tokenizer=_Tokenizer(),
            draft_model=object(),
            draft_backend=object(),
            target_ops=object(),
        )

    monkeypatch.setattr(
        generate,
        "load_runtime_bundle",
        _load_runtime_bundle,
    )
    monkeypatch.setattr(generate, "get_stop_token_ids", lambda _tokenizer: [])

    def _stream(**kwargs):
        captured_runtime.append(kwargs["runtime_context"])
        return _ClosableGenerateStream(
            [
                SummaryEvent(
                    elapsed_us=1.0,
                    prompt_token_count=1,
                    generated_token_ids=(),
                    generation_tokens=0,
                    accepted_from_draft=0,
                    acceptance_ratio=0.0,
                    cycles_completed=0,
                    phase_timings_us={},
                )
            ]
        )

    monkeypatch.setattr(generate, "stream_dflash_generate", _stream)

    assert (
        generate.run_generate(
            model_ref="m",
            prompt="p",
            max_tokens=1,
            use_chat_template=False,
            draft_ref=None,
            draft_quant=None,
        )
        == 0
    )
    assert captured_runtime[0].runtime.draft_window_size == 1536
    assert "split_full_attention_sdpa" not in captured_load[0]


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


class _ClosableGenerateStream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


def test_run_generate_rejects_stale_dict_engine_event(monkeypatch):
    class _Tokenizer:
        def decode(self, token):
            return str(token)

    monkeypatch.setattr(
        generate,
        "load_runtime_bundle",
        lambda **_kwargs: SimpleNamespace(
            target_model=object(),
            tokenizer=_Tokenizer(),
            draft_model=object(),
            draft_backend=object(),
            target_ops=object(),
        ),
    )
    stream = _ClosableGenerateStream(
        [
            PrefillCompleteEvent(
                prefill_us=1.0,
                prompt_token_count=1,
                logical_ctx_tokens=1,
                physical_prefill_tokens=1,
                prefill_tokens_restored=0,
                prefill_tokens_computed=1,
            ),
            {"event": "token", "token_id": 7},
        ]
    )
    monkeypatch.setattr(generate, "get_stop_token_ids", lambda _tokenizer: [])
    monkeypatch.setattr(generate, "stream_dflash_generate", lambda **_kwargs: stream)

    with pytest.raises(TypeError, match="Unsupported DFlash engine event: dict"):
        generate.run_generate(
            model_ref="m",
            prompt="p",
            max_tokens=1,
            use_chat_template=False,
            draft_ref=None,
            target_fa_window=0,
            quantize_kv_cache=False,
            prefill_step_size=1,
            draft_sink_size=1,
            draft_window_size=1,
            verify_len_cap=0,
            verify_mode="off",
            draft_quant=None,
        )

    assert stream.closed is True


def test_draft_quant_parser_no_env_fallback(monkeypatch):
    monkeypatch.setenv("DFLASH_DRAFT_QUANT", "w4")
    from dflash_mlx.runtime.loading import _resolve_draft_quant

    assert _resolve_draft_quant(None) is None
    assert parse_draft_quant_spec("w4").weight_bits == 4

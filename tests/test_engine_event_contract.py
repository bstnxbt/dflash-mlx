
from __future__ import annotations

import inspect
import re

from dflash_mlx import runtime, serve
from dflash_mlx.engine import events
from dflash_mlx.engine import types as engine_types


def _impl_source() -> str:
    return inspect.getsource(runtime._stream_dflash_generate_impl)


def _baseline_source() -> str:
    return inspect.getsource(runtime.stream_baseline_generate)


def test_summary_typeddict_keys_are_emitted():
    impl_src = _impl_source() + _baseline_source()
    for key in engine_types.StreamSummary.__optional_keys__:
        assert f'"{key}"' in impl_src, (
            f"StreamSummary key {key!r} is not emitted by any engine path"
        )


def test_token_typeddict_keys_are_emitted():
    impl_src = _impl_source() + _baseline_source()
    for key in engine_types.TokenEvent.__optional_keys__:
        assert f'"{key}"' in impl_src, (
            f"TokenEvent key {key!r} is not emitted by any engine path"
        )


def test_prefill_snapshot_typeddict_keys_are_emitted():
    impl_src = _impl_source()
    for key in engine_types.PrefillSnapshotReady.__optional_keys__:
        assert f'"{key}"' in impl_src, (
            f"PrefillSnapshotReady key {key!r} is not emitted"
        )


def test_event_name_constants_match_emitted_strings():
    impl_src = _impl_source() + _baseline_source()
    for name in events.ALL_EVENT_NAMES:
        assert f'"event": "{name}"' in impl_src, (
            f"event constant {name!r} is not emitted by any engine path"
        )


def test_streaming_impl_resolves():
    assert callable(runtime._stream_dflash_generate_impl)
    assert callable(runtime.stream_dflash_generate)
    wrapper_src = inspect.getsource(runtime.stream_dflash_generate)
    assert "_stream_dflash_generate_impl" in wrapper_src


def test_streaming_impl_yields_required_event_names():
    src = _impl_source()
    for name in (
        "prefill",
        "prefill_progress",
        "prefill_snapshot_ready",
        "generation_snapshot_ready",
        "token",
        "summary",
    ):
        assert f'"event": "{name}"' in src, f"missing event {name!r} in impl"


def test_baseline_yields_prefill_token_summary_in_order():
    src = _baseline_source()
    prefill_pos = src.index('"event": "prefill"')
    summary_pos = src.index('"event": "summary"')
    token_positions = [m.start() for m in re.finditer(r'"event": "token"', src)]
    assert prefill_pos < min(token_positions) < max(token_positions) < summary_pos


def test_token_event_carries_keys_serve_reads():
    src = _impl_source()
    token_block = _slice_event(src, "token")
    for key in ("token_id", "generated_tokens", "acceptance_ratio", "cycles_completed"):
        assert f'"{key}"' in token_block, f"token event missing {key!r}"


def test_summary_event_carries_keys_serve_reads():
    src = _impl_source()
    summary_block = _slice_event(src, "summary")
    for key in (
        "elapsed_us",
        "generation_tokens",
        "generated_token_ids",
        "acceptance_ratio",
        "cycles_completed",
        "phase_timings_us",
    ):
        assert f'"{key}"' in summary_block, f"summary event missing {key!r}"


def test_prefill_snapshot_ready_carries_cache_payload():
    src = _impl_source()
    block = _slice_event(src, "prefill_snapshot_ready")
    for key in ("target_cache", "target_hidden", "last_logits", "token_ids"):
        assert f'"{key}"' in block, f"prefill_snapshot_ready missing {key!r}"


def test_generation_snapshot_ready_carries_cache_payload():
    src = _impl_source()
    block = _slice_event(src, "generation_snapshot_ready")
    for key in ("target_cache", "target_hidden", "last_logits", "token_ids"):
        assert f'"{key}"' in block, f"generation_snapshot_ready missing {key!r}"


def test_prefill_progress_carries_processed_total():
    src = _impl_source()
    block = _slice_event(src, "prefill_progress")
    assert '"tokens_processed"' in block
    assert '"tokens_total"' in block


def test_serve_consumes_only_known_event_names():
    serve_src = inspect.getsource(serve)
    consumed = set(re.findall(r'event\.get\("event"\)\s*[!=]=\s*"([a-z_]+)"', serve_src))
    consumed.update(
        m for m in re.findall(r'"([a-z_]+)"', serve_src)
        if m in {
            "prefill",
            "prefill_progress",
            "prefill_snapshot_ready",
            "generation_snapshot_ready",
            "token",
            "cycle_complete",
            "summary",
        }
    )
    assert "prefill" in consumed
    assert "summary" in consumed
    impl_src = _impl_source() + _baseline_source()
    for name in consumed:
        assert f'"event": "{name}"' in impl_src, (
            f"serve.py reads {name!r} but no engine path emits it"
        )


def _slice_event(src: str, name: str) -> str:
    needle = f'"event": "{name}"'
    pos = src.index(needle)
    start = src.rfind("yield {", 0, pos)
    if start < 0:
        start = src.rfind("{", 0, pos)
    end = src.index("}", pos)
    return src[start : end + 1]

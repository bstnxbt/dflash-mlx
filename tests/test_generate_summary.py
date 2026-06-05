from __future__ import annotations

from typing import Any

from dflash_mlx.engine.events import SummaryEvent
from dflash_mlx.generate import format_generation_summary


def _summary(**overrides: Any) -> SummaryEvent:
    defaults: dict[str, Any] = dict(
        elapsed_us=2_000_000.0,
        prompt_token_count=10,
        generated_token_ids=(),
        generation_tokens=100,
        accepted_from_draft=75,
        acceptance_ratio=0.75,
        cycles_completed=25,
        phase_timings_us={"prefill": 1_000_000.0},
    )
    defaults.update(overrides)
    return SummaryEvent(**defaults)


def test_summary_line_without_copyspec() -> None:
    line = format_generation_summary(_summary())

    assert line == "100 tokens | 100.0 tok/s | 75.0% acceptance"


def test_summary_line_includes_copyspec_when_active() -> None:
    line = format_generation_summary(
        _summary(copyspec_hits=4, copyspec_tokens=60)
    )

    assert line == (
        "100 tokens | 100.0 tok/s | 75.0% acceptance"
        " | copyspec 4 blocks / 60 tokens"
    )

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)
from __future__ import annotations

from typing import Final


PREFILL: Final = "prefill"
PREFILL_PROGRESS: Final = "prefill_progress"
PREFILL_SNAPSHOT_READY: Final = "prefill_snapshot_ready"
GENERATION_SNAPSHOT_READY: Final = "generation_snapshot_ready"
TOKEN: Final = "token"
CYCLE_COMPLETE: Final = "cycle_complete"
SUMMARY: Final = "summary"


ALL_EVENT_NAMES: Final = frozenset(
    {
        PREFILL,
        PREFILL_PROGRESS,
        PREFILL_SNAPSHOT_READY,
        GENERATION_SNAPSHOT_READY,
        TOKEN,
        CYCLE_COMPLETE,
        SUMMARY,
    }
)

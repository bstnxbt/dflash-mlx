# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

@dataclass(frozen=True)
class TraceConfig:
    log_dir: Optional[Path] = None
    cycle_events: bool = False

@dataclass(frozen=True)
class DiagnosticsConfig:
    mode: Literal["off", "basic", "full"] = "off"
    run_dir: Optional[Path] = None
    memory_waterfall: bool = False
    trace: TraceConfig = field(default_factory=TraceConfig)

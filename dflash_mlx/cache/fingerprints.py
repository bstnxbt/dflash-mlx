from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DFlashPrefixKey:
    target_model_id: str
    draft_model_id: str
    capture_layer_ids: tuple[int, ...]
    draft_sink_size: int
    draft_window_size: int
    format_version: int = 1

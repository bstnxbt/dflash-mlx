# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any

from dflash_mlx.draft.dflash2 import (
    DFlash2DraftModel,
    DFlash2DraftModelArgs,
    has_dflash2_architecture,
    normalize_dflash2_config,
)
from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs


def get_draft_model_classes(config: dict[str, Any]):
    if has_dflash2_architecture(config):
        normalize_dflash2_config(config)
        return DFlash2DraftModel, DFlash2DraftModelArgs
    architectures = tuple(str(value) for value in config.get("architectures") or ())
    unknown_dflash = [
        value
        for value in architectures
        if value.startswith("DFlash") and value != "DFlashDraftModel"
    ]
    if unknown_dflash:
        raise ValueError(
            "Unsupported DFlash draft architecture(s): "
            + ", ".join(sorted(unknown_dflash))
        )
    return DFlashDraftModel, DFlashDraftModelArgs

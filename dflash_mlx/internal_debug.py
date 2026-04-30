# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os
from typing import Optional

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)

def verify_linear_override() -> Optional[bool]:
    raw = os.environ.get("DFLASH_VERIFY_LINEAR", "").strip()
    if raw == "1":
        return True
    if raw == "0":
        return False
    return None

def verify_qmm_enabled() -> bool:
    return os.environ.get("DFLASH_VERIFY_QMM", "") == "1"

def verify_qmm_variant() -> str:
    return os.environ.get("DFLASH_VERIFY_VARIANT", "auto")

def verify_qmm_kparts(default: int) -> int:
    return _env_int("DFLASH_VERIFY_QMM_KPARTS", default)

def verify_max_n(default: int) -> int:
    return _env_int("DFLASH_VERIFY_MAX_N", default)

def verify_include() -> str:
    return os.environ.get("DFLASH_VERIFY_INCLUDE", "all").strip().lower()

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

MemoryLimit = Literal["auto", "none"] | int

@dataclass(frozen=True)
class MetalLimitConfig:
    metal_available: bool
    recommended_bytes: int | None
    wired_request: MemoryLimit
    wired_bytes: int | None
    wired_applied: bool
    cache_request: MemoryLimit
    cache_bytes: int | None
    cache_applied: bool

def parse_memory_limit(raw: str) -> MemoryLimit:
    value = str(raw).strip().lower()
    if value in ("auto", "none"):
        return value
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b?|bytes?)?", value)
    if match is None:
        raise argparse.ArgumentTypeError("expected auto, none, or a byte value like 8GB")
    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "": 1,
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }
    if unit not in multipliers:
        raise argparse.ArgumentTypeError("unknown byte suffix")
    bytes_value = int(number * multipliers[unit])
    if bytes_value <= 0:
        raise argparse.ArgumentTypeError("memory limit must be > 0")
    return bytes_value

def apply_metal_limits(
    *,
    wired_request: MemoryLimit = "auto",
    cache_request: MemoryLimit = "auto",
) -> MetalLimitConfig:
    if not mx.metal.is_available():
        return MetalLimitConfig(
            metal_available=False,
            recommended_bytes=None,
            wired_request=wired_request,
            wired_bytes=None,
            wired_applied=False,
            cache_request=cache_request,
            cache_bytes=None,
            cache_applied=False,
        )

    recommended = int(mx.device_info()["max_recommended_working_set_size"])
    wired_bytes = _resolve_wired_limit(wired_request, recommended)
    if wired_bytes is not None:
        mx.set_wired_limit(wired_bytes)

    cache_bytes = _resolve_cache_limit(cache_request, wired_bytes, recommended)
    if cache_bytes is not None:
        mx.set_cache_limit(cache_bytes)

    return MetalLimitConfig(
        metal_available=True,
        recommended_bytes=recommended,
        wired_request=wired_request,
        wired_bytes=wired_bytes,
        wired_applied=wired_bytes is not None,
        cache_request=cache_request,
        cache_bytes=cache_bytes,
        cache_applied=cache_bytes is not None,
    )

def _resolve_wired_limit(request: MemoryLimit, recommended: int) -> int | None:
    if request == "auto":
        return recommended
    if request == "none":
        return None
    return int(request)

def _resolve_cache_limit(
    request: MemoryLimit,
    wired_bytes: int | None,
    recommended: int,
) -> int | None:
    if request == "auto":
        return int((wired_bytes if wired_bytes is not None else recommended) // 4)
    if request == "none":
        return None
    return int(request)

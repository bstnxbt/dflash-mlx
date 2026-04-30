# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

from dflash_mlx.diagnostics import TraceConfig

class _BenchLogger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dir: Optional[str] = None
        self._post_fp = None
        self._cycle_fp = None
        self._cache_fp = None
        self._initialized = False

    def _init_if_needed(self, trace: Optional[TraceConfig]) -> bool:
        desired_dir = trace.log_dir if trace is not None else None
        desired = os.fspath(desired_dir) if desired_dir is not None else None
        if self._initialized and self._dir == desired:
            return self._dir is not None
        with self._lock:
            desired_dir = trace.log_dir if trace is not None else None
            desired = os.fspath(desired_dir) if desired_dir is not None else None
            if self._initialized and self._dir == desired:
                return self._dir is not None
            self._close_locked()
            if not desired:
                self._dir = None
                self._initialized = True
                return False
            try:
                os.makedirs(desired, exist_ok=True)
                self._dir = desired
            except OSError:
                self._dir = None
                self._post_fp = None
                self._cycle_fp = None
                self._cache_fp = None
            self._initialized = True
            return self._dir is not None

    def _close_locked(self) -> None:
        for fp in (self._post_fp, self._cycle_fp, self._cache_fp):
            if fp is not None:
                try:
                    fp.close()
                except OSError:
                    pass
        self._post_fp = None
        self._cycle_fp = None
        self._cache_fp = None

    def _fp_for_kind(self, kind: str):
        if self._dir is None:
            return None
        if kind == "post":
            if self._post_fp is None:
                self._post_fp = open(os.path.join(self._dir, "post_events.jsonl"), "a", buffering=1)
            return self._post_fp
        if kind == "cycle":
            if self._cycle_fp is None:
                self._cycle_fp = open(os.path.join(self._dir, "cycle_events.jsonl"), "a", buffering=1)
            return self._cycle_fp
        if self._cache_fp is None:
            self._cache_fp = open(os.path.join(self._dir, "cache_events.jsonl"), "a", buffering=1)
        return self._cache_fp

    def enabled(self, trace: Optional[TraceConfig]) -> bool:
        return self._init_if_needed(trace)

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        payload.setdefault("ts", time.time())
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                fp = self._fp_for_kind(kind)
                if fp is None:
                    return
                fp.write(line)
            except OSError:
                pass

    def log_post(self, trace: Optional[TraceConfig], **fields: Any) -> None:
        if not self._init_if_needed(trace):
            return
        self._write("post", dict(fields))

    def log_cycle(self, trace: Optional[TraceConfig], **fields: Any) -> None:
        if not self._init_if_needed(trace):
            return
        self._write("cycle", dict(fields))

    def log_cache(self, trace: Optional[TraceConfig], **fields: Any) -> None:
        if not self._init_if_needed(trace):
            return
        self._write("cache", dict(fields))

_LOGGER = _BenchLogger()

def log_post(trace: Optional[TraceConfig], **fields: Any) -> None:
    _LOGGER.log_post(trace, **fields)

def log_cycle(trace: Optional[TraceConfig], **fields: Any) -> None:
    _LOGGER.log_cycle(trace, **fields)

def log_cache(trace: Optional[TraceConfig], **fields: Any) -> None:
    _LOGGER.log_cache(trace, **fields)

def enabled(trace: Optional[TraceConfig]) -> bool:
    return _LOGGER.enabled(trace)

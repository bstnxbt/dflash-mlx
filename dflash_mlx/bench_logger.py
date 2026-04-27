from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

class _BenchLogger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dir: Optional[str] = None
        self._post_fp = None
        self._cycle_fp = None
        self._cache_fp = None
        self._initialized = False

    def _init_if_needed(self) -> bool:
        if self._initialized:
            return self._dir is not None
        with self._lock:
            if self._initialized:
                return self._dir is not None
            raw = os.environ.get("DFLASH_BENCH_LOG_DIR", "").strip()
            if not raw:
                self._initialized = True
                return False
            try:
                os.makedirs(raw, exist_ok=True)
                self._post_fp = open(os.path.join(raw, "post_events.jsonl"), "a", buffering=1)
                self._cycle_fp = open(os.path.join(raw, "cycle_events.jsonl"), "a", buffering=1)
                self._cache_fp = open(os.path.join(raw, "cache_events.jsonl"), "a", buffering=1)
                self._dir = raw
            except OSError:
                self._dir = None
                self._post_fp = None
                self._cycle_fp = None
                self._cache_fp = None
            self._initialized = True
            return self._dir is not None

    def enabled(self) -> bool:
        return self._init_if_needed()

    def _write(self, fp, payload: dict[str, Any]) -> None:
        if fp is None:
            return
        payload.setdefault("ts", time.time())
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                fp.write(line)
            except OSError:
                pass

    def log_post(self, **fields: Any) -> None:
        if not self._init_if_needed():
            return
        self._write(self._post_fp, dict(fields))

    def log_cycle(self, **fields: Any) -> None:
        if not self._init_if_needed():
            return
        self._write(self._cycle_fp, dict(fields))

    def log_cache(self, **fields: Any) -> None:
        if not self._init_if_needed():
            return
        self._write(self._cache_fp, dict(fields))

_LOGGER = _BenchLogger()

def log_post(**fields: Any) -> None:
    _LOGGER.log_post(**fields)

def log_cycle(**fields: Any) -> None:
    _LOGGER.log_cycle(**fields)

def log_cache(**fields: Any) -> None:
    _LOGGER.log_cache(**fields)

def enabled() -> bool:
    return _LOGGER.enabled()

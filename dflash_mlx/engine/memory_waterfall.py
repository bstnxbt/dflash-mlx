# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import platform
import resource
import subprocess
from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.diagnostics import DiagnosticsConfig

_GB = 1_000_000_000.0

def memory_waterfall_enabled(diagnostics: Optional[DiagnosticsConfig] = None) -> bool:
    return bool(diagnostics is not None and diagnostics.memory_waterfall)

def should_sample_cycle(cycle: int) -> bool:
    return int(cycle) in (1, 2, 4, 8) or (int(cycle) > 0 and int(cycle) % 16 == 0)

def collect_memory_waterfall(
    *,
    phase: str,
    target_cache: Any = None,
    draft_cache: Any = None,
    target_hidden: Any = None,
    gen_hidden_chunks: Any = None,
    prefix_cache: Any = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    process = process_memory_bytes()
    target = target_cache_bytes(target_cache)
    draft = draft_cache_bytes(draft_cache)
    prefix = prefix_cache_bytes(prefix_cache)
    hidden_seen: set[int] = set()
    target_hidden_bytes = tree_nbytes(target_hidden, hidden_seen)
    gen_hidden_chunks_bytes = tree_nbytes(gen_hidden_chunks, hidden_seen)
    out: dict[str, Any] = {
        "memory_phase": str(phase),
        **process,
        **target,
        **draft,
        "target_hidden_active_bytes": int(target_hidden_bytes),
        "gen_hidden_chunks_bytes": int(gen_hidden_chunks_bytes),
        **prefix,
    }
    for key, value in list(out.items()):
        if key.endswith("_bytes"):
            out[key[:-6] + "_gb"] = _to_gb(value)
    for key in (
        "rss_bytes",
        "system_wired_bytes",
        "mlx_active_bytes",
        "mlx_cache_bytes",
        "mlx_peak_bytes",
        "untracked_bytes",
        "target_fa_kv_bytes",
        "target_gdn_state_bytes",
        "rollback_tape_bytes",
        "draft_kv_bytes",
        "target_hidden_active_bytes",
        "gen_hidden_chunks_bytes",
        "l1_snapshot_bytes",
        "l2_disk_bytes",
    ):
        out.setdefault(key[:-6] + "_gb", 0.0)
    if extra:
        out.update(extra)
    return out

def process_memory_bytes() -> dict[str, int]:
    rss = _current_rss_bytes()
    mlx_active = _mlx_memory_bytes("get_active_memory")
    mlx_cache = _mlx_memory_bytes("get_cache_memory")
    mlx_peak = _mlx_memory_bytes("get_peak_memory")
    return {
        "rss_bytes": int(rss),
        "system_wired_bytes": int(_system_wired_bytes()),
        "mlx_active_bytes": int(mlx_active),
        "mlx_cache_bytes": int(mlx_cache),
        "mlx_peak_bytes": int(mlx_peak),
        "untracked_bytes": int(max(0, rss - mlx_active - mlx_cache)),
    }

def target_cache_bytes(target_cache: Any) -> dict[str, int]:
    out = {
        "target_fa_kv_bytes": 0,
        "target_gdn_state_bytes": 0,
        "rollback_tape_bytes": 0,
    }
    for cache in _iter_cache_entries(target_cache):
        if _looks_recurrent(cache):
            out["target_gdn_state_bytes"] += tree_nbytes(getattr(cache, "cache", None))
            out["rollback_tape_bytes"] += tree_nbytes(
                [
                    getattr(cache, "_tape", None),
                    getattr(cache, "_tape_k", None),
                    getattr(cache, "_tape_g", None),
                    getattr(cache, "_tape_qkv", None),
                    getattr(cache, "_snapshot", None),
                ]
            )
        else:
            out["target_fa_kv_bytes"] += _kv_cache_nbytes(cache)
    return out

def draft_cache_bytes(draft_cache: Any) -> dict[str, int]:
    total = 0
    for cache in _iter_cache_entries(draft_cache):
        total += _kv_cache_nbytes(cache)
    if total == 0:
        total = _kv_cache_nbytes(draft_cache)
    return {"draft_kv_bytes": int(total)}

def prefix_cache_bytes(prefix_cache: Any) -> dict[str, int]:
    out = {
        "l1_snapshot_bytes": 0,
        "l1_snapshot_fa_kv_bytes": 0,
        "l1_snapshot_gdn_state_bytes": 0,
        "l1_snapshot_target_hidden_bytes": 0,
        "l1_snapshot_last_logits_bytes": 0,
        "l2_disk_bytes": 0,
        "prefix_prunes": 0,
        "cross_kind_prunes": 0,
        "byte_budget_evictions": 0,
        "l2_hits": 0,
        "l2_writes": 0,
        "l2_misses": 0,
    }
    if prefix_cache is None:
        return out
    try:
        stats = dict(prefix_cache.stats())
    except Exception:
        stats = {}
    out["l1_snapshot_bytes"] = int(stats.get("current_bytes", 0) or 0)
    out["prefix_prunes"] = int(stats.get("prefix_prunes", 0) or 0)
    out["cross_kind_prunes"] = int(stats.get("cross_kind_prunes", 0) or 0)
    out["byte_budget_evictions"] = int(stats.get("byte_budget_evictions", 0) or 0)
    out["l2_hits"] = int(stats.get("l2_hits", 0) or 0)
    out["l2_misses"] = int(stats.get("l2_misses", 0) or 0)
    l2 = stats.get("l2") if isinstance(stats.get("l2"), dict) else {}
    out["l2_disk_bytes"] = int(l2.get("current_bytes", 0) or 0)
    out["l2_writes"] = int(l2.get("writes", 0) or 0)

    entries = _prefix_entries(prefix_cache)
    if entries:
        fa = gdn = hidden = logits = total = 0
        for snapshot in entries:
            try:
                breakdown = dict(snapshot.nbytes_breakdown())
            except Exception:
                breakdown = {}
            fa += int(breakdown.get("fa_kv", 0) or 0)
            gdn += int(breakdown.get("gdn_state", 0) or 0)
            hidden += int(breakdown.get("target_hidden", 0) or 0)
            logits += int(breakdown.get("last_logits", 0) or 0)
            total += int(getattr(snapshot, "nbytes", 0) or 0)
        out["l1_snapshot_bytes"] = total or out["l1_snapshot_bytes"]
        out["l1_snapshot_fa_kv_bytes"] = fa
        out["l1_snapshot_gdn_state_bytes"] = gdn
        out["l1_snapshot_target_hidden_bytes"] = hidden
        out["l1_snapshot_last_logits_bytes"] = logits
    return out

def merge_memory_waterfall_peak(
    current: Optional[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(current or {})
    for key, value in payload.items():
        if key.endswith("_bytes") or key.endswith("_gb"):
            try:
                if float(value) > float(merged.get(key, -1)):
                    merged[key] = value
            except (TypeError, ValueError):
                pass
        elif key in (
            "prefix_prunes",
            "cross_kind_prunes",
            "byte_budget_evictions",
            "l2_hits",
            "l2_writes",
            "l2_misses",
        ):
            try:
                if int(value) > int(merged.get(key, -1)):
                    merged[key] = int(value)
            except (TypeError, ValueError):
                pass
    return merged

def format_memory_waterfall_summary(payload: dict[str, Any]) -> str:
    buckets = {
        "mlx_active": float(payload.get("mlx_active_gb", 0.0) or 0.0),
        "mlx_cache": float(payload.get("mlx_cache_gb", 0.0) or 0.0),
        "untracked": float(payload.get("untracked_gb", 0.0) or 0.0),
        "target_fa_kv": float(payload.get("target_fa_kv_gb", 0.0) or 0.0),
        "target_gdn_state": float(payload.get("target_gdn_state_gb", 0.0) or 0.0),
        "rollback_tape": float(payload.get("rollback_tape_gb", 0.0) or 0.0),
        "draft_kv": float(payload.get("draft_kv_gb", 0.0) or 0.0),
        "target_hidden": float(payload.get("target_hidden_active_gb", 0.0) or 0.0),
        "gen_hidden_chunks": float(payload.get("gen_hidden_chunks_gb", 0.0) or 0.0),
        "l1_snapshots": float(payload.get("l1_snapshot_gb", 0.0) or 0.0),
        "l2_disk": float(payload.get("l2_disk_gb", 0.0) or 0.0),
    }
    top = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[:5]
    rendered = ",".join(f"{name}:{value:.2f}GB" for name, value in top if value > 0.0)
    if not rendered:
        rendered = "none"
    return f"[dflash] memory-waterfall summary top={rendered}"

def tree_nbytes(value: Any, seen: Optional[set[int]] = None) -> int:
    if value is None:
        return 0
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if _is_array_like(value):
        return int(getattr(value, "nbytes", 0) or 0)
    if isinstance(value, dict):
        return sum(tree_nbytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(tree_nbytes(v, seen) for v in value)
    return 0

def _current_rss_bytes() -> int:
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        if out:
            return int(float(out) * 1024.0)
    except Exception:
        pass
    ru = resource.getrusage(resource.RUSAGE_SELF)
    raw = int(ru.ru_maxrss)
    if platform.system() == "Darwin":
        return raw
    return raw * 1024

def _system_wired_bytes() -> int:
    if platform.system() != "Darwin":
        return 0
    try:
        out = subprocess.check_output(
            ["vm_stat"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return 0
    page_size = 4096
    wired_pages = 0
    for line in out.splitlines():
        if "page size of" in line:
            parts = [part for part in line.split() if part.isdigit()]
            if parts:
                page_size = int(parts[0])
        if line.startswith("Pages wired down:"):
            raw = line.split(":", 1)[1].strip().rstrip(".").replace(",", "")
            try:
                wired_pages = int(raw)
            except ValueError:
                wired_pages = 0
    return int(wired_pages * page_size)

def _mlx_memory_bytes(name: str) -> int:
    fn = getattr(mx, name, None)
    if fn is None:
        return 0
    try:
        return int(fn())
    except Exception:
        return 0

def _to_gb(value: Any) -> float:
    try:
        return float(value) / _GB
    except (TypeError, ValueError):
        return 0.0

def _is_array_like(value: Any) -> bool:
    return hasattr(value, "nbytes") and hasattr(value, "shape")

def _iter_cache_entries(cache: Any) -> list[Any]:
    if cache is None:
        return []
    if isinstance(cache, (list, tuple)):
        return list(cache)
    return [cache]

def _looks_recurrent(cache: Any) -> bool:
    name = type(cache).__name__.lower()
    return "recurrent" in name or hasattr(cache, "_tape") or hasattr(cache, "_tape_k")

def _kv_cache_nbytes(cache: Any) -> int:
    if cache is None:
        return 0
    total = 0
    if hasattr(cache, "keys") or hasattr(cache, "values"):
        total += tree_nbytes([getattr(cache, "keys", None), getattr(cache, "values", None)])
    state = None
    try:
        state = getattr(cache, "state")
    except Exception:
        state = None
    if state is not None:
        total += tree_nbytes(state)
    return int(total)

def _prefix_entries(prefix_cache: Any) -> list[Any]:
    lock = getattr(prefix_cache, "_lock", None)
    entries = getattr(prefix_cache, "_entries", None)
    if not isinstance(entries, dict):
        return []
    if lock is None:
        return list(entries.values())
    try:
        with lock:
            return list(entries.values())
    except Exception:
        return []

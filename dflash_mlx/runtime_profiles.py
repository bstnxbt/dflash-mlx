# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Optional

GiB = 1024 * 1024 * 1024
RUNTIME_PROFILE_ENV = "DFLASH_RUNTIME_PROFILE"

@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    prefill_step_size: int
    draft_sink_size: int
    draft_window_size: int
    verify_len_cap: int
    prefix_cache: bool
    prefix_cache_max_entries: int
    prefix_cache_max_bytes: int
    clear_cache_boundaries: bool
    max_snapshot_tokens: int
    prefix_cache_l2: bool
    prefix_cache_l2_max_bytes: int
    verify_mode: str

@dataclass(frozen=True)
class EffectiveRuntimeConfig:
    profile: str
    prefill_step_size: int
    draft_sink_size: int
    draft_window_size: int
    verify_len_cap: int
    prefix_cache: bool
    prefix_cache_max_entries: int
    prefix_cache_max_bytes: int
    clear_cache_boundaries: bool
    max_snapshot_tokens: int
    prefix_cache_l2: bool
    prefix_cache_l2_dir: str
    prefix_cache_l2_max_bytes: int
    target_fa_window: int
    dflash_max_ctx: int
    memory_waterfall: bool
    bench_log_dir: str
    verify_mode: str

PROFILES: dict[str, RuntimeProfile] = {
    "balanced": RuntimeProfile(
        name="balanced",
        prefill_step_size=4096,
        draft_sink_size=64,
        draft_window_size=1024,
        verify_len_cap=0,
        prefix_cache=True,
        prefix_cache_max_entries=4,
        prefix_cache_max_bytes=8 * GiB,
        clear_cache_boundaries=False,
        max_snapshot_tokens=24000,
        prefix_cache_l2=False,
        prefix_cache_l2_max_bytes=50 * GiB,
        verify_mode="auto",
    ),
    "fast": RuntimeProfile(
        name="fast",
        prefill_step_size=8192,
        draft_sink_size=64,
        draft_window_size=1024,
        verify_len_cap=0,
        prefix_cache=True,
        prefix_cache_max_entries=4,
        prefix_cache_max_bytes=16 * GiB,
        clear_cache_boundaries=False,
        max_snapshot_tokens=0,
        prefix_cache_l2=False,
        prefix_cache_l2_max_bytes=50 * GiB,
        verify_mode="auto",
    ),
    "low-memory": RuntimeProfile(
        name="low-memory",
        prefill_step_size=1024,
        draft_sink_size=64,
        draft_window_size=1024,
        verify_len_cap=0,
        prefix_cache=True,
        prefix_cache_max_entries=2,
        prefix_cache_max_bytes=2 * GiB,
        clear_cache_boundaries=False,
        max_snapshot_tokens=8000,
        prefix_cache_l2=False,
        prefix_cache_l2_max_bytes=50 * GiB,
        verify_mode="auto",
    ),
    "long-session": RuntimeProfile(
        name="long-session",
        prefill_step_size=4096,
        draft_sink_size=64,
        draft_window_size=1024,
        verify_len_cap=0,
        prefix_cache=True,
        prefix_cache_max_entries=8,
        prefix_cache_max_bytes=8 * GiB,
        clear_cache_boundaries=False,
        max_snapshot_tokens=32000,
        prefix_cache_l2=True,
        prefix_cache_l2_max_bytes=50 * GiB,
        verify_mode="auto",
    ),
}

def profile_names() -> tuple[str, ...]:
    return tuple(PROFILES)

def format_profiles() -> str:
    rows = [
        "profile      prefill  draft_window  verify_cap  prefix  L1           clear     max_snapshot  L2          verify  note"
    ]
    notes = {
        "balanced": "recommended default for normal agentic coding",
        "fast": "more throughput, higher memory",
        "low-memory": "lower memory pressure, slower prefill",
        "long-session": "prefix/L2 oriented for multi-turn revisits",
    }
    for profile in PROFILES.values():
        rows.append(
            " ".join(
                [
                    f"{profile.name:<12}",
                    f"{profile.prefill_step_size:<8}",
                    f"{profile.draft_sink_size}+{profile.draft_window_size:<8}",
                    f"{_verify_cap(profile.verify_len_cap):<10}",
                    f"{_on_off(profile.prefix_cache):<7}",
                    (
                        f"{profile.prefix_cache_max_entries}x"
                        f"{_format_gib(profile.prefix_cache_max_bytes):<10}"
                    ),
                    f"{_clear_policy(profile.clear_cache_boundaries):<9}",
                    f"{profile.max_snapshot_tokens:<13}",
                    f"{_l2_summary(profile):<11}",
                    f"{profile.verify_mode:<7}",
                    notes[profile.name],
                ]
            )
        )
    return "\n".join(rows)

def _on_off(value: bool) -> str:
    return "on" if value else "off"

def _clear_policy(enabled: bool) -> str:
    return "boundary" if enabled else "off"

def _format_gib(value: int) -> str:
    if value % GiB == 0:
        return f"{value // GiB}GiB"
    return f"{value / GiB:.1f}GiB"

def _l2_summary(profile: RuntimeProfile) -> str:
    if not profile.prefix_cache_l2:
        return "off"
    return f"on/{_format_gib(profile.prefix_cache_l2_max_bytes)}"

def _verify_cap(value: int) -> str:
    return "block" if int(value) == 0 else str(int(value))

def resolve_runtime_config(args: Any) -> EffectiveRuntimeConfig:
    profile_name = _resolve_profile_name(getattr(args, "profile", None))
    profile = PROFILES[profile_name]
    cfg = EffectiveRuntimeConfig(
        profile=profile_name,
        prefill_step_size=_resolve_int(
            getattr(args, "prefill_step_size", None),
            "DFLASH_PREFILL_STEP_SIZE",
            profile.prefill_step_size,
        ),
        draft_sink_size=_resolve_int(
            getattr(args, "draft_sink_size", None),
            "DFLASH_DRAFT_SINK_SIZE",
            profile.draft_sink_size,
        ),
        draft_window_size=_resolve_int(
            getattr(args, "draft_window_size", None),
            "DFLASH_DRAFT_WINDOW_SIZE",
            profile.draft_window_size,
        ),
        verify_len_cap=_resolve_int(
            getattr(args, "verify_len_cap", None),
            "DFLASH_VERIFY_LEN_CAP",
            profile.verify_len_cap,
        ),
        prefix_cache=_resolve_bool(
            getattr(args, "prefix_cache", None),
            "DFLASH_PREFIX_CACHE",
            profile.prefix_cache,
        ),
        prefix_cache_max_entries=_resolve_int(
            getattr(args, "prefix_cache_max_entries", None),
            "DFLASH_PREFIX_CACHE_MAX_ENTRIES",
            profile.prefix_cache_max_entries,
        ),
        prefix_cache_max_bytes=_resolve_int(
            getattr(args, "prefix_cache_max_bytes", None),
            "DFLASH_PREFIX_CACHE_MAX_BYTES",
            profile.prefix_cache_max_bytes,
        ),
        clear_cache_boundaries=_resolve_bool(
            getattr(args, "clear_cache_boundaries", None),
            "DFLASH_CLEAR_CACHE_BOUNDARIES",
            profile.clear_cache_boundaries,
        ),
        max_snapshot_tokens=_resolve_int(
            getattr(args, "max_snapshot_tokens", None),
            "DFLASH_MAX_SNAPSHOT_TOKENS",
            profile.max_snapshot_tokens,
        ),
        prefix_cache_l2=_resolve_bool(
            getattr(args, "prefix_cache_l2", None),
            "DFLASH_PREFIX_CACHE_L2_ENABLED",
            profile.prefix_cache_l2,
        ),
        prefix_cache_l2_dir=_resolve_str(
            getattr(args, "prefix_cache_l2_dir", None),
            "DFLASH_PREFIX_CACHE_L2_DIR",
            os.path.expanduser("~/.cache/dflash/prefix_l2"),
        ),
        prefix_cache_l2_max_bytes=_resolve_int(
            getattr(args, "prefix_cache_l2_max_bytes", None),
            "DFLASH_PREFIX_CACHE_L2_MAX_BYTES",
            profile.prefix_cache_l2_max_bytes,
        ),
        target_fa_window=_resolve_int(
            getattr(args, "target_fa_window", None),
            "DFLASH_TARGET_FA_WINDOW",
            0,
        ),
        dflash_max_ctx=_resolve_int(
            getattr(args, "dflash_max_ctx", None),
            "DFLASH_MAX_CTX",
            0,
        ),
        memory_waterfall=bool(getattr(args, "memory_waterfall", None) or False),
        bench_log_dir=str(getattr(args, "bench_log_dir", None) or ""),
        verify_mode=_resolve_verify_mode(getattr(args, "verify_mode", None), profile.verify_mode),
    )
    return validate_runtime_config(cfg)

def validate_runtime_config(cfg: EffectiveRuntimeConfig) -> EffectiveRuntimeConfig:
    if cfg.prefill_step_size <= 0:
        raise ValueError("--prefill-step-size / prefill_step_size must be > 0")
    if cfg.draft_sink_size < 0:
        raise ValueError("--draft-sink-size / draft_sink_size must be >= 0")
    if cfg.draft_window_size <= 0:
        raise ValueError("--draft-window-size / draft_window_size must be > 0")
    if cfg.verify_len_cap < 0:
        raise ValueError("--verify-len-cap / verify_len_cap must be >= 0")
    if cfg.prefix_cache_max_entries <= 0:
        raise ValueError(
            "--prefix-cache-max-entries / prefix_cache_max_entries must be > 0"
        )
    if cfg.prefix_cache_max_bytes < 0:
        raise ValueError(
            "--prefix-cache-max-bytes / prefix_cache_max_bytes must be >= 0"
        )
    if cfg.max_snapshot_tokens < 0:
        raise ValueError("--max-snapshot-tokens / max_snapshot_tokens must be >= 0")
    if cfg.prefix_cache_l2 and not cfg.prefix_cache_l2_dir.strip():
        raise ValueError(
            "--prefix-cache-l2-dir / prefix_cache_l2_dir must not be empty when L2 is enabled"
        )
    if cfg.prefix_cache_l2 and cfg.prefix_cache_l2_max_bytes < 0:
        raise ValueError(
            "--prefix-cache-l2-max-bytes / prefix_cache_l2_max_bytes must be >= 0"
        )
    if cfg.target_fa_window < 0:
        raise ValueError("--target-fa-window / target_fa_window must be >= 0")
    if cfg.dflash_max_ctx < 0:
        raise ValueError("--dflash-max-ctx / dflash_max_ctx must be >= 0")
    if cfg.verify_mode not in ("auto", "off"):
        raise ValueError("--verify-mode / verify_mode must be auto or off")
    if cfg.bench_log_dir != "" and not cfg.bench_log_dir.strip():
        raise ValueError("--bench-log-dir / bench_log_dir must not be empty")
    if not cfg.prefix_cache and cfg.prefix_cache_l2:
        return replace(cfg, prefix_cache_l2=False)
    if cfg.target_fa_window > 0 and cfg.prefix_cache:
        return replace(cfg, prefix_cache=False, prefix_cache_l2=False)
    if cfg.target_fa_window > 0 and cfg.prefix_cache_l2:
        return replace(cfg, prefix_cache_l2=False)
    return cfg

def _resolve_profile_name(cli_value: Optional[str]) -> str:
    raw = cli_value or os.environ.get(RUNTIME_PROFILE_ENV, "").strip() or "balanced"
    if raw not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILES)}")
    return raw

def _resolve_int(cli_value: Optional[int], env_key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return int(raw)
    return int(default)

def _resolve_bool(cli_value: Optional[bool], env_key: str, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    raw = os.environ.get(env_key, "").strip().lower()
    if raw:
        return raw not in ("0", "false", "no", "off")
    return bool(default)

def _resolve_str(cli_value: Optional[str], env_key: str, default: str) -> str:
    if cli_value is not None:
        return str(cli_value)
    raw = os.environ.get(env_key)
    if raw is not None and raw.strip():
        return str(raw)
    return str(default)

def _resolve_verify_mode(cli_value: Optional[str], default: str) -> str:
    if cli_value is not None:
        return str(cli_value)
    raw = os.environ.get("DFLASH_VERIFY_MODE", "").strip()
    if raw:
        return raw
    return default

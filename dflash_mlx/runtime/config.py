# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, replace
from typing import Any

GiB = 1024 * 1024 * 1024
DEFAULT_PREFIX_CACHE_L2_DIR = os.path.expanduser("~/.cache/dflash/prefix_l2")
SURFACE_SERVE_DOCTOR = "serve_doctor"
SURFACE_GENERATE = "generate"
SURFACE_BENCHMARK = "benchmark"

_BYTE_COUNT_RE = re.compile(
    r"(-?[0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b?|bytes?)?",
    re.IGNORECASE,
)


def _parse_byte_count(raw: Any) -> int:
    value = str(raw).strip()
    match = _BYTE_COUNT_RE.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("expected raw bytes or a byte value like 2GB")
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
    try:
        multiplier = multipliers[unit]
    except KeyError as exc:
        raise argparse.ArgumentTypeError("unknown byte suffix") from exc
    return int(number * multiplier)


@dataclass(frozen=True)
class EffectiveRuntimeConfig:
    prefill_step_size: int
    draft_sink_size: int
    draft_window_size: int
    draft_full_context_min_ctx: int
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
    verify_mode: str
    copyspec_mode: str
    quantize_kv_cache: bool
    prefix_cache_l2_frontier_stride: int

DEFAULT_RUNTIME_CONFIG = EffectiveRuntimeConfig(
    prefill_step_size=2048,
    draft_sink_size=64,
    draft_window_size=1024,
    draft_full_context_min_ctx=16384,
    verify_len_cap=0,
    prefix_cache=True,
    prefix_cache_max_entries=8,
    prefix_cache_max_bytes=8 * GiB,
    clear_cache_boundaries=True,
    max_snapshot_tokens=32000,
    prefix_cache_l2=True,
    prefix_cache_l2_dir=DEFAULT_PREFIX_CACHE_L2_DIR,
    prefix_cache_l2_max_bytes=50 * GiB,
    target_fa_window=0,
    dflash_max_ctx=0,
    verify_mode="adaptive",
    copyspec_mode="conservative",
    quantize_kv_cache=False,
    prefix_cache_l2_frontier_stride=0,
)

@dataclass(frozen=True)
class RuntimeConfigFieldSpec:
    field: str
    flags: tuple[str, ...]
    env: str | None
    help: str
    value_type: Any | None = None
    action: Any | None = None
    choices: tuple[str, ...] = ()
    metavar: str | None = None
    surfaces: tuple[str, ...] = (SURFACE_SERVE_DOCTOR,)


@dataclass(frozen=True)
class RuntimeConfigSpec:
    fields: tuple[RuntimeConfigFieldSpec, ...]

    def __post_init__(self) -> None:
        names = [field.field for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("runtime config field names must be unique")

    @property
    def by_field(self) -> dict[str, RuntimeConfigFieldSpec]:
        return {field.field: field for field in self.fields}

    def require_field(self, field: str) -> RuntimeConfigFieldSpec:
        try:
            return self.by_field[field]
        except KeyError as exc:
            raise KeyError(f"unknown runtime config field: {field}") from exc

    def full_field_names(self) -> tuple[str, ...]:
        return tuple(field.field for field in self.fields)

    def surface_field_names(
        self,
        surface: str,
        *,
        order: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        return self._ordered_field_names(
            tuple(field.field for field in self.fields if surface in field.surfaces),
            order=order,
            label=f"runtime surface {surface}",
        )

    @staticmethod
    def _ordered_field_names(
        fields: tuple[str, ...],
        *,
        order: tuple[str, ...] | None,
        label: str,
    ) -> tuple[str, ...]:
        if order is None:
            return fields
        field_set = set(fields)
        order_set = set(order)
        if field_set != order_set:
            missing = sorted(field_set - order_set)
            extra = sorted(order_set - field_set)
            raise ValueError(f"{label} order mismatch: missing={missing}, extra={extra}")
        return order

RUNTIME_CONFIG_FIELDS: tuple[RuntimeConfigFieldSpec, ...] = (
    RuntimeConfigFieldSpec(
        field="prefill_step_size",
        flags=("--prefill-step-size",),
        env="DFLASH_PREFILL_STEP_SIZE",
        value_type=int,
        help="Prompt prefill chunk size.",
        metavar="INT",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="draft_sink_size",
        flags=("--draft-sink-size",),
        env="DFLASH_DRAFT_SINK_SIZE",
        value_type=int,
        help="Draft context cache sink tokens kept before the rolling window.",
        metavar="INT",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="draft_window_size",
        flags=("--draft-window-size",),
        env="DFLASH_DRAFT_WINDOW_SIZE",
        value_type=int,
        help="Draft context cache rolling window tokens.",
        metavar="INT",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="draft_full_context_min_ctx",
        flags=("--draft-full-context-min-ctx",),
        env="DFLASH_DRAFT_FULL_CONTEXT_MIN_CTX",
        value_type=int,
        help=(
            "Projected context (prompt + max new tokens) at or above which the "
            "draft full-attention layer attends all context features instead of "
            "the sink+window slice. 0 = always full-context. Short contexts pay "
            "a measured pure cost with no acceptance gain; deep contexts gain "
            "from matching the draft's full-attention training condition."
        ),
        metavar="INT",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="verify_len_cap",
        flags=("--verify-len-cap",),
        env="DFLASH_VERIFY_LEN_CAP",
        value_type=int,
        help="Max tokens verified per target forward.",
        metavar="INT",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="clear_cache_boundaries",
        flags=("--clear-cache-boundaries",),
        env="DFLASH_CLEAR_CACHE_BOUNDARIES",
        action=argparse.BooleanOptionalAction,
        help="Clear the MLX cache at safe request boundaries.",
    ),
    RuntimeConfigFieldSpec(
        field="verify_mode",
        flags=("--verify-mode",),
        env="DFLASH_VERIFY_MODE",
        choices=("dflash", "adaptive", "ddtree", "off"),
        help="Verify path mode. Default adaptive shortens low-acceptance blocks; use off only for debug/parity.",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="copyspec_mode",
        flags=("--copyspec-mode",),
        env="DFLASH_COPYSPEC_MODE",
        choices=("conservative", "off", "auto"),
        help=(
            "Prompt-lookup (copyspec) drafting. Default 'conservative' disables copyspec after one missed "
            "copy block. 'auto' self-gates: periodically A/B-probes copyspec on vs off and latches to whichever "
            "is faster, resetting adaptive state on disengage (a clear win on strong-copyspec targets like large "
            "MoE; self-disables where copyspec doesn't pay off). 'off' disables copyspec entirely."
        ),
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="max_snapshot_tokens",
        flags=("--max-snapshot-tokens",),
        env="DFLASH_MAX_SNAPSHOT_TOKENS",
        value_type=int,
        help="Skip prefix-cache snapshot inserts above this token count; 0 disables the cap.",
        metavar="INT",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_l2",
        flags=("--prefix-cache-l2",),
        env="DFLASH_PREFIX_CACHE_L2_ENABLED",
        action=argparse.BooleanOptionalAction,
        help="Enable SSD L2 for persistent and spilled prefix snapshots.",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_l2_dir",
        flags=("--prefix-cache-l2-dir",),
        env="DFLASH_PREFIX_CACHE_L2_DIR",
        value_type=str,
        help="Directory for prefix-cache L2 files.",
        metavar="PATH",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_l2_max_bytes",
        flags=("--prefix-cache-l2-max-bytes",),
        env="DFLASH_PREFIX_CACHE_L2_MAX_BYTES",
        value_type=_parse_byte_count,
        help=(
            "Byte budget for prefix-cache L2. "
            "Accepts raw bytes or suffixes like 50GB."
        ),
        metavar="BYTES",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_l2_frontier_stride",
        flags=("--prefix-cache-l2-frontier-stride",),
        env="DFLASH_PREFIX_CACHE_L2_FRONTIER_STRIDE",
        value_type=int,
        help=(
            "Token stride at which L2 frontier snapshots are written during cold "
            "prefill. 0 = auto (default: 8192, rounded up to a prefill-step "
            "multiple). Values below 8192 are raised to the disk-pressure floor."
        ),
        metavar="INT",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache",
        flags=("--prefix-cache",),
        env="DFLASH_PREFIX_CACHE",
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable the DFlash prefix cache that reuses cross-turn KV state. "
            "Default: enabled. Big win on multi-turn agentic workloads, "
            "~neutral on single-turn."
        ),
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_max_entries",
        flags=("--prefix-cache-max-entries",),
        env="DFLASH_PREFIX_CACHE_MAX_ENTRIES",
        value_type=int,
        help="Maximum number of cached prefix snapshots.",
        metavar="INT",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_max_bytes",
        flags=("--prefix-cache-max-bytes",),
        env="DFLASH_PREFIX_CACHE_MAX_BYTES",
        value_type=_parse_byte_count,
        help=(
            "Maximum total bytes the prefix cache may hold. "
            "Accepts raw bytes or suffixes like 2GB."
        ),
        metavar="BYTES",
    ),
    RuntimeConfigFieldSpec(
        field="target_fa_window",
        flags=("--target-fa-window",),
        env="DFLASH_TARGET_FA_WINDOW",
        value_type=int,
        help=(
            "Experimental target verifier full-attention KV window. "
            "N>0 uses a rotating KV cache of N tokens for target full-attention layers only."
        ),
        metavar="INT",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="quantize_kv_cache",
        flags=("--quantize-kv-cache",),
        env="DFLASH_QUANTIZE_KV_CACHE",
        action=argparse.BooleanOptionalAction,
        help=(
            "Quantize the target full-attention KV cache to 8-bit "
            "(group size 64). Halves KV memory growth at long context; "
            "linear (gated-delta) layer state is unaffected. Trade-off: "
            "decode slows as context grows (no fused quantized-KV "
            "attention kernel in MLX yet) — use when memory is the "
            "binding constraint. Default: disabled. Incompatible with "
            "--target-fa-window."
        ),
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="dflash_max_ctx",
        flags=("--dflash-max-ctx",),
        env="DFLASH_MAX_CTX",
        value_type=int,
        help="Hard cap on runtime context length.",
        metavar="INT",
    ),
)

RUNTIME_CONFIG_SPEC = RuntimeConfigSpec(RUNTIME_CONFIG_FIELDS)

def _add_runtime_config_argument(
    parser: argparse.ArgumentParser,
    option: RuntimeConfigFieldSpec,
    *,
    default: Any,
    help_text: str | None = None,
) -> None:
    kwargs: dict[str, Any] = {"default": default, "help": help_text or option.help}
    if option.value_type is not None:
        kwargs["type"] = option.value_type
    if option.action is not None:
        kwargs["action"] = option.action
    if option.choices:
        kwargs["choices"] = option.choices
    if option.metavar is not None:
        kwargs["metavar"] = option.metavar
    parser.add_argument(*option.flags, **kwargs)


def add_runtime_config_field_arguments(
    parser: argparse.ArgumentParser,
    fields: tuple[str, ...],
    *,
    defaults: dict[str, Any] | None = None,
    help_suffixes: dict[str, str] | None = None,
) -> None:
    defaults = defaults or {}
    help_suffixes = help_suffixes or {}
    for field in fields:
        option = RUNTIME_CONFIG_SPEC.require_field(field)
        help_text = option.help
        suffix = help_suffixes.get(field)
        if suffix:
            help_text = f"{help_text} {suffix}"
        _add_runtime_config_argument(
            parser,
            option,
            default=defaults.get(field),
            help_text=help_text,
        )


def add_runtime_config_arguments(parser: argparse.ArgumentParser) -> None:
    add_runtime_config_field_arguments(
        parser,
        RUNTIME_CONFIG_SPEC.full_field_names(),
    )


def runtime_config_field_defaults(
    fields: tuple[str, ...],
) -> dict[str, Any]:
    return {field: getattr(DEFAULT_RUNTIME_CONFIG, field) for field in fields}


_GENERATE_RUNTIME_FIELD_ORDER = (
    "verify_mode",
    "copyspec_mode",
    "prefill_step_size",
    "target_fa_window",
    "quantize_kv_cache",
    "draft_sink_size",
    "draft_window_size",
    "draft_full_context_min_ctx",
    "verify_len_cap",
)

_BENCHMARK_RUNTIME_FIELD_ORDER = (
    "verify_mode",
    "copyspec_mode",
    "prefill_step_size",
    "target_fa_window",
    "quantize_kv_cache",
    "draft_sink_size",
    "draft_window_size",
    "draft_full_context_min_ctx",
    "verify_len_cap",
)

GENERATE_RUNTIME_FIELDS = RUNTIME_CONFIG_SPEC.surface_field_names(
    SURFACE_GENERATE,
    order=_GENERATE_RUNTIME_FIELD_ORDER,
)
BENCHMARK_RUNTIME_FIELDS = RUNTIME_CONFIG_SPEC.surface_field_names(
    SURFACE_BENCHMARK,
    order=_BENCHMARK_RUNTIME_FIELD_ORDER,
)


def runtime_config_from_defaults(
    *,
    prefill_step_size: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    draft_full_context_min_ctx: int | None = None,
    verify_len_cap: int | None = None,
    prefix_cache: bool | None = None,
    prefix_cache_max_entries: int | None = None,
    prefix_cache_max_bytes: int | None = None,
    clear_cache_boundaries: bool | None = None,
    max_snapshot_tokens: int | None = None,
    prefix_cache_l2: bool | None = None,
    prefix_cache_l2_dir: str | None = None,
    prefix_cache_l2_max_bytes: int | None = None,
    target_fa_window: int = 0,
    dflash_max_ctx: int = 0,
    verify_mode: str | None = None,
    copyspec_mode: str | None = None,
    quantize_kv_cache: bool | None = None,
    prefix_cache_l2_frontier_stride: int | None = None,
) -> EffectiveRuntimeConfig:
    defaults = DEFAULT_RUNTIME_CONFIG
    return validate_runtime_config(
        EffectiveRuntimeConfig(
            prefill_step_size=(
                defaults.prefill_step_size
                if prefill_step_size is None
                else int(prefill_step_size)
            ),
            draft_sink_size=(
                defaults.draft_sink_size
                if draft_sink_size is None
                else int(draft_sink_size)
            ),
            draft_window_size=(
                defaults.draft_window_size
                if draft_window_size is None
                else int(draft_window_size)
            ),
            draft_full_context_min_ctx=(
                defaults.draft_full_context_min_ctx
                if draft_full_context_min_ctx is None
                else int(draft_full_context_min_ctx)
            ),
            verify_len_cap=(
                defaults.verify_len_cap
                if verify_len_cap is None
                else int(verify_len_cap)
            ),
            prefix_cache=(
                defaults.prefix_cache if prefix_cache is None else bool(prefix_cache)
            ),
            prefix_cache_max_entries=(
                defaults.prefix_cache_max_entries
                if prefix_cache_max_entries is None
                else int(prefix_cache_max_entries)
            ),
            prefix_cache_max_bytes=(
                defaults.prefix_cache_max_bytes
                if prefix_cache_max_bytes is None
                else int(prefix_cache_max_bytes)
            ),
            clear_cache_boundaries=(
                defaults.clear_cache_boundaries
                if clear_cache_boundaries is None
                else bool(clear_cache_boundaries)
            ),
            max_snapshot_tokens=(
                defaults.max_snapshot_tokens
                if max_snapshot_tokens is None
                else int(max_snapshot_tokens)
            ),
            prefix_cache_l2=(
                defaults.prefix_cache_l2 if prefix_cache_l2 is None else bool(prefix_cache_l2)
            ),
            prefix_cache_l2_dir=(
                defaults.prefix_cache_l2_dir if prefix_cache_l2_dir is None else str(prefix_cache_l2_dir)
            ),
            prefix_cache_l2_max_bytes=(
                defaults.prefix_cache_l2_max_bytes
                if prefix_cache_l2_max_bytes is None
                else int(prefix_cache_l2_max_bytes)
            ),
            target_fa_window=int(target_fa_window),
            dflash_max_ctx=int(dflash_max_ctx),
            verify_mode=defaults.verify_mode if verify_mode is None else str(verify_mode),
            copyspec_mode=defaults.copyspec_mode if copyspec_mode is None else str(copyspec_mode),
            quantize_kv_cache=(
                defaults.quantize_kv_cache
                if quantize_kv_cache is None
                else bool(quantize_kv_cache)
            ),
            prefix_cache_l2_frontier_stride=(
                defaults.prefix_cache_l2_frontier_stride
                if prefix_cache_l2_frontier_stride is None
                else int(prefix_cache_l2_frontier_stride)
            ),
        )
    )

def resolve_runtime_config(args: Any) -> EffectiveRuntimeConfig:
    defaults = DEFAULT_RUNTIME_CONFIG
    return runtime_config_from_defaults(
        prefill_step_size=_resolve_int(
            getattr(args, "prefill_step_size", None),
            _runtime_env("prefill_step_size"),
            defaults.prefill_step_size,
        ),
        draft_sink_size=_resolve_int(
            getattr(args, "draft_sink_size", None),
            _runtime_env("draft_sink_size"),
            defaults.draft_sink_size,
        ),
        draft_window_size=_resolve_int(
            getattr(args, "draft_window_size", None),
            _runtime_env("draft_window_size"),
            defaults.draft_window_size,
        ),
        draft_full_context_min_ctx=_resolve_int(
            getattr(args, "draft_full_context_min_ctx", None),
            _runtime_env("draft_full_context_min_ctx"),
            defaults.draft_full_context_min_ctx,
        ),
        verify_len_cap=_resolve_int(
            getattr(args, "verify_len_cap", None),
            _runtime_env("verify_len_cap"),
            defaults.verify_len_cap,
        ),
        quantize_kv_cache=_resolve_bool(
            getattr(args, "quantize_kv_cache", None),
            _runtime_env("quantize_kv_cache"),
            defaults.quantize_kv_cache,
        ),
        prefix_cache=_resolve_bool(
            getattr(args, "prefix_cache", None),
            _runtime_env("prefix_cache"),
            defaults.prefix_cache,
        ),
        prefix_cache_max_entries=_resolve_int(
            getattr(args, "prefix_cache_max_entries", None),
            _runtime_env("prefix_cache_max_entries"),
            defaults.prefix_cache_max_entries,
        ),
        prefix_cache_max_bytes=_resolve_byte_count(
            getattr(args, "prefix_cache_max_bytes", None),
            _runtime_env("prefix_cache_max_bytes"),
            defaults.prefix_cache_max_bytes,
        ),
        clear_cache_boundaries=_resolve_bool(
            getattr(args, "clear_cache_boundaries", None),
            _runtime_env("clear_cache_boundaries"),
            defaults.clear_cache_boundaries,
        ),
        max_snapshot_tokens=_resolve_int(
            getattr(args, "max_snapshot_tokens", None),
            _runtime_env("max_snapshot_tokens"),
            defaults.max_snapshot_tokens,
        ),
        prefix_cache_l2=_resolve_bool(
            getattr(args, "prefix_cache_l2", None),
            _runtime_env("prefix_cache_l2"),
            defaults.prefix_cache_l2,
        ),
        prefix_cache_l2_dir=_resolve_str(
            getattr(args, "prefix_cache_l2_dir", None),
            _runtime_env("prefix_cache_l2_dir"),
            defaults.prefix_cache_l2_dir,
        ),
        prefix_cache_l2_max_bytes=_resolve_byte_count(
            getattr(args, "prefix_cache_l2_max_bytes", None),
            _runtime_env("prefix_cache_l2_max_bytes"),
            defaults.prefix_cache_l2_max_bytes,
        ),
        target_fa_window=_resolve_int(
            getattr(args, "target_fa_window", None),
            _runtime_env("target_fa_window"),
            0,
        ),
        dflash_max_ctx=_resolve_int(
            getattr(args, "dflash_max_ctx", None),
            _runtime_env("dflash_max_ctx"),
            0,
        ),
        verify_mode=_resolve_verify_mode(getattr(args, "verify_mode", None), defaults.verify_mode),
        copyspec_mode=_resolve_copyspec_mode(getattr(args, "copyspec_mode", None), defaults.copyspec_mode),
        prefix_cache_l2_frontier_stride=_resolve_int(
            getattr(args, "prefix_cache_l2_frontier_stride", None),
            _runtime_env("prefix_cache_l2_frontier_stride"),
            defaults.prefix_cache_l2_frontier_stride,
        ),
    )

def runtime_config_sources(args: Any, cfg: EffectiveRuntimeConfig) -> dict[str, str]:
    return {
        field: _source_for_runtime_field(args, field)
        for field in cfg.__dataclass_fields__
    }

def validate_runtime_config(cfg: EffectiveRuntimeConfig) -> EffectiveRuntimeConfig:
    if cfg.prefill_step_size <= 0:
        raise ValueError("--prefill-step-size / prefill_step_size must be > 0")
    if cfg.draft_sink_size < 0:
        raise ValueError("--draft-sink-size / draft_sink_size must be >= 0")
    if cfg.draft_window_size <= 0:
        raise ValueError("--draft-window-size / draft_window_size must be > 0")
    if cfg.draft_full_context_min_ctx < 0:
        raise ValueError(
            "--draft-full-context-min-ctx / draft_full_context_min_ctx must be >= 0"
        )
    if cfg.verify_len_cap < 0:
        raise ValueError("--verify-len-cap / verify_len_cap must be >= 0")
    if cfg.quantize_kv_cache and cfg.target_fa_window > 0:
        raise ValueError(
            "--quantize-kv-cache / quantize_kv_cache cannot be combined with "
            "--target-fa-window / target_fa_window (rotating window caches "
            "do not support quantized KV)"
        )
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
    if cfg.prefix_cache_l2_frontier_stride < 0:
        raise ValueError(
            "--prefix-cache-l2-frontier-stride / "
            "prefix_cache_l2_frontier_stride must be >= 0"
        )
    if cfg.verify_mode not in ("dflash", "adaptive", "ddtree", "off"):
        raise ValueError(
            "--verify-mode / verify_mode must be dflash, adaptive, ddtree, or off"
        )
    if cfg.copyspec_mode not in ("conservative", "off", "auto"):
        raise ValueError(
            "--copyspec-mode / copyspec_mode must be conservative, off, or auto"
        )
    if not cfg.prefix_cache and cfg.prefix_cache_l2:
        return replace(cfg, prefix_cache_l2=False)
    if cfg.target_fa_window > 0 and cfg.prefix_cache:
        return replace(cfg, prefix_cache=False, prefix_cache_l2=False)
    if cfg.target_fa_window > 0 and cfg.prefix_cache_l2:
        return replace(cfg, prefix_cache_l2=False)
    return cfg

def _runtime_env(field: str) -> str:
    env_key = RUNTIME_CONFIG_SPEC.require_field(field).env
    if env_key is None:
        raise KeyError(f"runtime field {field!r} has no environment key")
    return env_key

def _resolve_int(cli_value: int | None, env_key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return int(raw)
    return int(default)

def _resolve_byte_count(cli_value: int | None, env_key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    raw = os.environ.get(env_key, "").strip()
    if raw:
        try:
            return _parse_byte_count(raw)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"{env_key} must be a byte count: {exc}") from exc
    return int(default)


def _resolve_bool(cli_value: bool | None, env_key: str, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    raw = os.environ.get(env_key, "").strip().lower()
    if raw:
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        raise ValueError(
            f"{env_key} must be a boolean: use 1/0, true/false, yes/no, or on/off"
        )
    return bool(default)

def _resolve_str(cli_value: str | None, env_key: str, default: str) -> str:
    if cli_value is not None:
        return str(cli_value)
    raw = os.environ.get(env_key)
    if raw is not None and raw.strip():
        return str(raw)
    return str(default)

def _resolve_verify_mode(cli_value: str | None, default: str) -> str:
    if cli_value is not None:
        return str(cli_value)
    raw = os.environ.get(_runtime_env("verify_mode"), "").strip()
    if raw:
        return raw
    return default

def _resolve_copyspec_mode(cli_value: str | None, default: str) -> str:
    if cli_value is not None:
        return str(cli_value)
    raw = os.environ.get(_runtime_env("copyspec_mode"), "").strip()
    if raw:
        return raw
    return default

def _source_for_runtime_field(args: Any, field: str) -> str:
    if getattr(args, field, None) is not None:
        return "cli"
    option = RUNTIME_CONFIG_SPEC.require_field(field)
    env_key = option.env
    if env_key is not None and os.environ.get(env_key, "").strip():
        return "env"
    return "default"


_OFFLINE_RUNTIME_FIELDS = tuple(dict.fromkeys((*GENERATE_RUNTIME_FIELDS, *BENCHMARK_RUNTIME_FIELDS)))
_OFFLINE_HELP_SUFFIXES = {
    "prefill_step_size": lambda value: f"Default: {value}.",
    "draft_sink_size": lambda value: f"Default: {value}.",
    "draft_window_size": lambda value: f"Default: {value}.",
    "verify_len_cap": lambda value: "Default: 0 = block size." if int(value) == 0 else f"Default: {value}.",
    "target_fa_window": lambda value: "Default: 0 = full KV." if int(value) == 0 else f"Default: {value}.",
}


def _offline_help_overrides(fields: tuple[str, ...], defaults: dict[str, Any]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for field in fields:
        suffix = _OFFLINE_HELP_SUFFIXES.get(field)
        if suffix is None:
            continue
        overrides[field] = suffix(defaults[field])
    return overrides


def add_offline_runtime_arguments(
    parser: argparse.ArgumentParser,
    fields: tuple[str, ...],
) -> None:
    defaults = runtime_config_field_defaults(fields)
    add_runtime_config_field_arguments(
        parser,
        fields,
        defaults=defaults,
        help_suffixes=_offline_help_overrides(fields, defaults),
    )


def offline_runtime_kwargs(args: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: getattr(args, field) for field in fields}


def offline_runtime_error_message(message: str) -> str:
    cleaned = str(message)
    for field in _OFFLINE_RUNTIME_FIELDS:
        cleaned = cleaned.replace(f" / {field}", "")
    return cleaned

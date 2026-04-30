# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from dflash_mlx.diagnostics import DiagnosticsConfig
from dflash_mlx.runtime_profiles import (
    EffectiveRuntimeConfig,
    PROFILES,
    validate_runtime_config,
)

if TYPE_CHECKING:
    from dflash_mlx.metal_limits import MetalLimitConfig
    from dflash_mlx.runtime import VerifyConfig

@dataclass(frozen=True)
class RuntimeContext:
    runtime: EffectiveRuntimeConfig
    diagnostics: DiagnosticsConfig
    verify: VerifyConfig
    metal_limits: MetalLimitConfig | None = None

def runtime_config_from_profile(
    profile: str = "balanced",
    *,
    prefill_step_size: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    verify_len_cap: int | None = None,
    prefix_cache: bool | None = None,
    prefix_cache_max_entries: int | None = None,
    prefix_cache_max_bytes: int | None = None,
    clear_cache_boundaries: bool | None = None,
    max_snapshot_tokens: int | None = None,
    prefix_cache_l2: bool | None = None,
    prefix_cache_l2_dir: str = "",
    prefix_cache_l2_max_bytes: int | None = None,
    target_fa_window: int = 0,
    dflash_max_ctx: int = 0,
    memory_waterfall: bool = False,
    bench_log_dir: str = "",
    verify_mode: str | None = None,
) -> EffectiveRuntimeConfig:
    runtime_profile = PROFILES[profile]
    return validate_runtime_config(
        EffectiveRuntimeConfig(
            profile=profile,
            prefill_step_size=(
                runtime_profile.prefill_step_size
                if prefill_step_size is None
                else int(prefill_step_size)
            ),
            draft_sink_size=(
                runtime_profile.draft_sink_size
                if draft_sink_size is None
                else int(draft_sink_size)
            ),
            draft_window_size=(
                runtime_profile.draft_window_size
                if draft_window_size is None
                else int(draft_window_size)
            ),
            verify_len_cap=(
                runtime_profile.verify_len_cap
                if verify_len_cap is None
                else int(verify_len_cap)
            ),
            prefix_cache=runtime_profile.prefix_cache if prefix_cache is None else bool(prefix_cache),
            prefix_cache_max_entries=(
                runtime_profile.prefix_cache_max_entries
                if prefix_cache_max_entries is None
                else int(prefix_cache_max_entries)
            ),
            prefix_cache_max_bytes=(
                runtime_profile.prefix_cache_max_bytes
                if prefix_cache_max_bytes is None
                else int(prefix_cache_max_bytes)
            ),
            clear_cache_boundaries=(
                runtime_profile.clear_cache_boundaries
                if clear_cache_boundaries is None
                else bool(clear_cache_boundaries)
            ),
            max_snapshot_tokens=(
                runtime_profile.max_snapshot_tokens
                if max_snapshot_tokens is None
                else int(max_snapshot_tokens)
            ),
            prefix_cache_l2=(
                runtime_profile.prefix_cache_l2 if prefix_cache_l2 is None else bool(prefix_cache_l2)
            ),
            prefix_cache_l2_dir=str(prefix_cache_l2_dir),
            prefix_cache_l2_max_bytes=(
                runtime_profile.prefix_cache_l2_max_bytes
                if prefix_cache_l2_max_bytes is None
                else int(prefix_cache_l2_max_bytes)
            ),
            target_fa_window=int(target_fa_window),
            dflash_max_ctx=int(dflash_max_ctx),
            memory_waterfall=bool(memory_waterfall),
            bench_log_dir=str(bench_log_dir),
            verify_mode=runtime_profile.verify_mode if verify_mode is None else str(verify_mode),
        )
    )

def build_runtime_context(
    runtime_config: EffectiveRuntimeConfig,
    diagnostics_config: DiagnosticsConfig | None = None,
    metal_limits: MetalLimitConfig | None = None,
) -> RuntimeContext:
    from dflash_mlx.runtime import VerifyConfig

    validated_runtime = validate_runtime_config(runtime_config)
    return RuntimeContext(
        runtime=validated_runtime,
        diagnostics=diagnostics_config or DiagnosticsConfig(),
        verify=VerifyConfig.from_mode(validated_runtime.verify_mode),
        metal_limits=metal_limits,
    )

def build_offline_runtime_context(
    *,
    target_fa_window: int,
    draft_sink_size: int = 64,
    draft_window_size: int = 1024,
    verify_len_cap: int = 0,
    verify_mode: str = "auto",
) -> RuntimeContext:
    runtime_config = runtime_config_from_profile(
        profile="balanced",
        prefix_cache=False,
        prefix_cache_l2=False,
        target_fa_window=int(target_fa_window),
        draft_sink_size=int(draft_sink_size),
        draft_window_size=int(draft_window_size),
        verify_len_cap=int(verify_len_cap),
        verify_mode=verify_mode,
    )
    return build_runtime_context(runtime_config)

def with_metal_limits(
    context: RuntimeContext,
    metal_limits: MetalLimitConfig | None,
) -> RuntimeContext:
    return replace(context, metal_limits=metal_limits)

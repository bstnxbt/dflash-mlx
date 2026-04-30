# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dflash_mlx.runtime_profiles import (
    EffectiveRuntimeConfig,
    format_profiles,
    profile_names,
    resolve_runtime_config,
)
from dflash_mlx.runtime_context import build_runtime_context

_VERIFY_INTERNAL_ENVS = (
    "DFLASH_VERIFY_LINEAR",
    "DFLASH_VERIFY_QMM",
    "DFLASH_VERIFY_VARIANT",
    "DFLASH_VERIFY_MAX_N",
    "DFLASH_VERIFY_QMM_KPARTS",
    "DFLASH_VERIFY_INCLUDE",
)

_CONFIG_FIELDS: dict[str, tuple[str | None, str | None]] = {
    "profile": ("DFLASH_RUNTIME_PROFILE", "balanced"),
    "prefill_step_size": ("DFLASH_PREFILL_STEP_SIZE", None),
    "draft_sink_size": ("DFLASH_DRAFT_SINK_SIZE", None),
    "draft_window_size": ("DFLASH_DRAFT_WINDOW_SIZE", None),
    "verify_len_cap": ("DFLASH_VERIFY_LEN_CAP", "0"),
    "prefix_cache": ("DFLASH_PREFIX_CACHE", None),
    "prefix_cache_max_entries": ("DFLASH_PREFIX_CACHE_MAX_ENTRIES", None),
    "prefix_cache_max_bytes": ("DFLASH_PREFIX_CACHE_MAX_BYTES", None),
    "clear_cache_boundaries": ("DFLASH_CLEAR_CACHE_BOUNDARIES", None),
    "max_snapshot_tokens": ("DFLASH_MAX_SNAPSHOT_TOKENS", None),
    "prefix_cache_l2": ("DFLASH_PREFIX_CACHE_L2_ENABLED", None),
    "prefix_cache_l2_dir": ("DFLASH_PREFIX_CACHE_L2_DIR", "default"),
    "prefix_cache_l2_max_bytes": ("DFLASH_PREFIX_CACHE_L2_MAX_BYTES", None),
    "target_fa_window": ("DFLASH_TARGET_FA_WINDOW", "0"),
    "dflash_max_ctx": ("DFLASH_MAX_CTX", "0"),
    "memory_waterfall": (None, "0"),
    "bench_log_dir": (None, ""),
    "verify_mode": ("DFLASH_VERIFY_MODE", None),
}

_CLI_FIELDS = {
    "profile": "profile",
    "prefill_step_size": "prefill_step_size",
    "draft_sink_size": "draft_sink_size",
    "draft_window_size": "draft_window_size",
    "verify_len_cap": "verify_len_cap",
    "prefix_cache": "prefix_cache",
    "prefix_cache_max_entries": "prefix_cache_max_entries",
    "prefix_cache_max_bytes": "prefix_cache_max_bytes",
    "clear_cache_boundaries": "clear_cache_boundaries",
    "max_snapshot_tokens": "max_snapshot_tokens",
    "prefix_cache_l2": "prefix_cache_l2",
    "prefix_cache_l2_dir": "prefix_cache_l2_dir",
    "prefix_cache_l2_max_bytes": "prefix_cache_l2_max_bytes",
    "target_fa_window": "target_fa_window",
    "dflash_max_ctx": "dflash_max_ctx",
    "memory_waterfall": "memory_waterfall",
    "bench_log_dir": "bench_log_dir",
    "verify_mode": "verify_mode",
}

@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any]

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Check the local DFlash runtime.")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--draft-model", "--draft", dest="draft_model", type=str, default=None)
    parser.add_argument("--profile", choices=profile_names(), default=None)
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--draft-sink-size", type=int, default=None)
    parser.add_argument("--draft-window-size", type=int, default=None)
    parser.add_argument("--verify-len-cap", type=int, default=None)
    parser.add_argument("--clear-cache-boundaries", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--memory-waterfall", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bench-log-dir", type=str, default=None)
    parser.add_argument("--verify-mode", choices=("auto", "off"), default=None)
    parser.add_argument("--max-snapshot-tokens", type=int, default=None)
    parser.add_argument("--prefix-cache-l2", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefix-cache-l2-dir", type=str, default=None)
    parser.add_argument("--prefix-cache-l2-max-bytes", type=int, default=None)
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefix-cache-max-entries", type=int, default=None)
    parser.add_argument("--prefix-cache-max-bytes", type=int, default=None)
    parser.add_argument("--target-fa-window", type=int, default=None)
    parser.add_argument("--dflash-max-ctx", type=int, default=None)
    return parser

def run(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(list(argv) if argv is not None else None)
    if args.list_profiles:
        print(format_profiles())
        return 0

    report = collect_report(args)
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text_report(report)

    fatals = int(report["summary"]["fatals"])
    warnings = int(report["summary"]["warnings"])
    if fatals:
        return 1
    if warnings:
        return 1 if args.strict else 2
    return 0

def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    raise SystemExit(run(argv, prog=prog))

def collect_report(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[DoctorCheck] = []
    cfg: EffectiveRuntimeConfig | None = None

    checks.append(_check_python())
    checks.append(_check_import("mlx"))
    checks.append(_check_import("mlx_lm"))
    checks.append(_check_metal())
    checks.append(_check_scripts_importable())

    try:
        cfg = resolve_runtime_config(args)
        checks.append(
            DoctorCheck(
                "profile_resolver",
                "ok",
                "runtime profile resolved",
                {"profile": cfg.profile},
            )
        )
    except Exception as exc:
        checks.append(
            DoctorCheck(
                "profile_resolver",
                "fatal",
                "runtime profile failed to resolve",
                {"error": str(exc)},
            )
        )

    if cfg is not None:
        checks.extend(_config_warnings(cfg))
        checks.append(_check_l2_config(cfg))
        checks.append(_check_bench_log_dir(cfg))

    checks.append(_check_internal_verify_env())
    checks.append(_check_model(args))
    if args.load_model:
        checks.append(_check_load_model(args, cfg))

    summary = _summary(checks)
    report = {
        "summary": summary,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "packages": {
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
        },
        "effective_config": _effective_config_payload(args, cfg),
        "model": _model_payload(args),
        "checks": [asdict(check) for check in checks],
    }
    return report

def _check_python() -> DoctorCheck:
    version_info = sys.version_info
    ok = version_info >= (3, 10)
    return DoctorCheck(
        "python",
        "ok" if ok else "fatal",
        "python version is supported" if ok else "python >= 3.10 is required",
        {"version": platform.python_version(), "executable": sys.executable},
    )

def _check_import(module_name: str) -> DoctorCheck:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return DoctorCheck(
            f"import_{module_name}",
            "fatal",
            f"failed to import {module_name}",
            {"error": str(exc)},
        )
    return DoctorCheck(
        f"import_{module_name}",
        "ok",
        f"{module_name} imports",
        {"module": getattr(module, "__name__", module_name)},
    )

def _check_metal() -> DoctorCheck:
    try:
        import mlx.core as mx

        available = bool(mx.metal.is_available())
        details = {"available": available}
        if available:
            try:
                details["device_info"] = dict(mx.device_info())
            except Exception as exc:
                details["device_info_error"] = str(exc)
        return DoctorCheck(
            "metal",
            "ok" if available else "fatal",
            "Metal backend is available" if available else "Metal backend is unavailable",
            details,
        )
    except Exception as exc:
        return DoctorCheck(
            "metal",
            "fatal",
            "failed to query Metal backend",
            {"error": str(exc)},
        )

def _check_scripts_importable() -> DoctorCheck:
    scripts = {
        "dflash": "dflash_mlx.cli:main",
    }
    details: dict[str, Any] = {}
    failed = []
    for script, target in scripts.items():
        module_name, func_name = target.split(":", 1)
        try:
            module = importlib.import_module(module_name)
            callable(getattr(module, func_name))
        except Exception as exc:
            failed.append(script)
            details[script] = {"target": target, "error": str(exc)}
        else:
            details[script] = {"target": target, "importable": True}
    return DoctorCheck(
        "scripts",
        "fatal" if failed else "ok",
        "console script modules import" if not failed else "some console scripts fail to import",
        details,
    )

def _config_warnings(cfg: EffectiveRuntimeConfig) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    if cfg.target_fa_window > 0:
        checks.append(
            DoctorCheck(
                "target_fa_window",
                "warning",
                "target FA window disables prefix-cache and L2 snapshots",
                {
                    "target_fa_window": cfg.target_fa_window,
                    "prefix_cache": cfg.prefix_cache,
                    "prefix_cache_l2": cfg.prefix_cache_l2,
                },
            )
        )
    return checks

def _check_l2_config(cfg: EffectiveRuntimeConfig) -> DoctorCheck:
    if not cfg.prefix_cache_l2:
        return DoctorCheck(
            "l2_cache",
            "ok",
            "L2 prefix cache is disabled",
            {
                "enabled": False,
                "dir": cfg.prefix_cache_l2_dir,
                "max_bytes": cfg.prefix_cache_l2_max_bytes,
            },
        )
    ok, error = _is_writable_dir(cfg.prefix_cache_l2_dir)
    return DoctorCheck(
        "l2_cache",
        "ok" if ok else "fatal",
        "L2 prefix cache directory is writable" if ok else "L2 prefix cache directory is not writable",
        {
            "enabled": True,
            "dir": cfg.prefix_cache_l2_dir,
            "max_bytes": cfg.prefix_cache_l2_max_bytes,
            "error": error,
        },
    )

def _check_bench_log_dir(cfg: EffectiveRuntimeConfig) -> DoctorCheck:
    if not cfg.bench_log_dir:
        return DoctorCheck(
            "bench_log_dir",
            "ok",
            "bench log directory is not configured",
            {"configured": False},
        )
    ok, error = _is_writable_dir(cfg.bench_log_dir)
    return DoctorCheck(
        "bench_log_dir",
        "ok" if ok else "fatal",
        "bench log directory is writable" if ok else "bench log directory is not writable",
        {"configured": True, "dir": cfg.bench_log_dir, "error": error},
    )

def _check_internal_verify_env() -> DoctorCheck:
    active = {
        key: os.environ[key]
        for key in _VERIFY_INTERNAL_ENVS
        if os.environ.get(key, "").strip()
    }
    return DoctorCheck(
        "internal_verify_env",
        "warning" if active else "ok",
        "internal verify env overrides are active" if active else "no internal verify env overrides",
        {"active": active},
    )

def _check_model(args: argparse.Namespace) -> DoctorCheck:
    if not args.model:
        return DoctorCheck(
            "model",
            "ok",
            "no model provided; model-specific checks skipped",
            {"model": None},
        )
    try:
        from dflash_mlx.generate import resolve_optional_draft_ref

        draft = resolve_optional_draft_ref(args.model, args.draft_model)
    except Exception as exc:
        return DoctorCheck(
            "model",
            "fatal",
            "failed to resolve draft model",
            {"model": args.model, "draft": args.draft_model, "error": str(exc)},
        )
    status = "ok" if draft else "fatal"
    return DoctorCheck(
        "model",
        status,
        "draft model resolved" if draft else "no draft model resolved for target",
        {"model": args.model, "draft": draft, "explicit_draft": bool(args.draft_model)},
    )

def _check_load_model(
    args: argparse.Namespace,
    cfg: EffectiveRuntimeConfig | None,
) -> DoctorCheck:
    if not args.model:
        return DoctorCheck(
            "load_model",
            "warning",
            "--load-model was requested without --model",
            {"model": None},
        )
    try:
        from dflash_mlx.runtime import load_target_bundle

        context = build_runtime_context(cfg) if cfg is not None else None
        load_target_bundle(
            args.model,
            lazy=True,
            verify_config=context.verify if context is not None else None,
        )
    except Exception as exc:
        return DoctorCheck(
            "load_model",
            "fatal",
            "target model failed to load",
            {"model": args.model, "error": str(exc)},
        )
    return DoctorCheck(
        "load_model",
        "ok",
        "target model loads",
        {"model": args.model},
    )

def _effective_config_payload(
    args: argparse.Namespace,
    cfg: EffectiveRuntimeConfig | None,
) -> dict[str, Any]:
    if cfg is None:
        return {"resolved": False, "values": None, "sources": {}}
    values = asdict(cfg)
    return {
        "resolved": True,
        "values": values,
        "sources": {
            field: _source_for_field(args, field, cfg.profile)
            for field in values
        },
    }

def _source_for_field(args: argparse.Namespace, field: str, profile: str) -> str:
    cli_attr = _CLI_FIELDS.get(field)
    if cli_attr and getattr(args, cli_attr, None) is not None:
        return "cli"
    env_key, default_value = _CONFIG_FIELDS[field]
    if env_key is not None and os.environ.get(env_key, "").strip():
        return "env"
    if field == "profile" and profile == "balanced":
        return "default"
    if default_value is not None:
        return "default"
    return "profile"

def _model_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "target": args.model,
        "draft": args.draft_model,
        "load_model": bool(args.load_model),
    }

def _summary(checks: list[DoctorCheck]) -> dict[str, Any]:
    fatals = sum(1 for check in checks if check.status == "fatal")
    warnings = sum(1 for check in checks if check.status == "warning")
    status = "fatal" if fatals else "warning" if warnings else "ok"
    return {
        "status": status,
        "checks": len(checks),
        "fatals": fatals,
        "warnings": warnings,
    }

def _print_text_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"dflash doctor: {summary['status']}")
    for check in report["checks"]:
        print(f"[{check['status']}] {check['name']}: {check['message']}")
    cfg = report["effective_config"]
    if cfg.get("resolved"):
        values = cfg["values"]
        print(
            "effective config: "
            f"profile={values['profile']} "
            f"prefill_step_size={values['prefill_step_size']} "
            f"prefix_cache={values['prefix_cache']} "
            f"l2={values['prefix_cache_l2']} "
            f"target_fa_window={values['target_fa_window']} "
            f"verify_mode={values['verify_mode']}"
        )

def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None

def _is_writable_dir(path: str) -> tuple[bool, str | None]:
    try:
        directory = Path(path).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".dflash-doctor-", delete=True):
            pass
    except Exception as exc:
        return False, str(exc)
    return True, None

if __name__ == "__main__":
    main()

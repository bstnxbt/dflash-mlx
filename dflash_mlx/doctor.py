# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
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

from dflash_mlx.runtime.config import (
    add_runtime_config_arguments,
    EffectiveRuntimeConfig,
    resolve_runtime_config,
    runtime_config_sources,
)
from dflash_mlx.runtime.context import build_runtime_context
from dflash_mlx.runtime.chip_detect import ChipProfile, detect_chip

_VERIFY_INTERNAL_ENVS = (
    "DFLASH_VERIFY_LINEAR",
    "DFLASH_VERIFY_QMM",
    "DFLASH_VERIFY_VARIANT",
    "DFLASH_VERIFY_MAX_N",
    "DFLASH_VERIFY_QMM_KPARTS",
    "DFLASH_VERIFY_INCLUDE",
)

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
    add_runtime_config_arguments(parser)
    return parser

def run(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(list(argv) if argv is not None else None)

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
                "runtime_config",
                "ok",
                "runtime config resolved",
                {
                    "prefill_step_size": cfg.prefill_step_size,
                    "prefix_cache": cfg.prefix_cache,
                    "prefix_cache_l2": cfg.prefix_cache_l2,
                },
            )
        )
    except Exception as exc:
        checks.append(
            DoctorCheck(
                "runtime_config",
                "fatal",
                "runtime config failed to resolve",
                {"error": str(exc)},
            )
        )

    if cfg is not None:
        checks.extend(_config_warnings(cfg))
        checks.extend(_chip_checks())
        checks.append(_check_l2_config(cfg))

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
            entrypoint = getattr(module, func_name)
            if not callable(entrypoint):
                raise TypeError(f"{target} is not callable")
        except Exception as exc:
            failed.append(script)
            details[script] = {"target": target, "error": str(exc)}
        else:
            details[script] = {"target": target, "callable": True}
    return DoctorCheck(
        "scripts",
        "fatal" if failed else "ok",
        "console script entrypoints resolve"
        if not failed
        else "some console scripts fail to resolve",
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

def _chip_checks() -> list[DoctorCheck]:
    try:
        return _chip_warnings(detect_chip())
    except Exception as exc:
        return [
            DoctorCheck(
                "chip_profile",
                "warning",
                "Apple Silicon chip profile detection failed",
                {"error": str(exc)},
            )
        ]


def _chip_warnings(profile: ChipProfile) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = [
        DoctorCheck(
            "chip_profile",
            "ok",
            "Apple Silicon chip profile resolved"
            if profile.metal_available and profile.arch_gen
            else "Apple Silicon chip profile unavailable",
            _chip_details(profile),
        )
    ]
    if profile.bf16_emulated:
        checks.append(
            DoctorCheck(
                "old_apple_bf16",
                "warning",
                "M1/M2 GPUs emulate BF16; quantized DFlash drafts load as FP16 on this runtime",
                _chip_details(profile),
            )
        )
    if profile.pre_m5_gpu:
        checks.append(
            DoctorCheck(
                "nax_unavailable",
                "warning",
                "NAX matrix kernels are unavailable before M5; older Apple GPUs use steel fallback kernels",
                _chip_details(profile),
            )
        )
    return checks


def _chip_details(profile: ChipProfile) -> dict[str, Any]:
    return {
        "metal_available": profile.metal_available,
        "architecture": profile.arch_string,
        "arch_gen": profile.arch_gen,
        "family": profile.family,
        "tier": profile.tier,
        "bf16_native": profile.bf16_native,
        "bf16_emulated": profile.bf16_emulated,
        "nax_capable": profile.nax_capable,
        "macos_version": profile.macos_version,
    }

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
        from dflash_mlx.runtime.registry import resolve_optional_draft_ref

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
        from dflash_mlx.runtime.bundle import load_runtime_bundle

        context = build_runtime_context(cfg) if cfg is not None else None
        bundle = load_runtime_bundle(
            model_ref=args.model,
            draft_ref=args.draft_model,
            verify_config=context.verify if context is not None else None,
        )
        required_fields = (
            "target_model",
            "tokenizer",
            "target_meta",
            "draft_model",
            "draft_meta",
            "draft_backend",
            "target_ops",
            "resolved_model_ref",
            "resolved_draft_ref",
        )
        missing_fields = [
            field for field in required_fields if getattr(bundle, field, None) is None
        ]
        if missing_fields:
            raise ValueError(
                "runtime bundle is incomplete: " + ", ".join(missing_fields)
            )
    except Exception as exc:
        return DoctorCheck(
            "load_model",
            "fatal",
            "runtime bundle failed to load",
            {"model": args.model, "error": str(exc)},
        )
    return DoctorCheck(
        "load_model",
        "ok",
        "runtime bundle loads",
        {
            "model": bundle.resolved_model_ref,
            "draft": bundle.resolved_draft_ref,
        },
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
        "sources": runtime_config_sources(args, cfg),
    }

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

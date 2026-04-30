# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

ArtifactKind = Literal["diagnostics", "benchmark", "trace"]

OUTPUT_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_ROOT = Path(".artifacts/dflash")

def resolve_artifact_root() -> Path:
    return DEFAULT_ARTIFACT_ROOT

def create_run_dir(
    kind: ArtifactKind,
    label: str,
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path is not None:
        run_dir = Path(explicit_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    base = resolve_artifact_root() / _kind_dir(kind) / f"{_timestamp()}-{_slug(label)}"
    run_dir = base
    suffix = 2
    while run_dir.exists():
        run_dir = Path(f"{base}-{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def write_manifest(
    run_dir: Path,
    *,
    kind: ArtifactKind,
    label: str,
    argv: list[str],
    model: str | None = None,
    draft: str | None = None,
    profile: str | None = None,
    effective_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_manifest(
        kind=kind,
        label=label,
        argv=argv,
        model=model,
        draft=draft,
        profile=profile,
        effective_config=effective_config,
    )
    write_json(run_dir / "manifest.json", payload)
    return payload

def build_manifest(
    *,
    kind: ArtifactKind,
    label: str,
    argv: list[str],
    model: str | None = None,
    draft: str | None = None,
    profile: str | None = None,
    effective_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "timestamp": _timestamp(),
        "argv": list(argv),
        "cwd": os.getcwd(),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_dirty": _git_dirty(),
        "dflash_version": _package_version("dflash-mlx"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "model": model,
        "draft": draft,
        "profile": profile,
        "effective_config": effective_config or {},
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
    }

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")

def slug(value: str) -> str:
    return _slug(value)

def _kind_dir(kind: ArtifactKind) -> str:
    if kind == "benchmark":
        return "benchmarks"
    if kind == "trace":
        return "traces"
    return "diagnostics"

def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def _slug(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower())
    out = re.sub(r"-+", "-", out).strip("-")
    return out or "run"

def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"

def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return True

def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"

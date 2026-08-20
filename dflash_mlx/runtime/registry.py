# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelRuntimeDefaults:
    draft_quant: str | None = None


@dataclass(frozen=True)
class ModelSupportSpec:
    target_names: tuple[str, ...]
    draft_ref: str
    target_family: str | None = None
    defaults: ModelRuntimeDefaults = field(default_factory=ModelRuntimeDefaults)


W4_DEFAULTS = ModelRuntimeDefaults(draft_quant="w4")

MODEL_SUPPORT_SPECS: tuple[ModelSupportSpec, ...] = (
    ModelSupportSpec(("Qwen3.5-4B",), "z-lab/Qwen3.5-4B-DFlash", "hybrid_gdn"),
    ModelSupportSpec(
        ("Qwen3.5-9B",), "z-lab/Qwen3.5-9B-DFlash", "hybrid_gdn", W4_DEFAULTS
    ),
    ModelSupportSpec(
        ("Qwen3.5-27B",), "z-lab/Qwen3.5-27B-DFlash", "hybrid_gdn", W4_DEFAULTS
    ),
    ModelSupportSpec(
        ("Qwen3.5-35B-A3B",),
        "z-lab/Qwen3.5-35B-A3B-DFlash",
        "hybrid_gdn",
        W4_DEFAULTS,
    ),
    ModelSupportSpec(
        ("Qwen3.6-27B",), "z-lab/Qwen3.6-27B-DFlash", "hybrid_gdn", W4_DEFAULTS
    ),
    ModelSupportSpec(
        ("Qwen3.6-35B-A3B",),
        "z-lab/Qwen3.6-35B-A3B-DFlash",
        "hybrid_gdn",
        W4_DEFAULTS,
    ),
    ModelSupportSpec(
        ("Qwen3.8-27B",), "z-lab/Qwen3.8-27B-DFlash2", "hybrid_gdn", W4_DEFAULTS
    ),
    ModelSupportSpec(("Qwen3-4B",), "z-lab/Qwen3-4B-DFlash-b16", "pure_attention"),
    ModelSupportSpec(("Qwen3-8B",), "z-lab/Qwen3-8B-DFlash-b16", "pure_attention"),
    ModelSupportSpec(
        ("gemma-4-31b-it",),
        "z-lab/gemma-4-31B-it-DFlash",
        "gemma4_swa",
        W4_DEFAULTS,
    ),
    ModelSupportSpec(
        ("gemma-4-26b-a4b-it",),
        "z-lab/gemma-4-26B-A4B-it-DFlash",
        "gemma4_swa",
        W4_DEFAULTS,
    ),
)

DRAFT_REGISTRY: dict[str, str] = {
    target_name: spec.draft_ref
    for spec in MODEL_SUPPORT_SPECS
    for target_name in spec.target_names
}

_NORMALIZED_SUPPORT_SPECS: dict[str, ModelSupportSpec] = {
    target_name.lower(): spec
    for spec in MODEL_SUPPORT_SPECS
    for target_name in spec.target_names
}


def supported_base_models() -> str:
    return ", ".join(DRAFT_REGISTRY.keys())


def _strip_model_org(model_ref: str) -> str:
    return str(model_ref).rsplit("/", 1)[-1].strip()


def resolve_model_support_spec(model_ref: str) -> ModelSupportSpec | None:
    stripped_name = _strip_model_org(model_ref)
    lowered_name = stripped_name.lower()

    exact = _NORMALIZED_SUPPORT_SPECS.get(lowered_name)
    if exact is not None:
        return exact

    matching_bases = [
        base_name
        for base_name in _NORMALIZED_SUPPORT_SPECS
        if lowered_name == base_name
        or lowered_name.startswith(base_name + "-")
        or lowered_name.startswith(base_name + "_")
    ]
    if not matching_bases:
        return None

    best_match = max(matching_bases, key=len)
    return _NORMALIZED_SUPPORT_SPECS[best_match]


def resolve_optional_draft_ref(model_ref: str, draft_ref: str | None) -> str | None:
    if draft_ref:
        return draft_ref
    spec = resolve_model_support_spec(model_ref)
    return spec.draft_ref if spec is not None else None


def resolve_effective_draft_quant(
    *,
    draft_quant: str | None,
    resolved_draft_ref: str | None,
    support_spec: ModelSupportSpec | None,
) -> str | None:
    requested = (draft_quant or "").strip()
    if requested:
        if requested.lower() == "none":
            return None
        return requested
    if support_spec is None or not support_spec.defaults.draft_quant:
        return None
    if resolved_draft_ref != support_spec.draft_ref:
        return None
    return support_spec.defaults.draft_quant


def validate_support_spec_family(
    *,
    model_ref: str,
    support_spec: ModelSupportSpec | None,
    actual_family: str,
) -> None:
    if (
        support_spec is not None
        and support_spec.target_family is not None
        and actual_family != support_spec.target_family
    ):
        raise ValueError(
            f"DFlash support spec target family mismatch for '{model_ref}': "
            f"expected {support_spec.target_family}, got {actual_family}"
        )

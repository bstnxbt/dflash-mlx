# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from types import SimpleNamespace

import pytest

from dflash_mlx.runtime import bundle as runtime_bundle
from dflash_mlx.runtime import stream_dflash_generate
from dflash_mlx.runtime.registry import (
    DRAFT_REGISTRY,
    resolve_effective_draft_quant,
    resolve_model_support_spec,
    resolve_optional_draft_ref,
)


EXPECTED_DRAFT_REGISTRY = {
    "Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
    "Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
    "Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
    "Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
    "Qwen3.6-27B": "z-lab/Qwen3.6-27B-DFlash",
    "Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
    "Qwen3.8-27B": "z-lab/Qwen3.8-27B-DFlash2",
    "Qwen3-4B": "z-lab/Qwen3-4B-DFlash-b16",
    "Qwen3-8B": "z-lab/Qwen3-8B-DFlash-b16",
    "gemma-4-31b-it": "z-lab/gemma-4-31B-it-DFlash",
    "gemma-4-26b-a4b-it": "z-lab/gemma-4-26B-A4B-it-DFlash",
}


def _loaded_target(target_model, tokenizer, meta, ops):
    return SimpleNamespace(
        model=target_model,
        tokenizer=tokenizer,
        meta=meta,
        target_ops=ops,
    )


def test_runtime_registry_preserves_draft_resolution():
    assert DRAFT_REGISTRY == EXPECTED_DRAFT_REGISTRY
    assert (
        resolve_optional_draft_ref("Qwen/Qwen3.5-9B", None)
        == "z-lab/Qwen3.5-9B-DFlash"
    )
    assert (
        resolve_optional_draft_ref("Qwen/Qwen3.5-27B", None)
        == "z-lab/Qwen3.5-27B-DFlash"
    )
    assert (
        resolve_optional_draft_ref("mlx-community/Qwen3.5-9B-4bit", None)
        == "z-lab/Qwen3.5-9B-DFlash"
    )
    assert (
        resolve_optional_draft_ref("mlx-community/Qwen3.6-27B-4bit", None)
        == "z-lab/Qwen3.6-27B-DFlash"
    )
    assert (
        resolve_optional_draft_ref("mlx-community/Qwen3.8-27B-4bit", None)
        == "z-lab/Qwen3.8-27B-DFlash2"
    )
    assert (
        resolve_optional_draft_ref("mlx-community/gemma-4-26b-a4b-it-4bit", None)
        == "z-lab/gemma-4-26B-A4B-it-DFlash"
    )
    assert resolve_model_support_spec("Qwen/Qwen3-4B").target_family == "pure_attention"
    assert (
        resolve_model_support_spec("mlx-community/gemma-4-31b-it-4bit").target_family
        == "gemma4_swa"
    )
    assert resolve_model_support_spec("Qwen/Qwen3.5-9B").defaults.draft_quant == "w4"
    assert resolve_model_support_spec("Qwen/Qwen3.8-27B").defaults.draft_quant == "w4"
    assert (
        resolve_model_support_spec("mlx-community/gemma-4-31b-it-4bit").defaults.draft_quant
        == "w4"
    )


def test_runtime_registry_resolves_effective_draft_quant():
    spec = resolve_model_support_spec("Qwen/Qwen3.5-9B")
    assert spec is not None
    assert (
        resolve_effective_draft_quant(
            draft_quant=None,
            resolved_draft_ref="z-lab/Qwen3.5-9B-DFlash",
            support_spec=spec,
        )
        == "w4"
    )
    assert (
        resolve_effective_draft_quant(
            draft_quant="none",
            resolved_draft_ref="z-lab/Qwen3.5-9B-DFlash",
            support_spec=spec,
        )
        is None
    )
    assert (
        resolve_effective_draft_quant(
            draft_quant=None,
            resolved_draft_ref="manual/draft",
            support_spec=spec,
        )
        is None
    )
    assert (
        resolve_effective_draft_quant(
            draft_quant="w8:gs128",
            resolved_draft_ref="z-lab/Qwen3.5-9B-DFlash",
            support_spec=spec,
        )
        == "w8:gs128"
    )


def test_runtime_registry_explicit_draft_wins():
    assert resolve_optional_draft_ref("unknown/model", "manual/draft") == "manual/draft"


def test_runtime_stream_requires_owned_dependencies():
    cases = [
        ({}, "runtime_context is required"),
        ({"runtime_context": object()}, "target_ops is required"),
        (
            {"runtime_context": object(), "target_ops": object()},
            "draft_backend is required",
        ),
    ]
    for kwargs, message in cases:
        stream = stream_dflash_generate(**kwargs)
        with pytest.raises(ValueError, match=message):
            next(stream)


def test_runtime_bundle_unknown_target_without_draft_fails_clearly(monkeypatch):
    monkeypatch.setattr(
        runtime_bundle,
        "load_target_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("load_target_bundle should not be called")
        ),
    )
    monkeypatch.setattr(
        runtime_bundle,
        "load_draft_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("load_draft_bundle should not be called")
        ),
    )

    with pytest.raises(ValueError, match="No DFlash draft model found for 'unknown/model'"):
        runtime_bundle.load_runtime_bundle(model_ref="unknown/model", draft_ref=None)


def test_runtime_bundle_loads_and_binds_draft(monkeypatch):
    target_model = object()
    tokenizer = object()
    draft_model = SimpleNamespace(bound=False)
    draft_backend = object()
    ops = SimpleNamespace(family=lambda _target_model: "hybrid_gdn")
    calls = []
    target_calls = []

    def fake_load_target_bundle(*args, **kwargs):
        target_calls.append((args, kwargs))
        return _loaded_target(
            target_model,
            tokenizer,
            {"resolved_model_ref": "Qwen/Qwen3.5-9B"},
            ops,
        )

    monkeypatch.setattr(runtime_bundle, "load_target_bundle", fake_load_target_bundle)
    monkeypatch.setattr(
        runtime_bundle,
        "load_draft_bundle",
        lambda draft_ref, **kwargs: calls.append(("draft", draft_ref, kwargs))
        or (draft_model, {"resolved_model_ref": draft_ref}),
    )

    def fake_bind(draft, target, *, target_ops):
        calls.append(("bind", draft, target, target_ops))
        draft.bound = True
        return target_ops

    monkeypatch.setattr(runtime_bundle, "bind_draft_to_target", fake_bind)
    monkeypatch.setattr(runtime_bundle, "EagerDraftBackend", lambda: draft_backend)

    bundle = runtime_bundle.load_runtime_bundle(
        model_ref="Qwen/Qwen3.5-9B",
        draft_ref=None,
        draft_quant="w4",
        verify_config=object(),
        quantize_kv_cache=True,
    )

    assert bundle.target_model is target_model
    assert bundle.tokenizer is tokenizer
    assert bundle.draft_model is draft_model
    assert bundle.draft_backend is draft_backend
    assert bundle.target_ops is ops
    assert bundle.resolved_model_ref == "Qwen/Qwen3.5-9B"
    assert bundle.resolved_draft_ref == "z-lab/Qwen3.5-9B-DFlash"
    assert bundle.support_spec is not None
    assert bundle.support_spec.draft_ref == "z-lab/Qwen3.5-9B-DFlash"
    assert bundle.support_spec.target_family == "hybrid_gdn"
    assert draft_model.bound is True
    assert target_calls[0][1]["quantize_kv_cache"] is True
    assert calls[0] == (
        "draft",
        "z-lab/Qwen3.5-9B-DFlash",
        {"lazy": True, "draft_quant": "w4"},
    )
    assert calls[1] == ("bind", draft_model, target_model, ops)


def test_runtime_bundle_applies_model_default_draft_quant(monkeypatch):
    target_model = object()
    tokenizer = object()
    draft_model = SimpleNamespace(bound=False)
    draft_backend = object()
    ops = SimpleNamespace(family=lambda _target_model: "hybrid_gdn")
    draft_calls = []

    monkeypatch.setattr(
        runtime_bundle,
        "load_target_bundle",
        lambda *args, **kwargs: _loaded_target(
            target_model,
            tokenizer,
            {"resolved_model_ref": "Qwen/Qwen3.5-9B"},
            ops,
        ),
    )
    monkeypatch.setattr(
        runtime_bundle,
        "load_draft_bundle",
        lambda draft_ref, **kwargs: draft_calls.append((draft_ref, kwargs))
        or (draft_model, {"resolved_model_ref": draft_ref}),
    )
    monkeypatch.setattr(runtime_bundle, "bind_draft_to_target", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_bundle, "EagerDraftBackend", lambda: draft_backend)

    bundle = runtime_bundle.load_runtime_bundle(
        model_ref="Qwen/Qwen3.5-9B",
        draft_ref=None,
    )

    assert draft_calls == [
        ("z-lab/Qwen3.5-9B-DFlash", {"lazy": True, "draft_quant": "w4"})
    ]
    assert bundle.effective_draft_quant == "w4"
    assert bundle.draft_meta["draft_quant_spec"] == "w4"
    assert bundle.draft_meta["draft_quant_source"] == "model_default"


def test_runtime_bundle_does_not_forward_sdpa_override(monkeypatch):
    target_model = object()
    tokenizer = object()
    draft_model = SimpleNamespace(bound=False)
    draft_backend = object()
    ops = SimpleNamespace(family=lambda _target_model: "hybrid_gdn")
    target_calls = []

    def fake_load_target_bundle(*args, **kwargs):
        target_calls.append((args, kwargs))
        return _loaded_target(
            target_model,
            tokenizer,
            {"resolved_model_ref": "mlx-community/Qwen3.6-35B-A3B-4bit"},
            ops,
        )

    monkeypatch.setattr(runtime_bundle, "load_target_bundle", fake_load_target_bundle)
    monkeypatch.setattr(
        runtime_bundle,
        "load_draft_bundle",
        lambda draft_ref, **kwargs: (draft_model, {"resolved_model_ref": draft_ref}),
    )
    monkeypatch.setattr(runtime_bundle, "bind_draft_to_target", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_bundle, "EagerDraftBackend", lambda: draft_backend)

    bundle = runtime_bundle.load_runtime_bundle(
        model_ref="mlx-community/Qwen3.6-35B-A3B-4bit",
        draft_ref=None,
    )

    assert bundle.support_spec is not None
    assert "split_full_attention_sdpa" not in target_calls[0][1]
    assert "split_full_attention_sdpa_default" not in target_calls[0][1]


def test_runtime_bundle_draft_quant_none_disables_model_default(monkeypatch):
    target_model = object()
    tokenizer = object()
    draft_model = SimpleNamespace(bound=False)
    draft_backend = object()
    ops = SimpleNamespace(family=lambda _target_model: "hybrid_gdn")
    draft_calls = []

    monkeypatch.setattr(
        runtime_bundle,
        "load_target_bundle",
        lambda *args, **kwargs: _loaded_target(
            target_model,
            tokenizer,
            {"resolved_model_ref": "Qwen/Qwen3.5-9B"},
            ops,
        ),
    )
    monkeypatch.setattr(
        runtime_bundle,
        "load_draft_bundle",
        lambda draft_ref, **kwargs: draft_calls.append((draft_ref, kwargs))
        or (draft_model, {"resolved_model_ref": draft_ref}),
    )
    monkeypatch.setattr(runtime_bundle, "bind_draft_to_target", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_bundle, "EagerDraftBackend", lambda: draft_backend)

    bundle = runtime_bundle.load_runtime_bundle(
        model_ref="Qwen/Qwen3.5-9B",
        draft_ref=None,
        draft_quant="none",
    )

    assert draft_calls == [
        ("z-lab/Qwen3.5-9B-DFlash", {"lazy": True, "draft_quant": None})
    ]
    assert bundle.effective_draft_quant is None
    assert bundle.draft_meta["draft_quant_spec"] is None
    assert bundle.draft_meta["draft_quant_source"] == "none"


def test_runtime_bundle_draft_override_still_exposes_support_spec(monkeypatch):
    target_model = object()
    draft_backend = object()
    ops = SimpleNamespace(family=lambda _target_model: "hybrid_gdn")
    monkeypatch.setattr(
        runtime_bundle,
        "load_target_bundle",
        lambda *args, **kwargs: _loaded_target(
            target_model,
            object(),
            {"resolved_model_ref": "Qwen/Qwen3.5-9B"},
            ops,
        ),
    )
    monkeypatch.setattr(
        runtime_bundle,
        "load_draft_bundle",
        lambda draft_ref, **kwargs: (object(), {"resolved_model_ref": draft_ref}),
    )
    monkeypatch.setattr(runtime_bundle, "bind_draft_to_target", lambda *args, **kwargs: None)
    monkeypatch.setattr(runtime_bundle, "EagerDraftBackend", lambda: draft_backend)

    bundle = runtime_bundle.load_runtime_bundle(
        model_ref="Qwen/Qwen3.5-9B",
        draft_ref="manual/draft",
    )

    assert bundle.resolved_draft_ref == "manual/draft"
    assert bundle.draft_backend is draft_backend
    assert bundle.support_spec is not None
    assert bundle.support_spec.draft_ref == "z-lab/Qwen3.5-9B-DFlash"


def test_runtime_bundle_rejects_target_family_mismatch(monkeypatch):
    ops = SimpleNamespace(family=lambda _target_model: "pure_attention")
    monkeypatch.setattr(
        runtime_bundle,
        "load_target_bundle",
        lambda *args, **kwargs: _loaded_target(
            object(),
            object(),
            {"resolved_model_ref": "Qwen/Qwen3.5-9B"},
            ops,
        ),
    )
    monkeypatch.setattr(
        runtime_bundle,
        "load_draft_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("load_draft_bundle should not be called")
        ),
    )
    monkeypatch.setattr(
        runtime_bundle,
        "bind_draft_to_target",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bind_draft_to_target should not be called")
        ),
    )

    with pytest.raises(
        ValueError,
        match="expected hybrid_gdn, got pure_attention",
    ):
        runtime_bundle.load_runtime_bundle(model_ref="Qwen/Qwen3.5-9B", draft_ref=None)


def test_runtime_bundle_preserves_specific_draft_load_value_error(monkeypatch):
    ops = SimpleNamespace(family=lambda _target_model: "hybrid_gdn")
    monkeypatch.setattr(
        runtime_bundle,
        "load_target_bundle",
        lambda *args, **kwargs: _loaded_target(
            object(),
            object(),
            {"resolved_model_ref": "Qwen/Qwen3.5-9B"},
            ops,
        ),
    )
    monkeypatch.setattr(
        runtime_bundle,
        "load_draft_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("contains draft-owned lm_head weights")
        ),
    )
    monkeypatch.setattr(
        runtime_bundle,
        "bind_draft_to_target",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bind_draft_to_target should not be called")
        ),
    )

    with pytest.raises(ValueError, match="contains draft-owned lm_head weights"):
        runtime_bundle.load_runtime_bundle(model_ref="Qwen/Qwen3.5-9B", draft_ref=None)

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

import mlx.core as mx
from dflash_mlx.runtime import (
    load_draft_bundle,
    load_target_bundle,
    stream_dflash_generate,
)

DRAFT_REGISTRY = {
    "Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
    "Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
    "Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
    "Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
    "Qwen3.6-27B": "z-lab/Qwen3.6-27B-DFlash",
    "Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
    "Qwen3-4B": "z-lab/Qwen3-4B-DFlash-b16",
    "Qwen3-8B": "z-lab/Qwen3-8B-DFlash-b16",
}

_NORMALIZED_DRAFT_REGISTRY = {
    key.lower(): value for key, value in DRAFT_REGISTRY.items()
}

def _supported_base_models() -> str:
    return ", ".join(DRAFT_REGISTRY.keys())

def _strip_model_org(model_ref: str) -> str:
    return str(model_ref).rsplit("/", 1)[-1].strip()

def get_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_token_ids = list(getattr(tokenizer, "eos_token_ids", None) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in eos_token_ids:
        eos_token_ids.append(int(eos_token_id))
    return eos_token_ids

def resolve_optional_draft_ref(model_ref: str, draft_ref: Optional[str]) -> Optional[str]:
    if draft_ref:
        return draft_ref

    stripped_name = _strip_model_org(model_ref)
    lowered_name = stripped_name.lower()

    exact = _NORMALIZED_DRAFT_REGISTRY.get(lowered_name)
    if exact is not None:
        return exact

    matching_bases = [
        base_name
        for base_name in _NORMALIZED_DRAFT_REGISTRY
        if lowered_name == base_name
        or lowered_name.startswith(base_name + "-")
        or lowered_name.startswith(base_name + "_")
    ]
    if not matching_bases:
        return None

    best_match = max(matching_bases, key=len)
    return _NORMALIZED_DRAFT_REGISTRY[best_match]

def decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        return str(tokenizer.decode(int(token_id)))

def generation_tps_from_summary(summary: dict[str, Any]) -> float:
    elapsed_us = float(summary.get("elapsed_us", 0.0))
    phase_timings = dict(summary.get("phase_timings_us", {}))
    prefill_us = float(summary.get("prefill_us", phase_timings.get("prefill", 0.0)))
    generation_tokens = int(summary.get("generation_tokens", 0))
    generation_us = max(0.0, elapsed_us - prefill_us)
    return (generation_tokens / (generation_us / 1e6)) if generation_us > 0.0 else 0.0

def load_runtime_components(
    *,
    model_ref: str,
    draft_ref: Optional[str],
    draft_quant: Optional[str] = None,
):
    resolved_draft_ref = resolve_optional_draft_ref(model_ref, draft_ref)
    if not resolved_draft_ref:
        raise ValueError(
            f"No DFlash draft model found for '{model_ref}'.\n"
            f"Use --draft to specify one, or check https://huggingface.co/z-lab for available drafts.\n"
            f"Supported base models: {_supported_base_models()}"
        )
    target_model, tokenizer, _ = load_target_bundle(model_ref, lazy=True)
    try:
        draft_model, _ = load_draft_bundle(resolved_draft_ref, lazy=True, draft_quant=draft_quant)
    except Exception as exc:
        raise ValueError(
            f"Failed to load DFlash draft model '{resolved_draft_ref}' for '{model_ref}'."
        ) from exc
    return target_model, tokenizer, draft_model, resolved_draft_ref

def run_generate(
    *,
    model_ref: str,
    prompt: str,
    max_tokens: int,
    use_chat_template: bool,
    draft_ref: Optional[str],
) -> int:
    target_model, tokenizer, draft_model, _ = load_runtime_components(
        model_ref=model_ref,
        draft_ref=draft_ref,
    )
    stop_token_ids = get_stop_token_ids(tokenizer)
    stream = stream_dflash_generate(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=use_chat_template,
        stop_token_ids=stop_token_ids,
    )

    summary: Optional[dict[str, Any]] = None
    for event in stream:
        if event.get("event") == "token":
            sys.stdout.write(decode_token(tokenizer, int(event["token_id"])))
            sys.stdout.flush()
        elif event.get("event") == "summary":
            summary = event

    if summary is None:
        return 1

    tps = generation_tps_from_summary(summary)
    acceptance_pct = float(summary.get("acceptance_ratio", 0.0)) * 100.0
    token_count = int(summary.get("generation_tokens", 0))
    sys.stderr.write(
        f"\n{token_count} tokens | {tps:.1f} tok/s | {acceptance_pct:.1f}% acceptance\n"
    )
    sys.stderr.flush()
    return 0

def main() -> None:
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_cache_limit(wired_limit // 4)
    parser = argparse.ArgumentParser(description="Generate text with DFlash on MLX.")
    parser.add_argument("--model", required=True, help="Target model reference.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--draft", default=None, help="Optional draft model override.")
    args = parser.parse_args()
    raise SystemExit(
        run_generate(
            model_ref=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            use_chat_template=not args.no_chat_template,
            draft_ref=args.draft,
        )
    )

if __name__ == "__main__":
    main()

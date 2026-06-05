# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any, Optional

from dflash_mlx.engine.events import SummaryEvent, TokenEvent, is_engine_event
from dflash_mlx.metal_limits import apply_metal_limits
from dflash_mlx.runtime import (
    get_stop_token_ids,
    stream_dflash_generate,
)
from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.config import (
    GENERATE_RUNTIME_FIELDS,
    add_offline_runtime_arguments,
    offline_runtime_error_message,
    offline_runtime_kwargs,
)
from dflash_mlx.runtime.context import (
    build_offline_runtime_config,
    build_offline_runtime_context,
)

def decode_token(tokenizer: Any, token_id: int) -> str:
    token = int(token_id)
    try:
        return str(tokenizer.decode([token]))
    except TypeError:
        return str(tokenizer.decode(token))

def generation_tps_from_summary(summary: SummaryEvent) -> float:
    elapsed_us = float(summary.elapsed_us)
    phase_timings = dict(summary.phase_timings_us)
    prefill_us = float(phase_timings.get("prefill", 0.0))
    generation_tokens = int(summary.generation_tokens)
    generation_us = max(0.0, elapsed_us - prefill_us)
    return (generation_tokens / (generation_us / 1e6)) if generation_us > 0.0 else 0.0

def run_generate(
    *,
    model_ref: str,
    prompt: str,
    max_tokens: int,
    use_chat_template: bool,
    draft_ref: Optional[str],
    target_fa_window: int | None = None,
    prefill_step_size: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    verify_len_cap: int | None = None,
    verify_mode: str | None = None,
    draft_quant: Optional[str] = None,
) -> int:
    runtime_context = build_offline_runtime_context(
        target_fa_window=target_fa_window,
        prefill_step_size=prefill_step_size,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
        verify_mode=verify_mode,
    )
    bundle = load_runtime_bundle(
        model_ref=model_ref,
        draft_ref=draft_ref,
        draft_quant=draft_quant,
        verify_config=runtime_context.verify,
    )
    target_model = bundle.target_model
    tokenizer = bundle.tokenizer
    draft_model = bundle.draft_model
    draft_backend = bundle.draft_backend
    target_ops = bundle.target_ops
    stop_token_ids = get_stop_token_ids(tokenizer)
    stream = stream_dflash_generate(
        target_model=target_model,
        target_ops=target_ops,
        tokenizer=tokenizer,
        draft_model=draft_model,
        draft_backend=draft_backend,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=use_chat_template,
        stop_token_ids=stop_token_ids,
        runtime_context=runtime_context,
    )

    summary: Optional[SummaryEvent] = None
    try:
        for event in stream:
            if isinstance(event, TokenEvent):
                sys.stdout.write(decode_token(tokenizer, int(event.token_id)))
                sys.stdout.flush()
            elif isinstance(event, SummaryEvent):
                summary = event
            elif not is_engine_event(event):
                raise TypeError(f"Unsupported DFlash engine event: {type(event).__name__}")
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()

    if summary is None:
        return 1

    sys.stderr.write(f"\n{format_generation_summary(summary)}\n")
    sys.stderr.flush()
    return 0

def format_generation_summary(summary: SummaryEvent) -> str:
    tps = generation_tps_from_summary(summary)
    acceptance_pct = float(summary.acceptance_ratio) * 100.0
    token_count = int(summary.generation_tokens)
    line = f"{token_count} tokens | {tps:.1f} tok/s | {acceptance_pct:.1f}% acceptance"
    copyspec_hits = int(summary.copyspec_hits)
    if copyspec_hits > 0:
        line += (
            f" | copyspec {copyspec_hits} blocks"
            f" / {int(summary.copyspec_tokens)} tokens"
        )
    return line

def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    apply_metal_limits()
    parser = argparse.ArgumentParser(prog=prog, description="Generate text with DFlash on MLX.")
    parser.add_argument("--model", required=True, help="Target model reference.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--draft", default=None, help="Optional draft model override.")
    parser.add_argument(
        "--draft-quant",
        default=None,
        metavar="SPEC",
        help="Draft quantization override, e.g. w4:gs64; use 'none' to disable model defaults.",
    )
    add_offline_runtime_arguments(parser, GENERATE_RUNTIME_FIELDS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    runtime_kwargs = offline_runtime_kwargs(args, GENERATE_RUNTIME_FIELDS)
    try:
        build_offline_runtime_config(**runtime_kwargs)
    except ValueError as exc:
        raise SystemExit(offline_runtime_error_message(str(exc))) from exc
    raise SystemExit(
        run_generate(
            model_ref=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            use_chat_template=not args.no_chat_template,
            draft_ref=args.draft,
            **runtime_kwargs,
            draft_quant=args.draft_quant,
        )
    )

if __name__ == "__main__":
    main()

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
import textwrap
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.generate import stream_generate as mlx_stream_generate
from mlx_lm.utils import load as load_pristine_target

from dflash_mlx.runtime import (
    _prepare_prompt_tokens,
    load_draft_bundle,
    load_target_bundle,
    resolve_model_ref,
    stream_dflash_generate,
)

DEFAULT_PROMPT = "Implement a REST API with auth"
CONTENT_START_ROW = 4

def _is_tty() -> bool:
    return sys.stdout.isatty()

def _terminal_rows() -> int:
    try:
        return os.get_terminal_size(sys.stdout.fileno()).lines
    except OSError:
        return 24

def _terminal_cols() -> int:
    try:
        return os.get_terminal_size(sys.stdout.fileno()).columns
    except OSError:
        return 100

def _colorize(text: str, sgr: str) -> str:
    if not _is_tty():
        return text
    return f"\x1b[{sgr}m{text}\x1b[0m"

def _center_text(text: str) -> str:
    width = _terminal_cols()
    if len(text) >= width:
        return text
    return text.center(width)

def _machine_label() -> str:
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
        ).strip()
        memory_gb = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        ) // (1024**3)
        return f"{chip} {memory_gb}GB"
    except Exception:
        return "local"

def _get_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_token_ids = list(getattr(tokenizer, "eos_token_ids", None) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in eos_token_ids:
        eos_token_ids.append(int(eos_token_id))
    return eos_token_ids

def _display_target_label(model_ref: str) -> str:
    ref = str(model_ref)
    if "Qwen3.5-9B" in ref:
        return "Qwen 9B bf16"
    if "Qwen3-8B" in ref:
        return "Qwen 8B bf16"
    if "27B" in ref and "4bit" in ref:
        return "Qwen 27B 4bit"
    if "27B" in ref and "nvfp4" in ref.lower():
        return "Qwen 27B nvfp4"
    return Path(ref).name or ref

@contextlib.contextmanager
def _suppress_load_noise():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def _save_cursor() -> None:
    sys.stdout.write("\x1b7")

def _restore_cursor() -> None:
    sys.stdout.write("\x1b8")

def _write_status_line(text: str) -> None:
    if not _is_tty():
        return
    _save_cursor()
    sys.stdout.write("\x1b[3;1H")
    sys.stdout.write("\x1b[2K")
    sys.stdout.write(_colorize(text, "36"))
    _restore_cursor()
    sys.stdout.flush()

def _clear_status_line() -> None:
    if not _is_tty():
        return
    _save_cursor()
    sys.stdout.write("\x1b[3;1H")
    sys.stdout.write("\x1b[2K")
    _restore_cursor()
    sys.stdout.flush()

def _finalize_output(
    *,
    mode: str,
    prompt: str,
    max_tokens: int,
    info_line: str,
    footer: str,
    footer_sgr: str,
) -> None:
    _clear_status_line()
    if _is_tty():
        sys.stdout.write("\x1b[r")
        sys.stdout.write("\x1b[2J\x1b[H")
        _print_header(mode)
        _print_prompt_panel(prompt, max_tokens=max_tokens, info_line=info_line, enable_scroll_region=False)
        sys.stdout.write(_colorize(footer, footer_sgr))
        sys.stdout.write("\n")
    else:
        sys.stdout.write("\n---\n")
        sys.stdout.write(footer)
        sys.stdout.write("\n")
    sys.stdout.flush()

def _maybe_decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([int(token_id)]))
    except Exception:
        return str(tokenizer.decode(int(token_id)))

def _live_tps(first_token_at: Optional[float], token_count: int) -> float:
    if first_token_at is None or token_count <= 0:
        return 0.0
    return token_count / max(1e-9, time.monotonic() - first_token_at)

def _avg_tps_since_prefill(
    *,
    started_at: float,
    prefill_us: Optional[float],
    token_count: int,
) -> float:
    if token_count <= 0 or prefill_us is None:
        return 0.0
    generation_s = max(1e-9, time.monotonic() - started_at - (prefill_us / 1e6))
    return token_count / generation_s

def _sample_tps(
    *,
    now: float,
    first_token_at: Optional[float],
    token_count: int,
    last_status_at: Optional[float],
    last_status_tokens: int,
    previous_current_tps: float,
) -> float:
    if token_count <= 0:
        return 0.0
    if first_token_at is None:
        return 0.0
    if last_status_at is None or token_count <= last_status_tokens:
        return previous_current_tps if previous_current_tps > 0.0 else _live_tps(first_token_at, token_count)
    return (token_count - last_status_tokens) / max(1e-9, now - last_status_at)

def _print_header(mode: str) -> None:
    title = "MLX stock" if mode == "baseline" else "DFlash"
    if _is_tty():
        title_sgr = "1;34" if mode == "baseline" else "1;32"
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(f"{_colorize(_center_text(title), title_sgr)}\n")
        sys.stdout.write("\x1b[2K\n")
        sys.stdout.write("\x1b[2K\n")
    else:
        sys.stdout.write(f"{title}\n\n")
    sys.stdout.flush()

def _print_prompt_panel(
    prompt: str,
    *,
    max_tokens: int,
    info_line: str,
    enable_scroll_region: bool = True,
) -> None:
    global CONTENT_START_ROW
    if not _is_tty():
        sys.stdout.write(f"{info_line}\n")
        sys.stdout.write(f"Prompt (max tokens: {max_tokens}): {prompt}\n\n")
        sys.stdout.flush()
        return

    cols = max(40, min(_terminal_cols(), 120))
    inner_width = max(20, cols - 4)
    wrapped = textwrap.wrap(prompt, width=inner_width) or [""]
    info_wrapped = textwrap.wrap(info_line, width=inner_width) or [""]

    top_title = f" Prompt | max tokens: {max_tokens} "
    top_fill = max(0, inner_width - len(top_title))
    top = f"╭{top_title}{'─' * top_fill}╮"
    bottom = f"╰{'─' * inner_width}╯"

    sys.stdout.write(f"{_colorize(top, '1;35')}\n")
    for line in info_wrapped:
        sys.stdout.write(
            f"{_colorize('│', '1;35')} {_colorize(line.ljust(inner_width - 2), '1;90')} {_colorize('│', '1;35')}\n"
        )
    if info_wrapped:
        sys.stdout.write(
            f"{_colorize('│', '1;35')} {' ' * (inner_width - 2)} {_colorize('│', '1;35')}\n"
        )
    for line in wrapped:
        sys.stdout.write(
            f"{_colorize('│', '1;35')} {line.ljust(inner_width - 2)} {_colorize('│', '1;35')}\n"
        )
    sys.stdout.write(f"{_colorize(bottom, '1;35')}\n\n")

    rows = _terminal_rows()
    content_start_row = 4 + len(info_wrapped) + 1 + len(wrapped) + 3
    content_start_row = min(max(4, content_start_row), max(4, rows - 1))
    CONTENT_START_ROW = content_start_row
    if enable_scroll_region:
        sys.stdout.write(f"\x1b[{content_start_row};{rows}r")
    else:
        sys.stdout.write("\x1b[r")
    sys.stdout.write(f"\x1b[{content_start_row};1H")
    sys.stdout.flush()

def _update_baseline_status(
    *,
    started_at: float,
    prefill_us: Optional[float],
    first_token_at: Optional[float],
    token_count: int,
    current_tps: float,
) -> None:
    _write_status_line(
        f"cur {current_tps:.1f} tok/s | avg {_avg_tps_since_prefill(started_at=started_at, prefill_us=prefill_us, token_count=token_count):.1f} tok/s | {token_count} tokens"
    )

def _update_dflash_status(
    *,
    started_at: float,
    prefill_us: Optional[float],
    first_token_at: Optional[float],
    token_count: int,
    acceptance_ratio: Optional[float],
    current_tps: float,
) -> None:
    accept_text = f" | accept: {100.0 * acceptance_ratio:.1f}%" if acceptance_ratio is not None else ""
    _write_status_line(
        f"cur {current_tps:.1f} tok/s | avg {_avg_tps_since_prefill(started_at=started_at, prefill_us=prefill_us, token_count=token_count):.1f} tok/s | {token_count} tokens{accept_text}"
    )

def run_baseline(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    info_line: str,
    use_chat_template: bool,
    quantize_kv_cache: bool,
    no_eos: bool,
) -> int:
    eos_ids = _get_stop_token_ids(tokenizer)
    started_at = time.monotonic()
    prompt_tokens = _prepare_prompt_tokens(
        tokenizer,
        prompt,
        use_chat_template=use_chat_template,
    )

    first_token_at: Optional[float] = None
    prefill_us: Optional[float] = None
    token_count = 0
    final_tps = 0.0
    current_tps = 0.0
    last_status_at: Optional[float] = None
    last_status_tokens = 0

    for response in mlx_stream_generate(
        target_model,
        tokenizer,
        prompt_tokens,
        max_tokens=max_tokens,
    ):
        if first_token_at is None:
            first_token_at = time.monotonic()
            prefill_us = (first_token_at - started_at) * 1e6
        token_text = response.text
        sys.stdout.write(token_text)
        sys.stdout.flush()
        token_count = int(response.generation_tokens)
        final_tps = float(response.generation_tps)
        if token_count % 8 == 0:
            now = time.monotonic()
            current_tps = _sample_tps(
                now=now,
                first_token_at=first_token_at,
                token_count=token_count,
                last_status_at=last_status_at,
                last_status_tokens=last_status_tokens,
                previous_current_tps=current_tps,
            )
            _update_baseline_status(
                started_at=started_at,
                prefill_us=prefill_us,
                first_token_at=first_token_at,
                token_count=token_count,
                current_tps=current_tps,
            )
            last_status_at = now
            last_status_tokens = token_count

    if token_count > 0:
        current_tps = _sample_tps(
            now=time.monotonic(),
            first_token_at=first_token_at,
            token_count=token_count,
            last_status_at=last_status_at,
            last_status_tokens=last_status_tokens,
            previous_current_tps=current_tps,
        )
    _update_baseline_status(
        started_at=started_at,
        prefill_us=prefill_us,
        first_token_at=first_token_at,
        token_count=token_count,
        current_tps=current_tps,
    )
    total_elapsed_s = max(0.0, time.monotonic() - started_at)
    footer = f"avg {final_tps:.1f} tok/s | {token_count}/{max_tokens} tokens | total {total_elapsed_s:.1f}s"
    _finalize_output(
        mode="baseline",
        prompt=prompt,
        max_tokens=max_tokens,
        info_line=info_line,
        footer=footer,
        footer_sgr="1;34",
    )
    return 0

def run_dflash(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: Any,
    prompt: str,
    max_tokens: int,
    info_line: str,
    block_tokens: int,
    use_chat_template: bool,
    quantize_kv_cache: bool,
    no_eos: bool,
) -> int:
    eos_ids = _get_stop_token_ids(tokenizer)
    started_at = time.monotonic()

    first_token_at: Optional[float] = None
    prefill_us: Optional[float] = None
    token_count = 0
    final_tps = 0.0
    acceptance_ratio: Optional[float] = None
    current_tps = 0.0
    last_status_at: Optional[float] = None
    last_status_tokens = 0

    for event in stream_dflash_generate(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=use_chat_template,
        block_tokens=block_tokens,
        stop_token_ids=[] if no_eos else eos_ids,
        suppress_token_ids=eos_ids if no_eos else None,
        quantize_kv_cache=quantize_kv_cache,
    ):
        if event["event"] == "prefill":
            prefill_us = float(event["prefill_us"])
            continue
        if event["event"] == "token":
            if first_token_at is None:
                first_token_at = time.monotonic()
            token_text = _maybe_decode_token(tokenizer, int(event["token_id"]))
            sys.stdout.write(token_text)
            sys.stdout.flush()
            token_count = int(event["generated_tokens"])
            acceptance_ratio = float(event["acceptance_ratio"])
            if token_count % 8 == 0:
                now = time.monotonic()
                current_tps = _sample_tps(
                    now=now,
                    first_token_at=first_token_at,
                    token_count=token_count,
                    last_status_at=last_status_at,
                    last_status_tokens=last_status_tokens,
                    previous_current_tps=current_tps,
                )
                _update_dflash_status(
                    started_at=started_at,
                    prefill_us=prefill_us,
                    first_token_at=first_token_at,
                    token_count=token_count,
                    acceptance_ratio=acceptance_ratio,
                    current_tps=current_tps,
                )
                last_status_at = now
                last_status_tokens = token_count
        elif event["event"] == "summary":
            acceptance_ratio = float(event["acceptance_ratio"])
            summary_prefill_us = float(event["phase_timings_us"]["prefill"])
            generation_us = max(0.0, float(event["elapsed_us"]) - summary_prefill_us)
            final_tps = token_count / (generation_us / 1e6) if generation_us > 0.0 else 0.0
    if token_count > 0:
        current_tps = _sample_tps(
            now=time.monotonic(),
            first_token_at=first_token_at,
            token_count=token_count,
            last_status_at=last_status_at,
            last_status_tokens=last_status_tokens,
            previous_current_tps=current_tps,
        )
    _update_dflash_status(
        started_at=started_at,
        prefill_us=prefill_us,
        first_token_at=first_token_at,
        token_count=token_count,
        acceptance_ratio=acceptance_ratio,
        current_tps=current_tps,
    )
    total_elapsed_s = max(0.0, time.monotonic() - started_at)
    footer = f"avg {final_tps:.1f} tok/s | {token_count}/{max_tokens} tokens | total {total_elapsed_s:.1f}s"
    if acceptance_ratio is not None:
        footer += f" | accept: {100.0 * acceptance_ratio:.1f}%"
    _finalize_output(
        mode="dflash",
        prompt=prompt,
        max_tokens=max_tokens,
        info_line=info_line,
        footer=footer,
        footer_sgr="1;32",
    )
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal capture-friendly DFlash demo without Rich.")
    parser.add_argument("--mode", choices=("baseline", "dflash"), required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--quantize-kv-cache", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
    return parser.parse_args()

def main() -> None:
    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_cache_limit(wired_limit // 4)
    args = parse_args()
    use_chat_template = not args.no_chat_template
    _print_header(args.mode)
    resolved_target_ref = resolve_model_ref(args.target_model, kind="target")
    info_line = f"{_display_target_label(resolved_target_ref)} | {_machine_label()}"
    _print_prompt_panel(args.prompt, max_tokens=args.max_tokens, info_line=info_line)

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    with _suppress_load_noise():
        if args.mode == "baseline":
            target_model, tokenizer, _ = load_pristine_target(
                resolved_target_ref,
                lazy=True,
                return_config=True,
            )
        else:
            target_model, tokenizer, _ = load_target_bundle(
                resolved_target_ref,
                lazy=True,
                split_full_attention_sdpa=True,
                quantize_kv_cache=args.quantize_kv_cache,
            )

    if args.mode == "baseline":
        raise SystemExit(
            run_baseline(
                target_model=target_model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                info_line=info_line,
                use_chat_template=use_chat_template,
                quantize_kv_cache=args.quantize_kv_cache,
                no_eos=args.no_eos,
            )
        )

    with _suppress_load_noise():
        draft_model, _ = load_draft_bundle(
            resolve_model_ref(args.draft_model, kind="draft"),
            lazy=True,
        )
    raise SystemExit(
        run_dflash(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            info_line=info_line,
            block_tokens=args.block_tokens,
            use_chat_template=use_chat_template,
            quantize_kv_cache=args.quantize_kv_cache,
            no_eos=args.no_eos,
        )
    )

if __name__ == "__main__":
    main()

# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import json
import sys
import os
import logging
import time
import warnings
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any, Optional

warnings.filterwarnings("ignore", message="mlx_lm.server is not recommended")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
try:
    from huggingface_hub.utils import disable_progress_bars
except Exception:
    disable_progress_bars = None

if disable_progress_bars is not None:
    try:
        disable_progress_bars()
    except Exception:
        pass

import mlx.core as mx
import mlx_lm.server as mlx_server

from dflash_mlx.generate import (
    get_stop_token_ids,
    load_runtime_components,
    resolve_optional_draft_ref,
)
from dflash_mlx.runtime import stream_dflash_generate


_STATEFUL_SERVER_API = "state" in getattr(mlx_server.Response, "__annotations__", {})


def _read_project_version() -> str:
    try:
        return package_version("dflash-mlx")
    except PackageNotFoundError:
        return "unknown"


class DFlashModelProvider(mlx_server.ModelProvider):
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        requested_model = self.default_model_map.get(model_path, model_path)
        if self.cli_args.model is not None:
            model_ref = self.cli_args.model
        elif requested_model == "default_model":
            raise ValueError(
                "A model path has to be given as a CLI argument or in the HTTP request"
            )
        else:
            model_ref = requested_model

        if draft_model_path == "default_model":
            draft_ref = self.cli_args.draft_model
        elif draft_model_path is not None:
            draft_ref = draft_model_path
        else:
            draft_ref = None
        resolved_draft_ref = resolve_optional_draft_ref(model_ref, draft_ref)

        if self.model_key == (model_ref, None, resolved_draft_ref):
            return self.model, self.tokenizer

        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None

        model, tokenizer, draft_model, resolved_draft_ref = load_runtime_components(
            model_ref=model_ref,
            draft_ref=draft_ref,
        )

        if self.cli_args.chat_template:
            tokenizer.chat_template = self.cli_args.chat_template
        if self.cli_args.use_default_chat_template and tokenizer.chat_template is None:
            tokenizer.chat_template = tokenizer.default_chat_template

        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.model_key = (model_ref, None, resolved_draft_ref)
        self.is_batchable = False
        return self.model, self.tokenizer


class DFlashResponseGenerator(mlx_server.ResponseGenerator):
    @staticmethod
    def _build_generation_context(tokenizer, prompt, stop_words=None, sequences=None):
        if _STATEFUL_SERVER_API:
            return mlx_server.GenerationContext(
                has_thinking=tokenizer.has_thinking,
                has_tool_calling=tokenizer.has_tool_calling,
                tool_parser=tokenizer.tool_parser,
                sequences=sequences or {},
                prompt=prompt,
            )
        return mlx_server.GenerationContext(
            has_tool_calling=tokenizer.has_tool_calling,
            tool_call_start=tokenizer.tool_call_start,
            tool_call_end=tokenizer.tool_call_end,
            tool_parser=tokenizer.tool_parser,
            has_thinking=tokenizer.has_thinking,
            think_start_id=tokenizer.think_start_id,
            think_end=tokenizer.think_end,
            think_end_id=tokenizer.think_end_id,
            eos_token_ids=tokenizer.eos_token_ids,
            stop_token_sequences=[
                tokenizer.encode(stop_word, add_special_tokens=False)
                for stop_word in (stop_words or [])
            ],
            prompt=prompt,
        )

    @staticmethod
    def _make_response(
        *,
        text: str,
        token: int,
        state: Optional[str],
        match: Optional[tuple[int, ...]],
        finish_reason: Optional[str],
    ):
        if _STATEFUL_SERVER_API:
            return mlx_server.Response(
                text,
                token,
                state or "normal",
                match,
                0.0,
                finish_reason,
                (),
            )
        return mlx_server.Response(
            text,
            token,
            0.0,
            finish_reason,
            (),
        )

    def _serve_single(self, request):
        request_tuple = request
        rqueue, request, args = request_tuple

        if args.max_tokens <= 256:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] fast-path AR | max_tokens={args.max_tokens}\n"
            )
            sys.stderr.flush()
            saved_draft_model = self.model_provider.draft_model
            try:
                self.model_provider.draft_model = None
                return super()._serve_single((rqueue, request, args))
            finally:
                self.model_provider.draft_model = saved_draft_model

        try:
            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            draft_model = self.model_provider.draft_model
            tokenized = self._tokenize(tokenizer, request, args)
            if isinstance(tokenized, tuple):
                prompt, _, _, initial_state = tokenized
            else:
                prompt = tokenized
                initial_state = "normal"

            sm = None
            sm_state = None
            sequences = {}
            if _STATEFUL_SERVER_API and hasattr(self, "_make_state_machine"):
                sm, sequences = self._make_state_machine(
                    self.model_provider.model_key,
                    tokenizer,
                    args.stop_words,
                    initial_state=initial_state,
                )
                sm_state = sm.make_state()

            ctx = self._build_generation_context(
                tokenizer,
                prompt,
                stop_words=args.stop_words,
                sequences=sequences,
            )
            rqueue.put(ctx)

            if args.seed is not None:
                mx.random.seed(args.seed)

            stop_token_ids = get_stop_token_ids(tokenizer)
            detokenizer = tokenizer.detokenizer
            if hasattr(detokenizer, "reset"):
                detokenizer.reset()
            eos_token_ids = set(int(token_id) for token_id in tokenizer.eos_token_ids)
            pending_token: Optional[int] = None
            pending_text = ""
            pending_state: Optional[str] = "normal"
            pending_match: Optional[tuple[int, ...]] = None
            pending_finish_reason: Optional[str] = None
            first_token_flushed = False
            finish_reason: Optional[str] = None
            summary_event: Optional[dict[str, Any]] = None
            request_start_ns = time.perf_counter_ns()
            prefill_elapsed_s = 0.0
            live_tok_s = 0.0
            live_token_count = 0
            live_acceptance_pct = 0.0
            live_prompt_len = len(prompt)
            printed_prefill_progress = False

            event_iter = stream_dflash_generate(
                target_model=model,
                tokenizer=tokenizer,
                draft_model=draft_model,
                prompt="",
                max_new_tokens=args.max_tokens,
                use_chat_template=False,
                stop_token_ids=stop_token_ids,
                prompt_tokens_override=prompt,
            )

            try:
                for event in event_iter:
                    if event.get("event") in ("prefill", "prefill_progress"):
                        processed = int(
                            event.get(
                                "tokens_processed",
                                event.get("prompt_token_count", len(prompt)),
                            )
                        )
                        total = int(
                            event.get(
                                "tokens_total",
                                event.get("prompt_token_count", len(prompt)),
                            )
                        )
                        elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
                        if event.get("event") == "prefill_progress":
                            sys.stderr.write(
                                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefill: {processed}/{total} tokens | {elapsed_s:.1f}s\n"
                            )
                            sys.stderr.flush()
                            printed_prefill_progress = True
                        else:
                            prefill_elapsed_s = elapsed_s
                            if not printed_prefill_progress:
                                sys.stderr.write(
                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefill: {processed}/{total} tokens | {elapsed_s:.1f}s\n"
                                )
                                sys.stderr.flush()
                        continue
                    if event.get("event") != "token":
                        if event.get("event") == "summary":
                            summary_event = event
                            generated_token_ids = list(event.get("generated_token_ids", []) or [])
                            if generated_token_ids:
                                last_token = int(generated_token_ids[-1])
                                if last_token in eos_token_ids:
                                    finish_reason = "stop"
                                elif int(event.get("generation_tokens", 0)) >= int(args.max_tokens):
                                    finish_reason = "length"
                                else:
                                    finish_reason = "stop"
                            else:
                                finish_reason = "stop"
                        continue

                    token = int(event["token_id"])
                    live_token_count += 1
                    live_acceptance_pct = float(event.get("acceptance_ratio", 0.0) or 0.0) * 100.0
                    elapsed_s = (time.perf_counter_ns() - request_start_ns) / 1e9
                    live_tok_s = live_token_count / max(0.001, elapsed_s - prefill_elapsed_s)
                    if live_token_count % 2048 == 0:
                        sys.stderr.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] {live_tok_s:.1f} tok/s | {live_acceptance_pct:.1f}% accepted | "
                            f"{live_token_count} tokens | {elapsed_s:.1f}s | "
                            f"prompt: {live_prompt_len} tokens\n"
                        )
                        sys.stderr.flush()
                    current_state = "normal"
                    match_sequence: Optional[tuple[int, ...]] = None
                    token_finish_reason: Optional[str] = None
                    if sm is not None:
                        sm_state, match_sequence, current_state = sm.match(sm_state, token)
                        if match_sequence is not None and current_state is None:
                            token_finish_reason = "stop"

                    text = ""
                    if token not in eos_token_ids:
                        detokenizer.add_token(token)
                        text = detokenizer.last_segment

                    if not first_token_flushed:
                        immediate_finish_reason = token_finish_reason
                        if immediate_finish_reason is None:
                            if token in eos_token_ids:
                                immediate_finish_reason = "stop"
                            elif live_token_count >= int(args.max_tokens):
                                immediate_finish_reason = "length"
                        rqueue.put(
                            self._make_response(
                                text=text,
                                token=token,
                                state=current_state or "normal",
                                match=match_sequence,
                                finish_reason=immediate_finish_reason,
                            )
                        )
                        first_token_flushed = True
                        if ctx._should_stop:
                            break
                        continue

                    if pending_token is not None:
                        rqueue.put(
                            self._make_response(
                                text=pending_text,
                                token=pending_token,
                                state=pending_state,
                                match=pending_match,
                                finish_reason=pending_finish_reason,
                            )
                        )

                    pending_token = token
                    pending_text = text
                    pending_state = current_state or "normal"
                    pending_match = match_sequence
                    pending_finish_reason = token_finish_reason

                    if ctx._should_stop:
                        break
            finally:
                event_iter.close()

            detokenizer.finalize()
            tail = detokenizer.last_segment
            if pending_token is not None:
                rqueue.put(
                    self._make_response(
                        text=pending_text + tail,
                        token=pending_token,
                        state=pending_state,
                        match=pending_match,
                        finish_reason=finish_reason or pending_finish_reason,
                    )
                )

            if summary_event is not None:
                generation_tokens = int(summary_event.get("generation_tokens", 0) or 0)
                elapsed_us = float(summary_event.get("elapsed_us", 0.0) or 0.0)
                phase_timings_us = dict(summary_event.get("phase_timings_us") or {})
                prefill_us = float(phase_timings_us.get("prefill", 0.0) or 0.0)
                decode_s = max(0.0, (elapsed_us - prefill_us) / 1_000_000.0)
                tok_s = (generation_tokens / decode_s) if decode_s > 0.0 else 0.0
                acceptance_pct = float(summary_event.get("acceptance_ratio", 0.0) or 0.0) * 100.0
                total_s = elapsed_us / 1_000_000.0
                sys.stderr.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] {tok_s:.1f} tok/s | {acceptance_pct:.1f}% accepted | "
                    f"{generation_tokens} tokens | {total_s:.1f}s | "
                    f"prompt: {len(prompt)} tokens\n"
                )
                sys.stderr.flush()

            rqueue.put(None)
        except Exception as e:
            rqueue.put(e)


class DFlashAPIHandler(mlx_server.APIHandler):
    def handle_completion(self, request, stop_words):
        try:
            return super().handle_completion(request, stop_words)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            return
        except ValueError as e:
            logging.warning("Tool parser error (likely malformed tool call): %s", e)
            self.close_connection = True
            return

    def generate_response(self, *args, **kwargs):
        response = super().generate_response(*args, **kwargs)
        served_model = (
            self.response_generator.model_provider.model_key[0]
            if self.response_generator.model_provider.model_key is not None
            else None
        )
        if served_model:
            response["model"] = served_model
        return response


def _print_startup_banner(
    *,
    port: int,
    model_provider: DFlashModelProvider,
) -> None:
    dflash_version = _read_project_version()
    server_name = getattr(mlx_server, "__name__", "mlx_lm.server")
    target_ref = None
    draft_ref = None
    if model_provider.model_key is not None:
        target_ref = model_provider.model_key[0]
        draft_ref = model_provider.model_key[2]
    target_ref = target_ref or model_provider.cli_args.model or "unknown"
    if not draft_ref:
        raise RuntimeError("DFlash server requires a resolved draft model before startup.")

    if model_provider.cli_args.draft_model:
        draft_suffix = " (explicit)"
    else:
        draft_suffix = " (auto-detected)"
    raw_lines = [
        f"DFlash v{dflash_version} - speculative decoding engine",
        f"Target: {target_ref}",
        f"Draft:  {draft_ref}{draft_suffix}",
        "Mode:   DFlash (speculative decoding active)",
        f"Server: {server_name} on port {port}",
    ]

    width = max(len(line) for line in raw_lines)
    use_color = sys.stderr.isatty()
    reset = "\033[0m" if use_color else ""
    border_color = "\033[38;5;39m" if use_color else ""
    title_color = "\033[1;38;5;51m" if use_color else ""
    body_color = "\033[38;5;252m" if use_color else ""

    def style(text: str, color: str) -> str:
        return f"{color}{text}{reset}" if use_color else text

    border = style("+" + "-" * (width + 2) + "+", border_color)
    lines = [border]
    for index, raw_line in enumerate(raw_lines):
        padded = f"| {raw_line.ljust(width)} |"
        lines.append(style(padded, title_color if index == 0 else body_color))
    lines.append(border)

    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()


def _run_with_dflash_server(host: str, port: int, model_provider: DFlashModelProvider):
    group = mx.distributed.init()
    prompt_cache = mlx_server.LRUPromptCache(model_provider.cli_args.prompt_cache_size)
    response_generator = DFlashResponseGenerator(model_provider, prompt_cache)
    if group.rank() == 0:
        _print_startup_banner(port=port, model_provider=model_provider)
        mlx_server._run_http_server(
            host,
            port,
            response_generator,
            handler_class=DFlashAPIHandler,
        )
    else:
        response_generator.join()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DFlash Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path or Hugging Face reference for the target model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the HTTP server (default: 8000)",
    )
    parser.add_argument(
        "--allowed-origins",
        type=lambda x: x.split(","),
        default="*",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--draft-model",
        "--draft",
        dest="draft_model",
        type=str,
        default=None,
        help="Optional DFlash draft model override.",
    )
    parser.add_argument(
        "--dflash-max-ctx",
        type=int,
        default=None,
        help="Maximum prompt token count for DFlash speculative decoding.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help=argparse.SUPPRESS,
        default=3,
    )
    parser.add_argument(
        "--quantize-draft",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        default="{}",
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
    )
    parser.add_argument(
        "--decode-concurrency",
        type=int,
        default=32,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prompt-concurrency",
        type=int,
        default=8,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=10,
        help="Maximum number of distinct KV caches to hold in the prompt cache",
    )
    parser.add_argument(
        "--prompt-cache-bytes",
        type=int,
        help="Maximum size in bytes of the KV caches",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.dflash_max_ctx is not None:
        if args.dflash_max_ctx <= 0:
            raise SystemExit("--dflash-max-ctx must be > 0")
        os.environ["DFLASH_MAX_CTX"] = str(args.dflash_max_ctx)

    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)
        mx.set_cache_limit(wired_limit // 4)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    _run_with_dflash_server(args.host, args.port, DFlashModelProvider(args))


if __name__ == "__main__":
    main()

# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from collections.abc import Sequence

warnings.filterwarnings("ignore", message="mlx_lm.server is not recommended")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
try:
    from huggingface_hub.utils import disable_progress_bars
except ImportError:
    disable_progress_bars = None

if disable_progress_bars is not None:
    try:
        disable_progress_bars()
    except Exception as exc:
        sys.stderr.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] huggingface progress bar disable failed: {exc}\n"
        )
        sys.stderr.flush()

import mlx_lm.server as mlx_server

from dflash_mlx import __version__ as _DFLASH_VERSION
from dflash_mlx.server.config import (
    build_parser as _build_parser,
    configure_logging,
    configure_metal_limits,
    normalize_cli_args,
)
from dflash_mlx.server.protocol import (
    STATEFUL_SERVER_API as _STATEFUL_SERVER_API,
    make_state_machine as _make_state_machine,
    thinking_enabled_for_request as _thinking_enabled_for_request,
)
from dflash_mlx.server.metrics import (
    get_live_metrics_payload as _get_live_metrics_payload,
)
from dflash_mlx.server.model_provider import DFlashModelProvider
from dflash_mlx.server.runtime import PreparedDFlashRequest, ServerRuntime
from dflash_mlx.runtime.context import with_metal_limits
from dflash_mlx.server.responses_adapter import (
    ResponsesAdapterError,
    chat_response_to_responses,
    responses_to_chat_body,
)
from dflash_mlx.server.tool_calls import (
    ToolCallParseError,
    apply_tool_choice,
    normalize_parallel_tool_calls,
)

def _read_project_version() -> str:
    return _DFLASH_VERSION

class DFlashResponseGenerator(mlx_server.ResponseGenerator):
    def __init__(self, model_provider, prompt_cache, server_runtime: ServerRuntime):
        super().__init__(model_provider, prompt_cache)
        self.server_runtime = server_runtime
        self.last_finish_reason = None
        self.last_prompt_token_count = None
        self.last_completion_token_count = None

    def generate(self, request, generation_args, progress_callback=None):
        self.last_finish_reason = None
        self.last_prompt_token_count = None
        self.last_completion_token_count = None
        ctx, response = super().generate(
            request,
            generation_args,
            progress_callback=progress_callback,
        )

        def _track(inner):
            for gen in inner:
                reason = getattr(gen, "finish_reason", None)
                if reason is not None:
                    self.last_finish_reason = reason
                prompt_tokens = getattr(gen, "prompt_tokens", None)
                if prompt_tokens is not None:
                    self.last_prompt_token_count = int(prompt_tokens)
                generation_tokens = getattr(gen, "generation_tokens", None)
                if generation_tokens is not None:
                    self.last_completion_token_count = int(generation_tokens)
                yield gen

        return ctx, _track(response)

    def _restore_thinking_enabled(self, had_previous: bool, previous: object) -> None:
        if had_previous:
            self._current_thinking_enabled = previous
        elif hasattr(self, "_current_thinking_enabled"):
            delattr(self, "_current_thinking_enabled")

    def _tokenize(self, tokenizer, request, args):
        tokenized = super()._tokenize(tokenizer, request, args)
        if (
            not bool(getattr(self, "_current_thinking_enabled", True))
            and isinstance(tokenized, tuple)
            and len(tokenized) == 4
        ):
            prompt, segments, segment_types, _initial_state = tokenized
            return prompt, segments, segment_types, "normal"
        return tokenized

    def _make_state_machine(
        self,
        model_key,
        tokenizer,
        stop_words,
        initial_state="normal",
    ):
        include_thinking = bool(getattr(self, "_current_thinking_enabled", True))
        cache_key = (model_key, tuple(stop_words), initial_state, include_thinking)
        cached = self._state_machine_cache.get(cache_key)
        if cached is not None:
            return cached

        sm, sequences = _make_state_machine(
            tokenizer=tokenizer,
            stop_words=stop_words,
            initial_state=initial_state,
            include_thinking=include_thinking,
        )
        if len(self._state_machine_cache) > 100:
            self._state_machine_cache.clear()
        self._state_machine_cache[cache_key] = (sm, sequences)
        return sm, sequences

    def _prepare_dflash_request(
        self,
        tokenizer,
        request,
        args,
        *,
        has_thinking: bool,
    ) -> PreparedDFlashRequest:
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

        return PreparedDFlashRequest(
            prompt=prompt,
            sequences=sequences,
            state_machine=sm,
            state_machine_state=sm_state,
            has_thinking=has_thinking,
        )

    def _serve_single(self, request):
        request_tuple = request
        rqueue, request, args = request_tuple

        request_id = self.server_runtime.next_request_id()
        had_previous_thinking_enabled = hasattr(self, "_current_thinking_enabled")
        previous_thinking_enabled = getattr(self, "_current_thinking_enabled", None)
        self._current_thinking_enabled = _thinking_enabled_for_request(
            self.model_provider.cli_args,
            args,
        )
        try:
            fastpath_max_tokens = int(
                getattr(self.model_provider.cli_args, "fastpath_max_tokens", 0) or 0
            )

            if fastpath_max_tokens > 0 and args.max_tokens <= fastpath_max_tokens:
                sys.stderr.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] fast-path AR | "
                    f"max_tokens={args.max_tokens} threshold={fastpath_max_tokens}\n"
                )
                sys.stderr.flush()
                saved_draft_model = self.model_provider.draft_model
                wall_t0 = time.perf_counter_ns()
                completed = False
                self.server_runtime.start_target_only_request(
                    request_id=request_id,
                    mode_used="ar_fastpath",
                    max_tokens=int(args.max_tokens),
                )
                try:
                    self.model_provider.draft_model = None
                    result = super()._serve_single((rqueue, request, args))
                    completed = True
                    return result
                finally:
                    self.model_provider.draft_model = saved_draft_model
                    if completed:
                        wall_ms = (time.perf_counter_ns() - wall_t0) / 1e6
                        self.server_runtime.record_target_only_request(
                            request_id=request_id,
                            mode_used="ar_fastpath",
                            wall_ms=wall_ms,
                            max_tokens=int(args.max_tokens),
                        )

            has_thinking = bool(self._current_thinking_enabled)
            prepared = self._prepare_dflash_request(
                self.model_provider.tokenizer,
                request,
                args,
                has_thinking=has_thinking,
            )
            self.server_runtime.serve_dflash_request(
                request_id=request_id,
                rqueue=rqueue,
                request=request,
                args=args,
                prepared=prepared,
            )
        except Exception as e:
            self.server_runtime.clear_request(request_id=request_id)
            rqueue.put(e)
        finally:
            self._restore_thinking_enabled(
                had_previous_thinking_enabled,
                previous_thinking_enabled,
            )

class DFlashAPIHandler(mlx_server.APIHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] == "/metrics":
            self.handle_metrics_request()
            return
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] == "/v1/responses":
            self.handle_responses_request()
            return
        return super().do_POST()

    def handle_metrics_request(self):
        payload = _get_live_metrics_payload()
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        self._set_completion_headers(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def handle_responses_request(self):
        try:
            body = self._read_json_body()
            chat_body = responses_to_chat_body(body)
            self._serve_responses_as_chat(chat_body)
        except ResponsesAdapterError as e:
            self._write_json_response(400, {"error": str(e)})

    def _serve_responses_as_chat(self, chat_body: dict[str, object]) -> None:
        previous_body = getattr(self, "body", None)
        previous_path = self.path
        previous_responses_mode = getattr(self, "_responses_mode", None)
        self.body = chat_body
        self.path = "/v1/chat/completions"
        self._responses_mode = True
        try:
            stop_words = self._load_completion_fields(chat_body)
            request = self.handle_chat_completions()
            self.handle_completion(request, stop_words)
        except ValueError as e:
            self._write_json_response(400, {"error": str(e)})
        finally:
            self.body = previous_body
            self.path = previous_path
            if previous_responses_mode is None:
                if hasattr(self, "_responses_mode"):
                    delattr(self, "_responses_mode")
            else:
                self._responses_mode = previous_responses_mode

    def _read_json_body(self) -> dict[str, object]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ResponsesAdapterError("Content-Length header is required")
        try:
            length = int(content_length)
        except ValueError as exc:
            raise ResponsesAdapterError("Invalid Content-Length header") from exc
        try:
            body = json.loads(self.rfile.read(length).decode())
        except json.JSONDecodeError as exc:
            raise ResponsesAdapterError(f"Invalid JSON in request body: {exc}") from exc
        if not isinstance(body, dict):
            raise ResponsesAdapterError("Request should be a JSON dictionary")
        return body

    def handle_chat_completions(self):
        self.tool_choice = self.body.get("tool_choice")
        self.parallel_tool_calls = normalize_parallel_tool_calls(
            self.body.get("parallel_tool_calls")
        )
        return super().handle_chat_completions()

    def handle_text_completions(self):
        self.tool_choice = None
        self.parallel_tool_calls = True
        return super().handle_text_completions()

    def _load_completion_fields(self, body: dict[str, object]) -> list[str]:
        self.stream = bool(body.get("stream", False))
        self.stream_options = body.get("stream_options", None)
        self.requested_model = body.get("model", "default_model")
        self.requested_draft_model = body.get("draft_model", "default_model")
        self.num_draft_tokens = body.get(
            "num_draft_tokens",
            self.response_generator.cli_args.num_draft_tokens,
        )
        self.adapter = body.get("adapters", None)
        self.max_tokens = body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = body.get(
                "max_tokens",
                self.response_generator.cli_args.max_tokens,
            )
        self.temperature = body.get(
            "temperature",
            self.response_generator.cli_args.temp,
        )
        self.top_p = body.get("top_p", self.response_generator.cli_args.top_p)
        self.top_k = body.get("top_k", self.response_generator.cli_args.top_k)
        self.min_p = body.get("min_p", self.response_generator.cli_args.min_p)
        self.repetition_penalty = body.get("repetition_penalty", 0.0)
        self.repetition_context_size = body.get("repetition_context_size", 20)
        self.presence_penalty = body.get("presence_penalty", 0.0)
        self.presence_context_size = body.get("presence_context_size", 20)
        self.frequency_penalty = body.get("frequency_penalty", 0.0)
        self.frequency_context_size = body.get("frequency_context_size", 20)
        self.xtc_probability = body.get("xtc_probability", 0.0)
        self.xtc_threshold = body.get("xtc_threshold", 0.0)
        self.logit_bias = body.get("logit_bias", None)
        self.logprobs = body.get("logprobs", False)
        self.top_logprobs = body.get("top_logprobs", -1)
        self.seed = body.get("seed", None)
        self.chat_template_kwargs = body.get("chat_template_kwargs")
        self.tool_choice = body.get("tool_choice")
        self.parallel_tool_calls = normalize_parallel_tool_calls(
            body.get("parallel_tool_calls")
        )
        self.validate_model_parameters()

        stop_words = body.get("stop") or []
        if isinstance(stop_words, str):
            return [stop_words]
        if not isinstance(stop_words, list):
            raise ValueError("stop must be a string or list")
        return stop_words

    def _write_json_response(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self._set_completion_headers(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _write_completion_error_response(self, status: int, message: str) -> None:
        self._write_json_response(
            status,
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    def _set_stream_headers(self, status_code: int = 200):
        super()._set_stream_headers(status_code)
        if status_code == 200:
            self._completion_response_started = True

    def _set_completion_headers(self, status_code: int = 200):
        super()._set_completion_headers(status_code)
        if status_code == 200:
            self._completion_response_started = True

    def handle_completion(self, request, stop_words):
        try:
            apply_tool_choice(request, getattr(self, "tool_choice", None))
            if not bool(getattr(self, "parallel_tool_calls", True)):
                raise ValueError("parallel_tool_calls=false is not supported")
        except ValueError as e:
            logging.warning("Rejected tool-call request: %s", e)
            self._write_completion_error_response(400, str(e))
            return
        self._completion_response_started = False
        try:
            return super().handle_completion(request, stop_words)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            return
        except ToolCallParseError as e:
            logging.error("Tool parser error: %s", e)
            if getattr(self, "_completion_response_started", False):
                # Headers (and possibly stream chunks) are already on the
                # wire; a late 400 would just drop the stream without a
                # finish_reason. Close the response cleanly instead.
                self._finalize_completion_after_tool_parse_error()
            else:
                self._write_completion_error_response(400, str(e))
            return

    def _finalize_completion_after_tool_parse_error(self) -> None:
        finish_reason = (
            getattr(self.response_generator, "last_finish_reason", None) or "length"
        )
        logging.error(
            "Closing in-flight completion with finish_reason=%s after tool parse "
            "failure (tool call likely truncated by max_tokens)",
            finish_reason,
        )
        response = self.generate_response(
            "",
            finish_reason,
            prompt_token_count=int(
                getattr(self.response_generator, "last_prompt_token_count", None) or 0
            ),
            completion_token_count=int(
                getattr(self.response_generator, "last_completion_token_count", None)
                or 0
            ),
        )
        payload = json.dumps(response)
        try:
            if self.stream:
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.write("data: [DONE]\n\n".encode())
            else:
                self.wfile.write(payload.encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    def generate_response(self, *args, **kwargs):
        response = super().generate_response(*args, **kwargs)
        _normalize_reasoning_content(response)
        served_model = (
            self.response_generator.model_provider.model_key[0]
            if self.response_generator.model_provider.model_key is not None
            else None
        )
        if served_model:
            response["model"] = served_model
        if getattr(self, "_responses_mode", False):
            return chat_response_to_responses(response)
        return response

def _normalize_reasoning_content(response: dict[str, object]) -> None:
    choices = response.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        holder = choice.get("delta") if "delta" in choice else choice.get("message")
        if not isinstance(holder, dict) or "reasoning" not in holder:
            continue
        if "reasoning_content" not in holder:
            holder["reasoning_content"] = holder["reasoning"]
        del holder["reasoning"]

def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = _build_parser()
    if prog is not None:
        parser.prog = prog
    args = normalize_cli_args(parser.parse_args(list(argv) if argv is not None else None))
    metal_limits = configure_metal_limits(args)
    args.runtime_context = with_metal_limits(args.runtime_context, metal_limits)
    configure_logging(args.log_level)
    ServerRuntime(
        host=args.host,
        port=args.port,
        model_provider=DFlashModelProvider(args),
        version=_read_project_version(),
    ).serve_forever(
        response_generator_factory=DFlashResponseGenerator,
        handler_class=DFlashAPIHandler,
    )

if __name__ == "__main__":
    main()

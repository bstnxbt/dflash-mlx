# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import itertools
import logging
import sys
import time
import warnings
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version as package_version

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

from dflash_mlx.server.config import (
    build_parser as _build_parser,
    configure_logging,
    configure_metal_limits,
    normalize_cli_args,
)
from dflash_mlx.server.protocol import (
    STATEFUL_SERVER_API as _STATEFUL_SERVER_API,
    build_generation_context as _build_generation_context,
    match_stream_token as _match_stream_token,
)
from dflash_mlx.bench_logger import (
    enabled as _bench_enabled,
    log_post as _bench_log_post,
)
from dflash_mlx.server.metrics import (
    log_bench_post as _log_bench_post,
    write_summary_line as _write_summary_line,
)
from dflash_mlx.server.model_provider import (
    DFlashModelProvider,
    wait_for_initial_model_load as _wait_for_initial_model_load,
)
from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
from dflash_mlx.runtime_context import with_metal_limits
from dflash_mlx.server.prefix_cache_flow import (
    PrefixCacheFlow,
    get_dflash_prefix_cache as _get_dflash_prefix_cache,
    log_prefix_cache_stats,
    shutdown_dflash_prefix_cache,
)
from dflash_mlx.server.prefix_cache_manager import (
    build_prefix_key as _build_prefix_key,
)
from dflash_mlx.server.request_loop import consume_dflash_events

def _read_project_version() -> str:
    try:
        return package_version("dflash-mlx")
    except PackageNotFoundError:
        return "unknown"

def _bytes_to_gib(value: int) -> str:
    return f"{float(value) / (1024 ** 3):.1f} GiB"

def _format_limit_request(value) -> str:
    if isinstance(value, int):
        return _bytes_to_gib(value)
    return str(value)

def _format_metal_limit(label: str, request, bytes_value: int | None, applied: bool) -> str:
    action = _bytes_to_gib(bytes_value) if applied and bytes_value is not None else "not set"
    return f"{label}: {_format_limit_request(request)} -> {action}"

_DFLASH_REQUEST_COUNTER = itertools.count(1)

def _build_prompt_regime(args, tokenizer) -> dict[str, object]:
    chat_template_args = getattr(args, "chat_template_args", None)
    if not isinstance(chat_template_args, dict):
        chat_template_args = {}
    return {
        "request_tokenization": "mlx_lm.server",
        "runtime_prompt_input": "prompt_tokens_override",
        "chat_template": True,
        "chat_template_args": dict(chat_template_args),
        "enable_thinking": bool(chat_template_args.get("enable_thinking", False)),
        "use_default_chat_template": bool(
            getattr(args, "use_default_chat_template", False)
        ),
        "custom_chat_template": bool(getattr(args, "chat_template", None)),
        "tokenizer_class": type(tokenizer).__name__,
    }

class DFlashResponseGenerator(mlx_server.ResponseGenerator):
    def _serve_single(self, request):
        request_tuple = request
        rqueue, request, args = request_tuple

        request_id = next(_DFLASH_REQUEST_COUNTER)
        runtime_context = self.model_provider.cli_args.runtime_context
        trace_config = runtime_context.diagnostics.trace
        bench_active = _bench_enabled(trace_config)

        if args.max_tokens <= 256:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] fast-path AR | max_tokens={args.max_tokens}\n"
            )
            sys.stderr.flush()
            saved_draft_model = self.model_provider.draft_model
            wall_t0 = time.perf_counter_ns()
            try:
                self.model_provider.draft_model = None
                return super()._serve_single((rqueue, request, args))
            finally:
                self.model_provider.draft_model = saved_draft_model
                if bench_active:
                    wall_ms = (time.perf_counter_ns() - wall_t0) / 1e6
                    _bench_log_post(
                        trace_config,
                        request_id=request_id,
                        mode_used="ar_fastpath",
                        max_tokens=int(args.max_tokens),
                        wall_ms=wall_ms,
                    )

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

            ctx = _build_generation_context(
                tokenizer,
                prompt,
                stop_words=args.stop_words,
                sequences=sequences,
            )
            rqueue.put(ctx)

            if args.seed is not None:
                mx.random.seed(args.seed)

            stop_token_ids = get_stop_token_ids(tokenizer)
            eos_token_ids = set(int(token_id) for token_id in tokenizer.eos_token_ids)
            request_start_ns = time.perf_counter_ns()
            prefix_flow = PrefixCacheFlow.for_request(
                model_provider=self.model_provider,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt=prompt,
                runtime_context=runtime_context,
            )
            ctx.prompt_cache_count = prefix_flow.hit_tokens

            event_iter = stream_dflash_generate(
                target_model=model,
                tokenizer=tokenizer,
                draft_model=draft_model,
                prompt="",
                max_new_tokens=args.max_tokens,
                use_chat_template=False,
                stop_token_ids=stop_token_ids,
                prompt_tokens_override=prompt,
                prefix_snapshot=prefix_flow.snapshot,
                stable_prefix_len=prefix_flow.stable_prefix_len,
                prefix_cache=prefix_flow.cache,
                runtime_context=runtime_context,
            )
            loop_result = consume_dflash_events(
                event_iter=event_iter,
                rqueue=rqueue,
                ctx=ctx,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=int(args.max_tokens),
                eos_token_ids=eos_token_ids,
                request_start_ns=request_start_ns,
                prefix_flow=prefix_flow,
                sm=sm,
                sm_state=sm_state,
                bench_active=bench_active,
                request_id=request_id,
                runtime_context=runtime_context,
            )
            summary_event = loop_result.summary_event

            if summary_event is not None:
                _write_summary_line(
                    summary_event=summary_event,
                    prompt_token_count=len(prompt),
                )

            if bench_active:
                _log_bench_post(
                    request_id=request_id,
                    summary_event=summary_event,
                    request_start_ns=loop_result.request_start_ns,
                    request_done_ns=time.perf_counter_ns(),
                    first_token_ns=loop_result.first_token_ns,
                    prefill_done_ns=loop_result.prefill_done_ns,
                    prompt_token_count=len(prompt),
                    live_token_count=loop_result.live_token_count,
                    cache_lookup_ms=loop_result.cache_lookup_ms,
                    cache_hit_tokens=loop_result.cache_hit_tokens,
                    cache_insert_ms=loop_result.cache_insert_ms,
                    finish_reason=loop_result.finish_reason,
                    max_tokens=args.max_tokens,
                    prompt_regime=_build_prompt_regime(args, tokenizer),
                    memory_waterfall_peak=loop_result.memory_waterfall_peak,
                    diagnostics=runtime_context.diagnostics,
                )
            try:
                mlx_active_gb = float(mx.get_active_memory()) / 1e9 if hasattr(mx, "get_active_memory") else 0.0
                mlx_cache_gb = float(mx.get_cache_memory()) / 1e9 if hasattr(mx, "get_cache_memory") else 0.0
                mlx_peak_gb = float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else 0.0
                import os as _os, resource as _resource, subprocess as _sp
                rusage = _resource.getrusage(_resource.RUSAGE_SELF)
                rss_peak_gb = float(rusage.ru_maxrss) / 1e9
                rss_now_gb = 0.0
                try:
                    out = _sp.check_output(
                        ["ps", "-o", "rss=", "-p", str(_os.getpid())],
                        stderr=_sp.DEVNULL, timeout=2,
                    ).decode().strip()
                    rss_now_gb = float(out) * 1024.0 / 1e9
                except Exception:
                    pass
                untracked_gb = max(0.0, rss_now_gb - mlx_active_gb - mlx_cache_gb)
                sys.stderr.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] req#{request_id} "
                    f"mlx_active={mlx_active_gb:.2f} mlx_cache={mlx_cache_gb:.2f} mlx_peak={mlx_peak_gb:.2f} "
                    f"rss_now={rss_now_gb:.2f} rss_peak={rss_peak_gb:.2f} untracked={untracked_gb:.2f} GB\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
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
    runtime_config = getattr(model_provider.cli_args, "runtime_config", None)
    target_fa_window = (
        int(runtime_config.target_fa_window) if runtime_config is not None else 0
    )
    pc_enabled = bool(runtime_config.prefix_cache) if runtime_config is not None else False
    if target_fa_window > 0:
        pc_status = "disabled (--target-fa-window)"
    else:
        pc_status = "enabled" if pc_enabled else "disabled (--no-prefix-cache)"
    target_fa_status = (
        "full KV" if target_fa_window == 0 else f"rotating window {target_fa_window}"
    )
    metal_limits = getattr(model_provider.cli_args, "metal_limits", None)
    raw_lines = [
        f"DFlash v{dflash_version} - speculative decoding engine",
        f"Target:       {target_ref}",
        f"Draft:        {draft_ref}{draft_suffix}",
        "Mode:         DFlash (speculative decoding active)",
        f"Prefix cache: {pc_status}",
        f"Target FA KV: {target_fa_status}",
        f"Server:       {server_name} on port {port}",
    ]
    if runtime_config is not None:
        l2_status = (
            f"on ({runtime_config.prefix_cache_l2_dir}, "
            f"{_bytes_to_gib(runtime_config.prefix_cache_l2_max_bytes)})"
            if runtime_config.prefix_cache_l2
            else "off"
        )
        bench_log_dir = runtime_config.bench_log_dir or "off"
        diagnostics_mode = getattr(model_provider.cli_args, "diagnostics", "off")
        diagnostics_dir = getattr(model_provider.cli_args, "diagnostics_dir_resolved", "")
        raw_lines.extend(
            [
                f"Profile:      {runtime_config.profile}",
                f"Prefill step: {runtime_config.prefill_step_size}",
                (
                    "Draft cache:  "
                    f"sink={runtime_config.draft_sink_size} "
                    f"window={runtime_config.draft_window_size}"
                ),
                f"Verify cap:   {runtime_config.verify_len_cap or 'block'}",
                f"Clear cache:  {'boundary' if runtime_config.clear_cache_boundaries else 'off'}",
                (
                    "L1 cache:     "
                    f"{'on' if runtime_config.prefix_cache else 'off'} "
                    f"entries={runtime_config.prefix_cache_max_entries} "
                    f"bytes={_bytes_to_gib(runtime_config.prefix_cache_max_bytes)}"
                ),
                f"L2 cache:     {l2_status}",
                f"Max snapshot: {runtime_config.max_snapshot_tokens}",
                f"Waterfall:    {'on' if runtime_config.memory_waterfall else 'off'}",
                f"Bench logs:   {bench_log_dir}",
                f"Diagnostics:  {diagnostics_mode}",
                f"Diagnostics dir: {diagnostics_dir or 'off'}",
                f"Verify mode:  {runtime_config.verify_mode}",
            ]
        )
    if metal_limits is not None:
        if metal_limits.metal_available:
            raw_lines.extend(
                [
                    _format_metal_limit(
                        "Wired limit",
                        metal_limits.wired_request,
                        metal_limits.wired_bytes,
                        metal_limits.wired_applied,
                    ),
                    _format_metal_limit(
                        "Cache limit",
                        metal_limits.cache_request,
                        metal_limits.cache_bytes,
                        metal_limits.cache_applied,
                    ),
                ]
            )
        else:
            raw_lines.extend(
                [
                    "Wired limit:  Metal unavailable -> not set",
                    "Cache limit:  Metal unavailable -> not set",
                ]
            )

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
        _wait_for_initial_model_load(model_provider, timeout_s=300.0)
        _print_startup_banner(port=port, model_provider=model_provider)
        try:
            mlx_server._run_http_server(
                host,
                port,
                response_generator,
                handler_class=DFlashAPIHandler,
            )
        finally:
            shutdown_dflash_prefix_cache()
    else:
        response_generator.join()

def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = _build_parser()
    if prog is not None:
        parser.prog = prog
    args = normalize_cli_args(parser.parse_args(list(argv) if argv is not None else None))
    metal_limits = configure_metal_limits(args)
    args.runtime_context = with_metal_limits(args.runtime_context, metal_limits)
    configure_logging(args.log_level)
    _run_with_dflash_server(args.host, args.port, DFlashModelProvider(args))

if __name__ == "__main__":
    main()

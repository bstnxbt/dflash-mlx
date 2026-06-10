# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import itertools
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx_lm.server as mlx_server

from dflash_mlx.cache.manager import shutdown_runtime_cache_manager
from dflash_mlx.observability.writer import enabled as _bench_enabled
from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
from dflash_mlx.server.metrics import (
    clear_live_request as _clear_live_request,
    configure_live_metrics as _configure_live_metrics,
    finalize_request_observability as _finalize_request_observability,
    record_target_only_request as _record_target_only_request,
    start_live_request as _start_live_request,
)
from dflash_mlx.server.model_provider import (
    DFlashModelProvider,
    wait_for_initial_model_load,
)
from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow
from dflash_mlx.server.protocol import (
    build_generation_context as _build_generation_context,
)
from dflash_mlx.server.request_loop import consume_dflash_events


ResponseGeneratorFactory = Callable[[DFlashModelProvider, Any, "ServerRuntime"], Any]


@dataclass(frozen=True)
class PreparedDFlashRequest:
    prompt: Any
    sequences: dict[str, Any]
    state_machine: Any | None
    state_machine_state: Any | None
    has_thinking: bool


class ServerRuntime:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        model_provider: DFlashModelProvider,
        version: str,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.model_provider = model_provider
        self.version = version
        self._request_counter = itertools.count(1)

    def next_request_id(self) -> int:
        return next(self._request_counter)

    def start_target_only_request(
        self,
        *,
        request_id: int,
        mode_used: str,
        max_tokens: int,
    ) -> None:
        _start_live_request(
            request_id=request_id,
            mode_used=mode_used,
            prompt_tokens=None,
            max_tokens=int(max_tokens),
        )

    def record_target_only_request(
        self,
        *,
        request_id: int,
        mode_used: str,
        wall_ms: float,
        max_tokens: int,
    ) -> None:
        _record_target_only_request(
            request_id=request_id,
            mode_used=mode_used,
            wall_ms=wall_ms,
            max_tokens=int(max_tokens),
            diagnostics=self.model_provider.cli_args.runtime_context.diagnostics,
        )

    def clear_request(self, *, request_id: int) -> None:
        _clear_live_request(request_id=request_id)

    def serve_dflash_request(
        self,
        *,
        request_id: int,
        rqueue: Any,
        request: Any,
        args: Any,
        prepared: PreparedDFlashRequest,
    ) -> None:
        runtime_context = self.model_provider.cli_args.runtime_context
        trace_config = runtime_context.diagnostics.trace
        bench_active = _bench_enabled(trace_config)
        model = self.model_provider.model
        tokenizer = self.model_provider.tokenizer
        draft_model = self.model_provider.draft_model
        draft_backend = getattr(self.model_provider, "draft_backend", None)
        target_ops = getattr(self.model_provider, "target_ops", None)
        if draft_backend is None:
            raise RuntimeError("DFlash draft backend is not loaded")
        if target_ops is None:
            raise RuntimeError("DFlash target ops are not loaded")

        ctx = _build_generation_context(
            tokenizer,
            prepared.prompt,
            stop_words=args.stop_words,
            sequences=prepared.sequences,
            has_thinking=prepared.has_thinking,
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
            prompt=prepared.prompt,
            request=request,
            request_id=request_id,
            max_new_tokens=int(args.max_tokens),
            runtime_context=runtime_context,
        )
        ctx.prompt_cache_count = prefix_flow.hit_tokens
        _start_live_request(
            request_id=request_id,
            mode_used="dflash",
            prompt_tokens=len(prepared.prompt),
            max_tokens=int(args.max_tokens),
            cache_hit_tokens=prefix_flow.hit_tokens,
            cache_lookup_ms=prefix_flow.lookup_ms,
            hit_kind=prefix_flow.hit_kind,
        )

        event_iter = stream_dflash_generate(
            target_model=model,
            target_ops=target_ops,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="",
            max_new_tokens=args.max_tokens,
            use_chat_template=False,
            quantize_kv_cache=bool(runtime_context.runtime.quantize_kv_cache),
            stop_token_ids=stop_token_ids,
            prompt_tokens_override=prepared.prompt,
            prefix_snapshot=prefix_flow.snapshot,
            snapshot_service=prefix_flow.snapshot_service,
            stable_prefix_len=prefix_flow.stable_prefix_len,
            prefix_cache_active=prefix_flow.cache_active,
            publish_generation_snapshot=prefix_flow.publish_generation_snapshot,
            prefix_hit_kind=prefix_flow.hit_kind,
            runtime_context=runtime_context,
        )
        # The engine owns the snapshot from here; keeping this server-side
        # ref would pin the multi-GB arrays for the whole request.
        prefix_flow.snapshot = None
        loop_result = consume_dflash_events(
            event_iter=event_iter,
            rqueue=rqueue,
            ctx=ctx,
            tokenizer=tokenizer,
            prompt=prepared.prompt,
            max_tokens=int(args.max_tokens),
            eos_token_ids=eos_token_ids,
            request_start_ns=request_start_ns,
            prefix_flow=prefix_flow,
            sm=prepared.state_machine,
            sm_state=prepared.state_machine_state,
            bench_active=bench_active,
            request_id=request_id,
            runtime_context=runtime_context,
        )

        _finalize_request_observability(
            request_id=request_id,
            summary_event=loop_result.summary_event,
            request_start_ns=loop_result.request_start_ns,
            request_done_ns=time.perf_counter_ns(),
            first_token_ns=loop_result.first_token_ns,
            prefill_done_ns=loop_result.prefill_done_ns,
            prompt_token_count=len(prepared.prompt),
            live_token_count=loop_result.live_token_count,
            cache_lookup_ms=loop_result.cache_lookup_ms,
            cache_hit_tokens=loop_result.cache_hit_tokens,
            cache_insert_ms=loop_result.cache_insert_ms,
            finish_reason=loop_result.finish_reason,
            max_tokens=args.max_tokens,
            prompt_regime=build_prompt_regime(args, tokenizer, request),
            memory_waterfall_peak=loop_result.memory_waterfall_peak,
            memory_waterfall_start=loop_result.memory_waterfall_start,
            memory_waterfall_end=loop_result.memory_waterfall_end,
            diagnostics=runtime_context.diagnostics,
            prefill_event=loop_result.prefill_event,
            runtime_config=runtime_context.runtime,
        )
        rqueue.put(None)

    def wait_until_ready(self, *, timeout_s: float = 300.0) -> None:
        wait_for_initial_model_load(self.model_provider, timeout_s=timeout_s)

    def configure_metrics(self) -> None:
        _configure_live_metrics(
            version=self.version,
            model_provider=self.model_provider,
        )

    def print_startup_banner(self) -> None:
        _print_startup_banner(
            port=self.port,
            model_provider=self.model_provider,
            version=self.version,
        )

    def stop_response_generator(self, response_generator: Any) -> None:
        response_generator.stop_and_join()

    def shutdown(self) -> None:
        shutdown_runtime_cache_manager()

    def serve_forever(
        self,
        *,
        response_generator_factory: ResponseGeneratorFactory,
        handler_class: type,
    ) -> None:
        group = mx.distributed.init()
        rank = group.rank()
        response_generator = None
        try:
            prompt_cache = mlx_server.LRUPromptCache(
                self.model_provider.cli_args.prompt_cache_size
            )
            response_generator = response_generator_factory(
                self.model_provider,
                prompt_cache,
                self,
            )
            if rank == 0:
                self.wait_until_ready(timeout_s=300.0)
                self.configure_metrics()
                self.print_startup_banner()
                mlx_server._run_http_server(
                    self.host,
                    self.port,
                    response_generator,
                    handler_class=handler_class,
                )
            else:
                response_generator.join()
        finally:
            try:
                if rank == 0 and response_generator is not None:
                    self.stop_response_generator(response_generator)
            finally:
                self.shutdown()


def build_prompt_regime(args: Any, tokenizer: Any, request: Any = None) -> dict[str, object]:
    chat_template_args = getattr(args, "chat_template_args", None)
    if not isinstance(chat_template_args, dict):
        chat_template_args = {}
    request_type = getattr(request, "request_type", None)
    is_chat = request_type == "chat"
    enable_thinking = chat_template_args.get(
        "enable_thinking",
        getattr(tokenizer, "has_thinking", False),
    )
    return {
        "request_type": request_type or "unknown",
        "request_tokenization": "mlx_lm.server",
        "runtime_prompt_input": "prompt_tokens_override",
        "chat_template": bool(is_chat and getattr(tokenizer, "has_chat_template", False)),
        "chat_template_args": dict(chat_template_args) if is_chat else {},
        "enable_thinking": bool(is_chat and enable_thinking),
        "use_default_chat_template": bool(
            is_chat and getattr(args, "use_default_chat_template", False)
        ),
        "custom_chat_template": bool(is_chat and getattr(args, "chat_template", None)),
        "tokenizer_class": type(tokenizer).__name__,
    }


def _bytes_to_gib(value: int) -> str:
    return f"{float(value) / (1024 ** 3):.1f} GiB"


def _format_limit_request(value: Any) -> str:
    if isinstance(value, int):
        return _bytes_to_gib(value)
    return str(value)


def _format_metal_limit(
    label: str,
    request: Any,
    bytes_value: int | None,
    applied: bool,
) -> str:
    action = _bytes_to_gib(bytes_value) if applied and bytes_value is not None else "not set"
    return f"{label}: {_format_limit_request(request)} -> {action}"


def _print_startup_banner(
    *,
    port: int,
    model_provider: DFlashModelProvider,
    version: str,
) -> None:
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
    draft_quant = getattr(model_provider, "effective_draft_quant", None)
    draft_meta = getattr(model_provider, "draft_meta", {}) or {}
    draft_quant_source = draft_meta.get("draft_quant_source")
    draft_quant_line = (
        f"{draft_quant} ({draft_quant_source})" if draft_quant else "none"
    )
    draft_load_dtype = draft_meta.get("draft_load_dtype") or "checkpoint"
    chat_template_args = getattr(model_provider.cli_args, "chat_template_args", {})
    if not isinstance(chat_template_args, dict):
        chat_template_args = {}
    thinking_enabled = bool(
        chat_template_args.get(
            "enable_thinking",
            getattr(getattr(model_provider, "tokenizer", None), "has_thinking", False),
        )
    )
    fastpath_max_tokens = int(
        getattr(model_provider.cli_args, "fastpath_max_tokens", 0) or 0
    )
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
        f"DFlash v{version} - speculative decoding engine",
        f"Target:       {target_ref}",
        f"Draft:        {draft_ref}{draft_suffix}",
        f"Draft quant:  {draft_quant_line}",
        f"Draft dtype:  {draft_load_dtype}",
        "Mode:         DFlash (speculative decoding active)",
        f"Thinking:     {'enabled' if thinking_enabled else 'disabled'}",
        (
            f"Fast path:    AR <= {fastpath_max_tokens} tokens"
            if fastpath_max_tokens > 0
            else "Fast path:    off"
        ),
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
        diagnostics_config = getattr(model_provider.cli_args, "diagnostics_config", None)
        trace_config = getattr(diagnostics_config, "trace", None)
        trace_log_dir = getattr(trace_config, "log_dir", None)
        diagnostics_mode = getattr(model_provider.cli_args, "diagnostics", "off")
        diagnostics_dir = getattr(model_provider.cli_args, "diagnostics_dir_resolved", "")
        diagnostics_display = diagnostics_dir or (
            str(trace_log_dir) if diagnostics_mode == "off" and trace_log_dir else "off"
        )
        waterfall_enabled = bool(getattr(diagnostics_config, "memory_waterfall", False))
        raw_lines.extend(
            [
                "Runtime:      default",
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
                f"Waterfall:    {'on' if waterfall_enabled else 'off'}",
                f"Diagnostics:  {diagnostics_mode}",
                f"Diagnostics dir: {diagnostics_display}",
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

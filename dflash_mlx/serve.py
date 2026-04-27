# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import itertools
import logging
import sys
import threading
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

from dflash_mlx.server.config import (
    build_parser as _build_parser,
    configure_logging,
    configure_metal_limits,
    normalize_cli_args,
)
from dflash_mlx.server.protocol import (
    STATEFUL_SERVER_API as _STATEFUL_SERVER_API,
    build_generation_context as _build_generation_context,
    make_response as _make_response,
    match_stream_token as _match_stream_token,
)
from dflash_mlx.bench_logger import (
    enabled as _bench_enabled,
    log_cycle as _bench_log_cycle,
    log_post as _bench_log_post,
)
from dflash_mlx.server.metrics import (
    log_bench_post as _log_bench_post,
    write_summary_line as _write_summary_line,
)
from dflash_mlx.generate import get_stop_token_ids
from dflash_mlx.server.model_provider import (
    DFlashModelProvider,
    wait_for_initial_model_load as _wait_for_initial_model_load,
)
from dflash_mlx.cache.codecs import build_snapshot, target_cache_is_serializable
from dflash_mlx.cache.policies import compute_stable_prefix_len, prefix_cache_enabled
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.runtime import stream_dflash_generate
from dflash_mlx.server.prefix_cache_manager import (
    build_prefix_key as _build_prefix_key,
    chat_template_marker_ids as _chat_template_marker_ids,
    format_stats_line,
    make_prefix_cache,
)

def _read_project_version() -> str:
    try:
        return package_version("dflash-mlx")
    except PackageNotFoundError:
        return "unknown"

_DFLASH_PREFIX_CACHE_SINGLETON: Optional[DFlashPrefixCache] = None
_DFLASH_PREFIX_CACHE_LOCK = threading.Lock()
_DFLASH_REQUEST_COUNTER = itertools.count(1)

def _get_dflash_prefix_cache() -> Optional[DFlashPrefixCache]:
    global _DFLASH_PREFIX_CACHE_SINGLETON
    if not prefix_cache_enabled():
        return None
    if _DFLASH_PREFIX_CACHE_SINGLETON is not None:
        return _DFLASH_PREFIX_CACHE_SINGLETON
    with _DFLASH_PREFIX_CACHE_LOCK:
        if _DFLASH_PREFIX_CACHE_SINGLETON is not None:
            return _DFLASH_PREFIX_CACHE_SINGLETON
        _DFLASH_PREFIX_CACHE_SINGLETON = make_prefix_cache()
    return _DFLASH_PREFIX_CACHE_SINGLETON

def log_prefix_cache_stats(label: str = "") -> None:
    cache = _DFLASH_PREFIX_CACHE_SINGLETON
    if cache is None:
        return
    format_stats_line(cache, label)

class DFlashResponseGenerator(mlx_server.ResponseGenerator):
    def _serve_single(self, request):
        request_tuple = request
        rqueue, request, args = request_tuple

        request_id = next(_DFLASH_REQUEST_COUNTER)
        bench_active = _bench_enabled()

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
            prefill_done_ns: Optional[int] = None
            first_token_ns: Optional[int] = None
            prefill_elapsed_s = 0.0
            live_tok_s = 0.0
            live_token_count = 0
            live_acceptance_pct = 0.0
            live_prompt_len = len(prompt)
            printed_prefill_progress = False

            prefix_cache = _get_dflash_prefix_cache()
            prefix_snapshot = None
            prefix_key = None
            stable_prefix_len: Optional[int] = None
            cache_lookup_ms = 0.0
            cache_hit_tokens = 0
            cache_insert_ms = 0.0
            if prefix_cache is not None:
                prefix_key = _build_prefix_key(self.model_provider, draft_model)
                im_start_id, assistant_id = _chat_template_marker_ids(tokenizer)
                stable_prefix_len = compute_stable_prefix_len(
                    prompt,
                    im_start_id=im_start_id,
                    assistant_id=assistant_id,
                )
                lookup_tokens = prompt[:stable_prefix_len]
                _lookup_t0 = time.perf_counter_ns()
                matched_len, prefix_snapshot = prefix_cache.lookup(
                    lookup_tokens, prefix_key
                )
                cache_lookup_ms = (time.perf_counter_ns() - _lookup_t0) / 1e6
                cache_hit_tokens = int(matched_len)
                if matched_len > 0:
                    saved = int(matched_len)
                    total = int(len(prompt))
                    sys.stderr.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix cache hit "
                        f"{saved}/{total} tokens (stable prefix {stable_prefix_len})\n"
                    )
                    sys.stderr.flush()
                log_prefix_cache_stats(label="lookup")
            ctx.prompt_cache_count = cache_hit_tokens

            event_iter = stream_dflash_generate(
                target_model=model,
                tokenizer=tokenizer,
                draft_model=draft_model,
                prompt="",
                max_new_tokens=args.max_tokens,
                use_chat_template=False,
                stop_token_ids=stop_token_ids,
                prompt_tokens_override=prompt,
                prefix_snapshot=prefix_snapshot,
                stable_prefix_len=stable_prefix_len,
            )

            client_done = False
            try:
                for event in event_iter:
                    if bench_active and event.get("event") == "cycle_complete":
                        _evt = {k: v for k, v in event.items() if k != "event"}
                        _bench_log_cycle(request_id=request_id, **_evt)
                        continue
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
                            rqueue.put((processed, total))
                            printed_prefill_progress = True
                        else:
                            prefill_elapsed_s = elapsed_s
                            prefill_done_ns = time.perf_counter_ns()
                            if not printed_prefill_progress:
                                sys.stderr.write(
                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefill: {processed}/{total} tokens | {elapsed_s:.1f}s\n"
                                )
                                sys.stderr.flush()
                        continue
                    if event.get("event") == "prefill_snapshot_ready":
                        if prefix_cache is not None and prefix_key is not None:
                            try:
                                evt_cache = event.get("target_cache")
                                evt_hidden = event.get("target_hidden")
                                evt_logits = event.get("last_logits")
                                evt_tokens = event.get("token_ids") or []
                                if (
                                    evt_cache is not None
                                    and evt_hidden is not None
                                    and evt_logits is not None
                                    and target_cache_is_serializable(evt_cache)
                                ):
                                    snap = build_snapshot(
                                        token_ids=list(evt_tokens),
                                        target_cache=evt_cache,
                                        target_hidden=evt_hidden,
                                        last_logits=evt_logits,
                                        key=prefix_key,
                                        kind="prefill",
                                    )
                                    _ins_t0 = time.perf_counter_ns()
                                    prefix_cache.insert(snap)
                                    cache_insert_ms += (time.perf_counter_ns() - _ins_t0) / 1e6
                            except Exception as _cache_err:
                                sys.stderr.write(
                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                                    f"[dflash] prefix cache insert failed: {_cache_err}\n"
                                )
                                sys.stderr.flush()
                        continue
                    if event.get("event") == "generation_snapshot_ready":
                        if prefix_cache is not None and prefix_key is not None:
                            try:
                                evt_cache = event.get("target_cache")
                                evt_hidden = event.get("target_hidden")
                                evt_tokens = event.get("token_ids") or []
                                if (
                                    evt_cache is not None
                                    and evt_hidden is not None
                                    and target_cache_is_serializable(evt_cache)
                                ):
                                    snap = build_snapshot(
                                        token_ids=list(evt_tokens),
                                        target_cache=evt_cache,
                                        target_hidden=evt_hidden,
                                        last_logits=event.get("last_logits"),
                                        key=prefix_key,
                                        kind="generation",
                                    )
                                    _ins_t0 = time.perf_counter_ns()
                                    prefix_cache.insert(snap)
                                    cache_insert_ms += (time.perf_counter_ns() - _ins_t0) / 1e6
                                    sys.stderr.write(
                                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                                        f"[dflash] end-of-request snapshot saved "
                                        f"({len(evt_tokens)} tokens)\n"
                                    )
                                    sys.stderr.flush()
                            except Exception as _cache_err:
                                sys.stderr.write(
                                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                                    f"[dflash] end-of-request snapshot failed: {_cache_err}\n"
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

                    if client_done:
                        break
                    token = int(event["token_id"])
                    if first_token_ns is None:
                        first_token_ns = time.perf_counter_ns()
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
                    token_finish_reason: Optional[str] = None
                    sm_state, match_sequence, current_state, terminal_match = (
                        _match_stream_token(sm, sm_state, token)
                    )
                    if terminal_match or token in eos_token_ids:
                        token_finish_reason = "stop"
                    elif live_token_count >= int(args.max_tokens):
                        token_finish_reason = "length"

                    text = ""
                    if token not in eos_token_ids:
                        detokenizer.add_token(token)
                        text = detokenizer.last_segment

                    if not first_token_flushed:
                        rqueue.put(
                            _make_response(
                                text=text,
                                token=token,
                                state=current_state or "normal",
                                match=match_sequence,
                                finish_reason=token_finish_reason,
                            )
                        )
                        first_token_flushed = True
                        if ctx._should_stop:
                            break
                        if token_finish_reason is not None:
                            client_done = True
                        continue

                    if pending_token is not None:
                        rqueue.put(
                            _make_response(
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
                    if token_finish_reason is not None:
                        client_done = True
            finally:
                event_iter.close()

            detokenizer.finalize()
            tail = detokenizer.last_segment
            if pending_token is not None:
                rqueue.put(
                    _make_response(
                        text=pending_text + tail,
                        token=pending_token,
                        state=pending_state,
                        match=pending_match,
                        finish_reason=finish_reason or pending_finish_reason,
                    )
                )

            if summary_event is not None:
                _write_summary_line(
                    summary_event=summary_event,
                    prompt_token_count=len(prompt),
                )

            if bench_active:
                _log_bench_post(
                    request_id=request_id,
                    summary_event=summary_event,
                    request_start_ns=request_start_ns,
                    request_done_ns=time.perf_counter_ns(),
                    first_token_ns=first_token_ns,
                    prefill_done_ns=prefill_done_ns,
                    prompt_token_count=len(prompt),
                    live_token_count=live_token_count,
                    cache_lookup_ms=cache_lookup_ms,
                    cache_hit_tokens=cache_hit_tokens,
                    cache_insert_ms=cache_insert_ms,
                    finish_reason=finish_reason,
                    max_tokens=args.max_tokens,
                )
            if hasattr(mx, "get_peak_memory"):
                try:
                    peak_gb = float(mx.get_peak_memory()) / 1e9
                    sys.stderr.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] req#{request_id} peak_memory={peak_gb:.2f} GB\n"
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
    pc_enabled = prefix_cache_enabled()
    pc_status = "enabled" if pc_enabled else "disabled (--no-prefix-cache)"
    raw_lines = [
        f"DFlash v{dflash_version} - speculative decoding engine",
        f"Target:       {target_ref}",
        f"Draft:        {draft_ref}{draft_suffix}",
        "Mode:         DFlash (speculative decoding active)",
        f"Prefix cache: {pc_status}",
        f"Server:       {server_name} on port {port}",
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
        _wait_for_initial_model_load(model_provider, timeout_s=300.0)
        _print_startup_banner(port=port, model_provider=model_provider)
        mlx_server._run_http_server(
            host,
            port,
            response_generator,
            handler_class=DFlashAPIHandler,
        )
    else:
        response_generator.join()

def main() -> None:
    args = normalize_cli_args(_build_parser().parse_args())
    configure_metal_limits()
    configure_logging(args.log_level)
    _run_with_dflash_server(args.host, args.port, DFlashModelProvider(args))

if __name__ == "__main__":
    main()

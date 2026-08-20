from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx

from dflash_mlx.diagnostics import DiagnosticsConfig
from dflash_mlx.engine.acceptance import match_acceptance_length
from dflash_mlx.engine.config import resolve_speculative_cycle_config
from dflash_mlx.engine.ddtree import (
    build_flat_ddtree,
    clone_cache_for_batch,
    draft_block_with_topk,
    flat_tree_path_token_ids,
    follow_verified_tree,
)
from dflash_mlx.engine.events import PrefillCompleteEvent
from dflash_mlx.engine.sampling import (
    build_suppress_token_mask,
    eval_logits_and_captured,
    greedy_tokens_with_mask,
)
from dflash_mlx.engine.spec_epoch import (
    SpeculativeSession,
    _RequestState,
    _SessionRequest,
    _YieldPauseTracker,
)
from dflash_mlx.runtime import get_stop_token_ids
from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.context import build_offline_runtime_config, build_runtime_context


def progress(message: str) -> None:
    print(f"[ddtree-flat-oracle] {message}", file=sys.stderr, flush=True)


def consume_generator_return(generator: Any) -> tuple[list[Any], Any]:
    events: list[Any] = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as done:
            return events, done.value


def request_files(source_trace: Path, request_indices: list[int]) -> list[Path]:
    request_dir = source_trace / "requests"
    paths = sorted(request_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No trace requests found under {request_dir}")
    if not request_indices:
        return paths
    selected: list[Path] = []
    for index in request_indices:
        if index < 1 or index > len(paths):
            raise ValueError(f"request index {index} outside 1..{len(paths)}")
        selected.append(paths[index - 1])
    return selected


def git_output(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def normalize_openai_messages(messages: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        item = dict(message)
        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls: list[Any] = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    normalized_calls.append(tool_call)
                    continue
                call = dict(tool_call)
                function = call.get("function")
                if isinstance(function, dict):
                    function = dict(function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        function["arguments"] = json.loads(arguments) if arguments else {}
                    call["function"] = function
                normalized_calls.append(call)
            item["tool_calls"] = normalized_calls
        normalized.append(item)
    return normalized


def trace_prompt_tokens(
    tokenizer: Any,
    request_file: Path,
    *,
    default_chat_template_kwargs: dict[str, Any],
) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
    payload = json.loads(request_file.read_text(encoding="utf-8"))
    body = payload.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"{request_file} does not contain a request body")
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{request_file} body.messages must be a list")
    messages = normalize_openai_messages(messages)
    template_kwargs = dict(default_chat_template_kwargs)
    request_kwargs = body.get("chat_template_kwargs")
    if isinstance(request_kwargs, dict):
        template_kwargs.update(request_kwargs)
    tools = body.get("tools")
    if tools is not None:
        template_kwargs["tools"] = tools
    try:
        prompt_tokens = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **template_kwargs,
            )
        )
    except TypeError as exc:
        if "tools" not in template_kwargs:
            raise
        fallback_kwargs = dict(template_kwargs)
        fallback_kwargs.pop("tools", None)
        try:
            prompt_tokens = list(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    **fallback_kwargs,
                )
            )
        except TypeError:
            raise exc
    return [int(token) for token in prompt_tokens], payload, body


def tree_posterior_token_ids(
    posterior: mx.array,
    path_lengths: list[int],
) -> list[int]:
    posterior_rows = posterior.tolist()
    return [
        int(posterior_rows[row_index][int(path_len) - 1])
        for row_index, path_len in enumerate(path_lengths)
    ]


def run_request(
    *,
    bundle: Any,
    runtime_context: Any,
    request_file: Path,
    prompt_tokens: list[int],
    max_output_tokens: int,
    requested_block_tokens: int,
    width: int,
    budget: int,
    oracle_batch_size: int,
    no_eos: bool,
) -> dict[str, Any]:
    progress(f"{request_file.name}: opening session")
    stop_token_ids = get_stop_token_ids(bundle.tokenizer)
    suppress_token_ids = stop_token_ids if no_eos else None
    supports_prefix_snapshot = bool(
        getattr(
            bundle.target_ops.capabilities_for(bundle.target_model),
            "supports_prefix_snapshot",
            True,
        )
    )
    allow_full_context_draft_layers = bool(
        getattr(
            bundle.target_ops.capabilities_for(bundle.target_model),
            "supports_full_context_draft_layers",
            False,
        )
    )
    session = SpeculativeSession.open(
        target_model=bundle.target_model,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        target_ops=bundle.target_ops,
        supports_prefix_snapshot=supports_prefix_snapshot,
        allow_full_context_draft_layers=allow_full_context_draft_layers,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_output_tokens,
        prefix_snapshot=None,
        quantize_kv_cache=False,
        target_fa_window=int(runtime_context.runtime.target_fa_window),
        runtime_context=runtime_context,
    )
    request = _SessionRequest.from_tokens(
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_output_tokens,
        block_tokens=requested_block_tokens,
        stop_token_ids=[] if no_eos else stop_token_ids,
        suppress_token_ids=suppress_token_ids,
        prefix_snapshot=None,
        snapshot_service=None,
        stable_prefix_len=None,
        prefix_cache_active=False,
        publish_generation_snapshot=False,
    )
    state = _RequestState()
    yield_pause = _YieldPauseTracker(enabled=False)
    started_ns = time.perf_counter_ns()
    try:
        progress(f"{request_file.name}: prefill start prompt_tokens={len(prompt_tokens)}")
        prefill_events, prefill = consume_generator_return(
            session._run_prefill_events(
                request=request,
                state=state,
                yield_pause=yield_pause,
            )
        )
        prefill_complete = next(
            (event for event in prefill_events if isinstance(event, PrefillCompleteEvent)),
            None,
        )
        if prefill_complete is None:
            raise RuntimeError("prefill did not return a PrefillCompleteEvent")
        progress(
            f"{request_file.name}: prefill done prefill_us={prefill_complete.prefill_us:.0f}"
        )
        state.start = request.prompt_len
        suppress_token_mask = build_suppress_token_mask(
            int(state.prefill_logits.shape[-1]),
            suppress_token_ids,
        )
        cycle_config = resolve_speculative_cycle_config(
            runtime_context.runtime,
            bundle.draft_model,
            requested_block_tokens,
        )
        block_tokens = int(cycle_config.effective_block_tokens)
        target_layer_ids = list(bundle.draft_model.target_layer_ids)
        capture_layer_ids = {int(layer_id) + 1 for layer_id in target_layer_ids}
        generated: list[int] = []
        cycles: list[dict[str, Any]] = []
        accepted_from_draft = 0
        draft_us_total = 0.0
        verify_us_total = 0.0
        replay_us_total = 0.0

        while len(generated) < max_output_tokens:
            assert state.staged_first is not None
            remaining = max_output_tokens - len(generated)
            block_len = max(1, min(block_tokens, remaining))
            start_len = int(state.start)
            root = state.staged_first.astype(mx.uint32)
            root_token_id = int(root.item())
            draft_context = prefill.feature_store.require_current_hidden()
            if block_len > 1:
                drafted_all, top_ids, top_values, draft_us = draft_block_with_topk(
                    target_model=bundle.target_model,
                    target_ops=bundle.target_ops,
                    draft_model=bundle.draft_model,
                    draft_cache=session.draft_cache,
                    prefix_tokens=root,
                    draft_context=draft_context,
                    block_len=block_len,
                    suppress_token_mask=suppress_token_mask,
                    top_width=max(1, int(width)),
                )
                draft_us_total += draft_us
                greedy_suffix = [int(token) for token in drafted_all[: block_len - 1].tolist()]
                greedy_row = [root_token_id, *greedy_suffix[: block_len - 1]]
                tree = build_flat_ddtree(
                    top_token_ids_desc=top_ids[: block_len - 1],
                    top_scores_desc=top_values[: block_len - 1],
                    budget=int(budget),
                )
            else:
                top_ids = []
                top_values = []
                greedy_row = [root_token_id]
                tree = build_flat_ddtree(
                    top_token_ids_desc=[],
                    top_scores_desc=[],
                    budget=0,
                )

            paths = flat_tree_path_token_ids(tree, root_token_id=root_token_id)
            pad_token_id = int(getattr(bundle.draft_model, "mask_token_id", root_token_id))
            path_lengths = [len(path) for path in paths]
            rows = [
                [*path, *([pad_token_id] * (block_len - len(path)))]
                for path in paths
            ]
            verify_start_ns = time.perf_counter_ns()
            posterior_token_ids: list[int] = []
            chunk_size = max(1, int(oracle_batch_size))
            for chunk_start in range(0, len(rows), chunk_size):
                chunk_rows = rows[chunk_start : chunk_start + chunk_size]
                chunk_lengths = path_lengths[chunk_start : chunk_start + chunk_size]
                verify_ids = mx.array(chunk_rows, dtype=mx.uint32)
                chunk_cache = clone_cache_for_batch(session.target_cache, len(chunk_rows))
                bundle.target_ops.arm_rollback(chunk_cache, prefix_len=int(state.start))
                chunk_logits, chunk_hidden = bundle.target_ops.verify_block(
                    target_model=bundle.target_model,
                    verify_ids=verify_ids,
                    target_cache=chunk_cache,
                    capture_layer_ids=set(),
                )
                eval_logits_and_captured(chunk_logits, chunk_hidden)
                chunk_posterior = greedy_tokens_with_mask(chunk_logits, suppress_token_mask)
                mx.eval(chunk_posterior)
                posterior_token_ids.extend(
                    tree_posterior_token_ids(chunk_posterior, chunk_lengths)
                )
            verify_us = (time.perf_counter_ns() - verify_start_ns) / 1_000.0
            verify_us_total += verify_us
            accepted_slots, next_token = follow_verified_tree(tree, posterior_token_ids)
            greedy_verify_start_ns = time.perf_counter_ns()
            greedy_verify_ids = mx.array([greedy_row], dtype=mx.uint32)
            greedy_cache = clone_cache_for_batch(session.target_cache, 1)
            bundle.target_ops.arm_rollback(greedy_cache, prefix_len=int(state.start))
            greedy_logits, greedy_hidden = bundle.target_ops.verify_block(
                target_model=bundle.target_model,
                verify_ids=greedy_verify_ids,
                target_cache=greedy_cache,
                capture_layer_ids=set(),
            )
            eval_logits_and_captured(greedy_logits, greedy_hidden)
            greedy_posterior = greedy_tokens_with_mask(greedy_logits, suppress_token_mask)
            mx.eval(greedy_posterior)
            greedy_verify_us = (time.perf_counter_ns() - greedy_verify_start_ns) / 1_000.0
            verify_us_total += greedy_verify_us
            greedy_acceptance = int(
                match_acceptance_length(
                    greedy_verify_ids[0, 1:],
                    greedy_posterior[0, :-1],
                ).item()
            )
            final_slot = int(accepted_slots[-1])
            commit_count = len(accepted_slots)
            acceptance_len = max(0, commit_count - 1)
            emitted_commit_count = min(commit_count, remaining)
            greedy_emitted_commit_count = min(int(greedy_acceptance) + 1, remaining)
            accepted_from_draft += max(0, emitted_commit_count - 1)
            commit_verify_start_ns = time.perf_counter_ns()
            selected_ids = mx.array([rows[final_slot]], dtype=mx.uint32)
            bundle.target_ops.arm_rollback(session.target_cache, prefix_len=int(state.start))
            logits, hidden_states = bundle.target_ops.verify_block(
                target_model=bundle.target_model,
                verify_ids=selected_ids,
                target_cache=session.target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            eval_logits_and_captured(logits, hidden_states)
            selected_posterior = greedy_tokens_with_mask(logits, suppress_token_mask)
            mx.eval(selected_posterior)
            commit_verify_us = (time.perf_counter_ns() - commit_verify_start_ns) / 1_000.0
            verify_us_total += commit_verify_us
            selected_next_token = int(selected_posterior[0, commit_count - 1].item())
            if selected_next_token != int(next_token):
                raise RuntimeError(
                    "selected path posterior changed during commit verify: "
                    f"{next_token} -> {selected_next_token}"
                )
            committed_hidden = bundle.target_ops.extract_context_feature(
                hidden_states,
                target_layer_ids,
            )[:, :commit_count, :]
            mx.eval(committed_hidden)
            state.start = start_len + commit_count
            prefill.feature_store.commit_generation(
                committed_hidden,
                collect_snapshot=False,
            )
            replay_start_ns = time.perf_counter_ns()
            replay_ns = bundle.target_ops.restore_after_acceptance(
                session.target_cache,
                target_len=state.start,
                acceptance_length=acceptance_len,
                drafted_tokens=block_len - 1,
            )
            replay_us = (time.perf_counter_ns() - replay_start_ns) / 1_000.0
            replay_us_total += replay_us
            state.last_cycle_logits = logits[:, commit_count - 1, :]
            state.staged_first = selected_posterior[0, commit_count - 1 : commit_count]
            committed_ids = [root_token_id]
            committed_ids.extend(
                int(tree.token_ids[slot - 1])
                for slot in accepted_slots[1:]
            )
            for token_id in committed_ids:
                if len(generated) >= max_output_tokens:
                    break
                generated.append(int(token_id))
            branch_extra = max(0, emitted_commit_count - greedy_emitted_commit_count)
            cycles.append(
                {
                    "cycle": len(cycles) + 1,
                    "start": start_len,
                    "block_len": block_len,
                    "tree_slots": tree.size,
                    "tree_nodes": tree.n_nodes,
                    "verify_rows": len(rows),
                    "acceptance_len": acceptance_len,
                    "commit_count": commit_count,
                    "emitted_commit_count": emitted_commit_count,
                    "greedy_acceptance_len": int(greedy_acceptance),
                    "greedy_commit_count": int(greedy_acceptance) + 1,
                    "greedy_emitted_commit_count": greedy_emitted_commit_count,
                    "branch_extra_tokens": branch_extra,
                    "branch_win": branch_extra > 0,
                    "accepted_slots": accepted_slots,
                    "root_top_ids": top_ids[0][: min(len(top_ids[0]), width)] if top_ids else [],
                    "root_top_values": top_values[0][: min(len(top_values[0]), width)] if top_values else [],
                    "draft_us": draft_us if block_len > 1 else 0.0,
                    "verify_us": verify_us,
                    "greedy_verify_us": greedy_verify_us,
                    "commit_verify_us": commit_verify_us,
                    "replay_us": replay_us,
                    "target_replay_ns": int(replay_ns),
                }
            )
            stop_hit = bool(stop_token_ids and any(token in stop_token_ids for token in committed_ids))
            if stop_hit:
                break
            progress(
                f"{request_file.name}: cycles={len(cycles)} tokens={len(generated)} "
                f"tree_tpc={len(generated) / len(cycles):.2f}"
            )
    finally:
        session.close()

    elapsed_us = (time.perf_counter_ns() - started_ns - yield_pause.pause_ns) / 1_000.0
    tree_commit_counts = [int(row["emitted_commit_count"]) for row in cycles]
    greedy_commit_counts = [int(row["greedy_emitted_commit_count"]) for row in cycles]
    tree_tpc = len(generated) / len(cycles) if cycles else 0.0
    greedy_equivalent_tpc = (
        sum(greedy_commit_counts) / len(greedy_commit_counts)
        if greedy_commit_counts
        else 0.0
    )
    gain = tree_tpc / greedy_equivalent_tpc if greedy_equivalent_tpc > 0 else 0.0
    cycle_reduction = 1.0 - (1.0 / gain) if gain > 0 else 0.0
    branch_wins = sum(1 for row in cycles if row["branch_win"])
    return {
        "request_file": str(request_file),
        "prompt_token_count": len(prompt_tokens),
        "prefill": prefill_complete.to_payload(),
        "summary": {
            "elapsed_us": elapsed_us,
            "decode_capture_heavy_tps": (
                len(generated)
                / max(1e-9, ((elapsed_us - prefill_complete.prefill_us) / 1_000_000.0))
            ),
            "generation_tokens": len(generated),
            "oracle_cycles": len(cycles),
            "tree_tokens_per_cycle": tree_tpc,
            "same_context_greedy_tokens_per_cycle": greedy_equivalent_tpc,
            "oracle_gain_vs_same_context_greedy": gain,
            "estimated_cycle_reduction_same_context": cycle_reduction,
            "accepted_from_draft": accepted_from_draft,
            "acceptance_ratio": accepted_from_draft / len(generated) if generated else 0.0,
            "branch_wins": branch_wins,
            "winner_branch_rate": branch_wins / len(cycles) if cycles else 0.0,
            "branch_extra_tokens": sum(int(row["branch_extra_tokens"]) for row in cycles),
            "tree_commit_mean": statistics.mean(tree_commit_counts) if tree_commit_counts else 0.0,
            "greedy_commit_mean": statistics.mean(greedy_commit_counts) if greedy_commit_counts else 0.0,
            "tree_nodes_mean": (
                statistics.mean(int(row["tree_nodes"]) for row in cycles)
                if cycles
                else 0.0
            ),
            "verify_rows_per_cycle": (
                statistics.mean(int(row["verify_rows"]) for row in cycles)
                if cycles
                else 0.0
            ),
            "draft_us_total": draft_us_total,
            "verify_us_total": verify_us_total,
            "replay_us_total": replay_us_total,
        },
        "cycles": cycles,
        "generated_token_ids": generated,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["aggregate_summary"]
    gate = float(summary["estimated_cycle_reduction_same_context"])
    verdict = (
        "KEEP: flat-tree oracle clears the 25% cycle-reduction gate."
        if gate >= 0.25
        else "DROP/PARK: flat-tree oracle does not clear the 25% cycle-reduction gate."
    )
    lines = [
        "# DDTree Flat Oracle Replay",
        "",
        "Lab-only measurement: it verifies every flat-tree root-to-node path as an independent row, then simulates the true tree walk. Timings are diagnostic, not a product speed claim.",
        "",
        "## Summary",
        "",
        f"- requests: `{summary['requests']}`",
        f"- generation_tokens: `{summary['generation_tokens']}`",
        f"- oracle_cycles: `{summary['oracle_cycles']}`",
        f"- tree_tokens_per_cycle: `{summary['tree_tokens_per_cycle']:.3f}`",
        f"- same_context_greedy_tokens_per_cycle: `{summary['same_context_greedy_tokens_per_cycle']:.3f}`",
        f"- oracle_gain_vs_same_context_greedy: `{summary['oracle_gain_vs_same_context_greedy']:.3f}x`",
        f"- estimated_cycle_reduction_same_context: `{summary['estimated_cycle_reduction_same_context']:.1%}`",
        f"- branch_wins: `{summary['branch_wins']}`",
        f"- winner_branch_rate: `{summary['winner_branch_rate']:.1%}`",
        f"- branch_extra_tokens: `{summary['branch_extra_tokens']}`",
        f"- tree_nodes_mean: `{summary['tree_nodes_mean']:.1f}`",
        f"- verify_rows_per_cycle: `{summary['verify_rows_per_cycle']:.1f}`",
        "",
        "## Verdict",
        "",
        verdict,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    cycles = sum(int(item["summary"]["oracle_cycles"]) for item in results)
    generation_tokens = sum(int(item["summary"]["generation_tokens"]) for item in results)
    greedy_tokens = sum(
        float(item["summary"]["same_context_greedy_tokens_per_cycle"])
        * int(item["summary"]["oracle_cycles"])
        for item in results
    )
    branch_wins = sum(int(item["summary"]["branch_wins"]) for item in results)
    branch_extra = sum(int(item["summary"]["branch_extra_tokens"]) for item in results)
    tree_tpc = generation_tokens / cycles if cycles else 0.0
    greedy_tpc = greedy_tokens / cycles if cycles else 0.0
    gain = tree_tpc / greedy_tpc if greedy_tpc > 0 else 0.0
    cycle_reduction = 1.0 - (1.0 / gain) if gain > 0 else 0.0
    weighted_mean_fields = ("tree_nodes_mean", "verify_rows_per_cycle")
    summary: dict[str, Any] = {
        "requests": len(results),
        "generation_tokens": generation_tokens,
        "oracle_cycles": cycles,
        "tree_tokens_per_cycle": tree_tpc,
        "same_context_greedy_tokens_per_cycle": greedy_tpc,
        "oracle_gain_vs_same_context_greedy": gain,
        "estimated_cycle_reduction_same_context": cycle_reduction,
        "branch_wins": branch_wins,
        "winner_branch_rate": branch_wins / cycles if cycles else 0.0,
        "branch_extra_tokens": branch_extra,
    }
    for field in weighted_mean_fields:
        summary[field] = (
            sum(float(item["summary"][field]) * int(item["summary"]["oracle_cycles"]) for item in results)
            / cycles
            if cycles
            else 0.0
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay real trace requests through a DDTree flat oracle.")
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--request-index", type=int, action="append", default=[])
    parser.add_argument("--target", default="mlx-community/Qwen3.6-27B-4bit")
    parser.add_argument("--draft", default="z-lab/Qwen3.6-27B-DFlash")
    parser.add_argument("--draft-quant", default="w4")
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--oracle-batch-size", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--draft-sink-size", type=int, default=64)
    parser.add_argument("--draft-window-size", type=int, default=1024)
    parser.add_argument("--target-fa-window", type=int, default=0)
    parser.add_argument("--verify-len-cap", type=int, default=0)
    parser.add_argument("--chat-template-args", default='{"enable_thinking": true}')
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--out-root", type=Path, default=Path(".artifacts/dflash/ddtree-flat-oracle-20260513"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    template_args = json.loads(args.chat_template_args)
    if not isinstance(template_args, dict):
        raise ValueError("--chat-template-args must decode to an object")
    label = args.label or f"{args.source_trace.name}-w{args.width}-b{args.budget}"
    out_dir = args.out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = build_offline_runtime_config(
        prefill_step_size=int(args.prefill_step_size),
        target_fa_window=int(args.target_fa_window),
        draft_sink_size=int(args.draft_sink_size),
        draft_window_size=int(args.draft_window_size),
        verify_len_cap=int(args.verify_len_cap),
        verify_mode="dflash",
    )
    runtime_context = build_runtime_context(
        runtime_config,
        DiagnosticsConfig(mode="off", run_dir=out_dir),
    )
    load_started = time.perf_counter()
    progress(f"loading target={args.target} draft={args.draft}")
    bundle = load_runtime_bundle(
        model_ref=args.target,
        draft_ref=args.draft,
        draft_quant=args.draft_quant,
        verify_config=runtime_context.verify,
    )
    progress(f"loaded bundle in {time.perf_counter() - load_started:.1f}s")
    selected_files = request_files(args.source_trace, args.request_index)
    results: list[dict[str, Any]] = []
    for request_file in selected_files:
        tokenize_started = time.perf_counter()
        progress(f"{request_file.name}: tokenization start")
        prompt_tokens, trace_payload, body = trace_prompt_tokens(
            bundle.tokenizer,
            request_file,
            default_chat_template_kwargs=template_args,
        )
        progress(
            f"{request_file.name}: tokenization done tokens={len(prompt_tokens)} "
            f"elapsed={time.perf_counter() - tokenize_started:.1f}s"
        )
        max_tokens = min(int(args.max_output_tokens), int(body.get("max_tokens") or args.max_output_tokens))
        result = run_request(
            bundle=bundle,
            runtime_context=runtime_context,
            request_file=request_file,
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_tokens,
            requested_block_tokens=int(args.block_tokens),
            width=int(args.width),
            budget=int(args.budget),
            oracle_batch_size=int(args.oracle_batch_size),
            no_eos=bool(args.no_eos),
        )
        result["trace_request"] = {
            "idx": trace_payload.get("idx"),
            "path": trace_payload.get("path"),
            "body_model": body.get("model"),
            "body_max_tokens": body.get("max_tokens"),
            "prompt_token_count": len(prompt_tokens),
        }
        results.append(result)
        request_out = out_dir / f"{request_file.stem}-result.json"
        request_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    payload = {
        "config": {
            "argv": sys.argv,
            "git_hash": git_output(["rev-parse", "HEAD"]),
            "git_status_short": git_output(["status", "--short"]),
            "out_dir": str(out_dir),
            "source_trace": str(args.source_trace),
            "request_indices": args.request_index,
            "prompt_regime": "trace body.messages via tokenizer.apply_chat_template",
            "chat_template_args": template_args,
            "openai_tool_call_arguments_normalized": True,
            "target": args.target,
            "resolved_target": bundle.resolved_model_ref,
            "draft": args.draft,
            "resolved_draft": bundle.resolved_draft_ref,
            "draft_quant": args.draft_quant,
            "effective_draft_quant": bundle.effective_draft_quant,
            "block_tokens": args.block_tokens,
            "width": args.width,
            "budget": args.budget,
            "oracle_batch_size": args.oracle_batch_size,
            "max_output_tokens": args.max_output_tokens,
            "no_eos": args.no_eos,
            "runtime": asdict(runtime_config),
        },
        "aggregate_summary": aggregate_results(results),
        "requests": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(out_dir / "summary.md", payload)
    print(json.dumps(payload["aggregate_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

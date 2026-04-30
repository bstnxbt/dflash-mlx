# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)
from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.cache.codecs import hydrate_target_cache
from dflash_mlx.cache.snapshot import (
    DFlashPrefixSnapshot,
    validate_prefix_snapshot as _validate_prefix_snapshot,
)
from dflash_mlx.engine.acceptance import match_acceptance_length as _match_acceptance_length
from dflash_mlx.engine.adaptive_fallback import (
    AdaptiveFallbackState,
    resolve_adaptive_fallback_config,
)
from dflash_mlx.engine.fallback import stream_baseline_generate
from dflash_mlx.engine.prefill import (
    compute_snapshot_boundary,
    init_target_hidden_from_snapshot,
)
from dflash_mlx.engine.rollback import cleanup_generation_caches as _cleanup_generation_caches
from dflash_mlx.engine.config import (
    _draft_window_override_enabled,
    _effective_draft_window_size,
    _profile_dflash_cycles_enabled,
    _resolve_dflash_max_ctx,
    _resolve_draft_window,
    _resolve_target_fa_window,
    _resolve_verify_len_cap,
)
from dflash_mlx.engine.target_verifier import (
    extract_context_feature_from_dict,
    target_forward_with_hidden_states,
)
from dflash_mlx.model import DFlashDraftModel


def stream_dflash_generate_impl(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    prefix_snapshot: Optional[DFlashPrefixSnapshot] = None,
    stable_prefix_len: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    from dflash_mlx.runtime import (
        _eval_logits_and_captured,
        _ns_to_us,
        _prepare_prompt_tokens,
        build_suppress_token_mask,
        configure_full_attention_split,
        detect_engine,
        greedy_tokens_with_mask,
        make_draft_backend,
        make_target_cache,
    )
    if quantize_kv_cache:
        configure_full_attention_split(target_model, enabled=False)

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    fallback_reason: Optional[str] = None

    prompt_len = len(prompt_tokens)
    dflash_max_ctx = _resolve_dflash_max_ctx()
    target_fa_window = _resolve_target_fa_window()
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        yield from stream_baseline_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            fallback_reason=fallback_reason,
        )
        return
    draft_sink_size, draft_window_size = _resolve_draft_window()
    draft_window_size = _effective_draft_window_size(
        draft_model,
        draft_window_size,
        context_len=prompt_len + max(0, int(max_new_tokens)),
        allow_full_attention_context=not _draft_window_override_enabled(),
    )
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    engine = detect_engine(target_model)
    draft_backend = make_draft_backend()

    snap_prefix_len = _validate_prefix_snapshot(prefix_snapshot, prompt_tokens)
    if snap_prefix_len > 0 and (quantize_kv_cache or target_fa_window > 0):
        snap_prefix_len = 0
    if snap_prefix_len > 0:
        template_cache = make_target_cache(
            target_model,
            enable_speculative_linear_cache=True,
            quantize_kv_cache=quantize_kv_cache,
            target_fa_window=target_fa_window,
        )
        try:
            assert prefix_snapshot is not None
            target_cache = hydrate_target_cache(prefix_snapshot, template_cache)
        except (ValueError, TypeError):
            snap_prefix_len = 0
            target_cache = make_target_cache(
                target_model,
                enable_speculative_linear_cache=True,
                quantize_kv_cache=quantize_kv_cache,
                target_fa_window=target_fa_window,
            )
        finally:
            del template_cache
    else:
        target_cache = make_target_cache(
            target_model,
            enable_speculative_linear_cache=True,
            quantize_kv_cache=quantize_kv_cache,
            target_fa_window=target_fa_window,
        )
    draft_cache = draft_backend.make_cache(
        draft_model=draft_model,
        sink_size=draft_sink_size,
        window_size=draft_window_size,
    )
    target_layer_id_list = list(draft_model.target_layer_ids)
    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}
    profile_cycles = _profile_dflash_cycles_enabled()

    try:
        start_ns = time.perf_counter_ns()
        _yield_pause_ns = 0
        prefill_start_ns = time.perf_counter_ns()
        prefill_step_size = int(os.environ.get("DFLASH_PREFILL_STEP_SIZE", "8192"))
        prefill_logits = None
        target_hidden: Optional[mx.array] = None

        if snap_prefix_len > 0:
            assert prefix_snapshot is not None
            target_hidden = init_target_hidden_from_snapshot(
                prefix_snapshot,
                snap_prefix_len=snap_prefix_len,
                prompt_len=prompt_len,
            )

        snapshot_boundary = compute_snapshot_boundary(prompt_len, stable_prefix_len)
        prefill_context_len = max(0, snapshot_boundary - 1)
        chunked_start = min(snap_prefix_len, prefill_context_len)
        for chunk_start in range(chunked_start, prefill_context_len, prefill_step_size):
            chunk_end = min(chunk_start + prefill_step_size, prefill_context_len)
            chunk_ids = prompt_array[:, chunk_start:chunk_end]
            prefill_logits, prefill_hidden_states = target_forward_with_hidden_states(
                target_model,
                input_ids=chunk_ids,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(prefill_logits, prefill_hidden_states)
            feat = extract_context_feature_from_dict(
                prefill_hidden_states,
                target_layer_id_list,
            )
            if target_hidden is None:
                target_hidden = mx.zeros(
                    (feat.shape[0], prompt_len, feat.shape[-1]),
                    dtype=feat.dtype,
                )
            target_hidden[:, chunk_start:chunk_end, :] = feat
            mx.eval(target_hidden)
            del feat, prefill_hidden_states
            _pre_yield = time.perf_counter_ns()
            yield {
                "event": "prefill_progress",
                "tokens_processed": chunk_end,
                "tokens_total": prompt_len,
            }
            _yield_pause_ns += time.perf_counter_ns() - _pre_yield

        if (
            snap_prefix_len > 0
            and snap_prefix_len == snapshot_boundary
            and prefix_snapshot is not None
            and prefix_snapshot.last_logits is not None
        ):
            last_logits_2d = prefix_snapshot.last_logits
            prefill_logits = mx.expand_dims(last_logits_2d, axis=1)
            mx.eval(prefill_logits)
        elif snapshot_boundary > 0 and snap_prefix_len < snapshot_boundary:
            final_prompt_start = snapshot_boundary - 1
            prefill_logits, prefill_hidden_states = target_forward_with_hidden_states(
                target_model,
                input_ids=prompt_array[:, final_prompt_start:snapshot_boundary],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(prefill_logits, prefill_hidden_states)
            feat = extract_context_feature_from_dict(
                prefill_hidden_states,
                target_layer_id_list,
            )
            if target_hidden is None:
                target_hidden = mx.zeros(
                    (feat.shape[0], prompt_len, feat.shape[-1]),
                    dtype=feat.dtype,
                )
            target_hidden[:, final_prompt_start:snapshot_boundary, :] = feat
            mx.eval(target_hidden)
            del feat, prefill_hidden_states
        _pre_yield = time.perf_counter_ns()
        yield {
            "event": "prefill_progress",
            "tokens_processed": snapshot_boundary,
            "tokens_total": prompt_len,
        }
        _yield_pause_ns += time.perf_counter_ns() - _pre_yield
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        _pre_yield = time.perf_counter_ns()
        yield {
            "event": "prefill_snapshot_ready",
            "token_ids": list(prompt_tokens[:snapshot_boundary]),
            "target_cache": target_cache,
            "target_hidden": target_hidden[:, :snapshot_boundary, :] if target_hidden is not None else None,
            "last_logits": prefill_logits[:, -1, :] if prefill_logits is not None else None,
            "from_snapshot": bool(snap_prefix_len > 0),
            "snap_prefix_len": snap_prefix_len,
            "snapshot_boundary": snapshot_boundary,
        }
        _yield_pause_ns += time.perf_counter_ns() - _pre_yield

        if snapshot_boundary < prompt_len:
            tail_logits, tail_hidden_states = target_forward_with_hidden_states(
                target_model,
                input_ids=prompt_array[:, snapshot_boundary:prompt_len],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(tail_logits, tail_hidden_states)
            tail_feat = extract_context_feature_from_dict(
                tail_hidden_states,
                target_layer_id_list,
            )
            if target_hidden is None:
                target_hidden = mx.zeros(
                    (tail_feat.shape[0], prompt_len, tail_feat.shape[-1]),
                    dtype=tail_feat.dtype,
                )
            target_hidden[:, snapshot_boundary:prompt_len, :] = tail_feat
            mx.eval(target_hidden)
            prefill_logits = tail_logits
            del tail_feat, tail_hidden_states
            _pre_yield = time.perf_counter_ns()
            yield {
                "event": "prefill_progress",
                "tokens_processed": prompt_len,
                "tokens_total": prompt_len,
            }
            _yield_pause_ns += time.perf_counter_ns() - _pre_yield

        prefill_ns = time.perf_counter_ns() - prefill_start_ns

        prefill_target_hidden_for_snapshot = target_hidden
        gen_hidden_chunks: list[mx.array] = []
        last_cycle_logits: Optional[mx.array] = None

        suppress_token_mask = build_suppress_token_mask(int(prefill_logits.shape[-1]), suppress_token_ids)
        staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_token_mask).reshape(-1)

        _pre_yield = time.perf_counter_ns()
        yield {
            "event": "prefill",
            "prefill_us": prefill_ns / 1_000.0,
            "prompt_token_count": prompt_len,
        }
        _yield_pause_ns += time.perf_counter_ns() - _pre_yield

        first_token_yielded = False
        if max_new_tokens > 0:
            first_token_yielded = True
            _pre_yield = time.perf_counter_ns()
            yield {
                "event": "token",
                "token_id": int(staged_first.item()),
                "generated_tokens": 1,
                "acceptance_ratio": 0.0,
                "cycles_completed": 0,
            }
            _yield_pause_ns += time.perf_counter_ns() - _pre_yield

        draft_block_size = int(draft_model.block_size)
        requested_block_tokens = draft_block_size if block_tokens is None else int(block_tokens)
        effective_block_tokens = max(1, min(requested_block_tokens, draft_block_size))
        adaptive_fallback = AdaptiveFallbackState(
            config=resolve_adaptive_fallback_config(os.environ),
            initial_block_tokens=effective_block_tokens,
        )
        block_token_buffer = mx.full(
            (effective_block_tokens,),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        mask_token_tail = mx.full(
            (max(0, effective_block_tokens - 1),),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        generated_token_ids: list[int] = []
        accepted_from_draft = 0
        cycles_completed = 0
        verify_len_cap = _resolve_verify_len_cap(target_model, effective_block_tokens)
        start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        seen_draft_cycle = False
        acceptance_history: list[int] = []
        cycle_profiles: list[dict[str, Any]] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
        }
        prefetched_draft: Optional[dict[str, Any]] = None

        while len(generated_token_ids) < max_new_tokens:
            cycle_start_ns = time.perf_counter_ns()
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            commit_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            remaining = max_new_tokens - len(generated_token_ids)
            block_len = adaptive_fallback.block_tokens_for_cycle(remaining)
            adaptive_ar_cycle = adaptive_fallback.in_cooldown
            block_token_buffer[:block_len] = int(draft_model.mask_token_id)
            block_token_buffer[:1] = staged_first
            block_token_ids = block_token_buffer[:block_len]
            current_staged_first = staged_first
            drafted = None

            if block_len > 1:
                if profile_cycles:
                    draft_start_ns = time.perf_counter_ns()
                    drafted = draft_backend.draft_greedy(
                        target_model=target_model,
                        draft_model=draft_model,
                        draft_cache=draft_cache,
                        staged_first=current_staged_first,
                        target_hidden=target_hidden,
                        block_len=block_len,
                        mask_token_tail=mask_token_tail,
                        suppress_token_mask=suppress_token_mask,
                        async_launch=False,
                    )
                    mx.eval(drafted)
                    draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    block_token_ids[1:block_len] = drafted
                else:
                    if (
                        prefetched_draft is not None
                        and int(prefetched_draft["block_len"]) == block_len
                    ):
                        drafted = prefetched_draft["drafted"]
                        current_staged_first = prefetched_draft["staged_first"]
                    else:
                        draft_start_ns = time.perf_counter_ns()
                        drafted = draft_backend.draft_greedy(
                            target_model=target_model,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=current_staged_first,
                            target_hidden=target_hidden,
                            block_len=block_len,
                            mask_token_tail=mask_token_tail,
                            suppress_token_mask=suppress_token_mask,
                            async_launch=True,
                        )
                        draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    prefetched_draft = None
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            verify_token_count = min(block_len, verify_len_cap)
            if profile_cycles or block_len <= 1:
                verify_token_ids = block_token_ids[:verify_token_count]
            elif verify_token_count <= 1:
                verify_token_ids = current_staged_first[:1]
            else:
                verify_token_ids = mx.concatenate(
                    [current_staged_first[:1], drafted[: verify_token_count - 1]],
                    axis=0,
                )
            verify_ids = verify_token_ids[None]
            engine.arm_rollback(target_cache, prefix_len=start)
            verify_start_ns = time.perf_counter_ns()
            verify_logits, verify_hidden_states = engine.verify(
                target_model=target_model,
                verify_ids=verify_ids,
                target_cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            if profile_cycles:
                _eval_logits_and_captured(verify_logits, verify_hidden_states)
            verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
            verify_ns_total += verify_cycle_ns

            acceptance_start_ns = time.perf_counter_ns()
            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
            if not profile_cycles:
                mx.async_eval(posterior)
            acceptance_len = int(
                _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
            )
            acceptance_history.append(acceptance_len)
            acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            hidden_extract_start_ns = time.perf_counter_ns()
            committed_hidden = extract_context_feature_from_dict(
                verify_hidden_states,
                target_layer_id_list,
            )[:, : (1 + acceptance_len), :]
            if profile_cycles:
                mx.eval(committed_hidden, posterior)
            else:
                mx.async_eval(committed_hidden)
            hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            commit_start_ns = time.perf_counter_ns()
            start += commit_count
            if adaptive_ar_cycle and target_hidden is not None:
                target_hidden = mx.concatenate([target_hidden, committed_hidden], axis=1)
                if profile_cycles:
                    mx.eval(target_hidden)
                else:
                    mx.async_eval(target_hidden)
            else:
                target_hidden = committed_hidden
            gen_hidden_chunks.append(committed_hidden)
            last_cycle_logits = verify_logits[:, acceptance_len, :]
            replay_cycle_ns = engine.rollback(
                target_cache,
                target_len=start,
                acceptance_len=acceptance_len,
                drafted_tokens=max(0, verify_token_count - 1),
                force_replay=verify_token_count <= 1,
            )
            replay_ns_total += replay_cycle_ns
            cycles_completed += 1
            commit_wall_ns = time.perf_counter_ns() - commit_start_ns
            commit_ns_total += commit_wall_ns
            commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

            accepted_from_draft += acceptance_len
            staged_first_next = posterior[acceptance_len : acceptance_len + 1]

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            remaining_after_commit = max_new_tokens - len(generated_token_ids) - commit_count
            observed_cycle_us = (time.perf_counter_ns() - cycle_start_ns) / 1_000.0
            adaptive_decision = adaptive_fallback.record_cycle(
                block_len=block_len,
                commit_count=commit_count,
                cycle_us=observed_cycle_us,
                can_continue=not stop_hit and remaining_after_commit > 0,
            )
            if adaptive_decision is not None:
                prefetched_draft = None
                _pre_yield = time.perf_counter_ns()
                yield adaptive_decision.as_event()
                _yield_pause_ns += time.perf_counter_ns() - _pre_yield
            if not profile_cycles:
                next_block_len = adaptive_fallback.block_tokens_for_cycle(remaining_after_commit)
                if not stop_hit and remaining_after_commit > 0 and next_block_len > 1:
                    draft_start_ns = time.perf_counter_ns()
                    next_drafted = draft_backend.draft_greedy(
                        target_model=target_model,
                        draft_model=draft_model,
                        draft_cache=draft_cache,
                        staged_first=staged_first_next,
                        target_hidden=target_hidden,
                        block_len=next_block_len,
                        mask_token_tail=mask_token_tail,
                        suppress_token_mask=suppress_token_mask,
                        async_launch=True,
                    )
                    launch_ns = time.perf_counter_ns() - draft_start_ns
                    draft_ns_total += launch_ns
                    draft_incremental_ns += launch_ns
                    prefetched_draft = {
                        "block_len": next_block_len,
                        "staged_first": staged_first_next,
                        "drafted": next_drafted,
                    }
                else:
                    prefetched_draft = None
            committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
            for token_id in committed_ids:
                if len(generated_token_ids) >= max_new_tokens:
                    break
                generated_token_ids.append(token_id)
                if first_token_yielded:
                    first_token_yielded = False
                    continue
                _pre_yield = time.perf_counter_ns()
                yield {
                    "event": "token",
                    "token_id": token_id,
                    "generated_tokens": len(generated_token_ids),
                    "acceptance_ratio": (
                        accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
                    ),
                    "cycles_completed": cycles_completed,
                }
                _yield_pause_ns += time.perf_counter_ns() - _pre_yield

            if stop_hit:
                break

            staged_first = staged_first_next

            if profile_cycles:
                cycle_total_ns = time.perf_counter_ns() - cycle_start_ns
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profile_entry = {
                    "cycle": cycles_completed,
                    "block_len": int(block_len),
                    "commit_count": int(commit_count),
                    "acceptance_len": int(acceptance_len),
                    "draft_us": _ns_to_us(draft_cycle_ns),
                    "verify_us": _ns_to_us(verify_cycle_ns),
                    "acceptance_us": _ns_to_us(acceptance_cycle_ns),
                    "hidden_extraction_us": _ns_to_us(hidden_extract_cycle_ns),
                    "rollback_us": _ns_to_us(replay_cycle_ns),
                    "other_us": _ns_to_us(other_cycle_ns),
                    "cycle_total_us": _ns_to_us(cycle_total_ns),
                }
                cycle_profiles.append(cycle_profile_entry)
                _pre_yield = time.perf_counter_ns()
                yield {"event": "cycle_complete", **cycle_profile_entry}
                _yield_pause_ns += time.perf_counter_ns() - _pre_yield
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns

        if (
            generated_token_ids
            and prefill_target_hidden_for_snapshot is not None
            and gen_hidden_chunks
        ):
            try:
                gen_hidden = (
                    gen_hidden_chunks[0]
                    if len(gen_hidden_chunks) == 1
                    else mx.concatenate(gen_hidden_chunks, axis=1)
                )
                end_target_hidden = mx.concatenate(
                    [prefill_target_hidden_for_snapshot, gen_hidden], axis=1
                )
                mx.eval(end_target_hidden)
                if last_cycle_logits is not None:
                    mx.eval(last_cycle_logits)
                end_total_len = prompt_len + len(generated_token_ids)
                _pre_yield = time.perf_counter_ns()
                yield {
                    "event": "generation_snapshot_ready",
                    "token_ids": list(prompt_tokens) + list(generated_token_ids),
                    "target_cache": target_cache,
                    "target_hidden": end_target_hidden,
                    "last_logits": last_cycle_logits,
                    "snapshot_boundary": end_total_len,
                }
                _yield_pause_ns += time.perf_counter_ns() - _pre_yield
            except Exception as _gen_snap_err:
                sys.stderr.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"[dflash] generation_snapshot_ready build failed: {_gen_snap_err}\n"
                )
                sys.stderr.flush()

        elapsed_us = (time.perf_counter_ns() - start_ns - _yield_pause_ns) / 1_000.0
        first_20 = acceptance_history[:20]
        last_20 = acceptance_history[-20:]
        summary = {
            "event": "summary",
            "elapsed_us": elapsed_us,
            "prompt_token_count": prompt_len,
            "generated_token_ids": generated_token_ids,
            "generation_tokens": len(generated_token_ids),
            "accepted_from_draft": accepted_from_draft,
            "acceptance_ratio": (
                accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
            ),
            "block_tokens": effective_block_tokens,
            "cycles_completed": cycles_completed,
            "phase_timings_us": {
                "prefill": prefill_ns / 1_000.0,
                "draft": draft_ns_total / 1_000.0,
                "draft_prefill": draft_prefill_ns / 1_000.0,
                "draft_incremental": draft_incremental_ns / 1_000.0,
                "verify": verify_ns_total / 1_000.0,
                "replay": replay_ns_total / 1_000.0,
                "commit": commit_ns_total / 1_000.0,
            },
            "verify_len_cap": int(verify_len_cap),
            "quantize_kv_cache": bool(quantize_kv_cache),
            "target_fa_window": int(target_fa_window),
            "draft_sink_size": int(draft_sink_size),
            "draft_window_size": int(draft_window_size),
            "tokens_per_cycle": (len(generated_token_ids) / cycles_completed) if cycles_completed > 0 else 0.0,
            "acceptance_history": list(acceptance_history),
            "acceptance_first_20_avg": (sum(first_20) / len(first_20)) if first_20 else 0.0,
            "acceptance_last_20_avg": (sum(last_20) / len(last_20)) if last_20 else 0.0,
            "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
        }
        summary.update(adaptive_fallback.summary_fields())
        if profile_cycles:
            summary["cycle_profile_us"] = cycle_profiles
            summary["cycle_profile_totals_us"] = {
                key: _ns_to_us(value) for key, value in profile_totals_ns.items()
            }
        yield summary
    finally:
        _cleanup_generation_caches(target_cache, draft_cache)
        del draft_cache
        del target_cache
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

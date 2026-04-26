# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx


def _make_gated_delta_kernel_with_tape(
    *, has_mask: bool = False, vectorized: bool = False
):
    if not mx.metal.is_available():
        return None

    mask_load = (
        "bool do_step = static_cast<float>(mask[b_idx * T + t]) > 0.5f;"
        if has_mask
        else "constexpr bool do_step = true;"
    )

    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y, tape: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          {mask_load}

          // Save pre-step state so we can restore it when masked.
          float old_state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {{
            old_state[i] = state[i];
          }}

          float out = 0.0f;
          float delta = 0.0f;

          // do_step is a uniform predicate — all threads in the simdgroup
          // take the same branch, so simd_sum calls are safe inside.
          if (do_step) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem  += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                    * static_cast<float>(beta_[hv_idx]);

            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] += k_[s_idx] * delta;
              out      += state[i] * static_cast<float>(q_[s_idx]);
            }}
            out = simd_sum(out);
          }}

          if (thread_index_in_simdgroup == 0) {{
            y[dv_idx]     = static_cast<InT>(out);
            tape_[dv_idx] = delta;
          }}

          // Quantize new state, then conditional move: restore old state
          // when masked so decay and quantization are fully skipped.
          for (int i = 0; i < n_per_t; ++i) {{
            float quant_new = static_cast<float>(static_cast<InT>(state[i]));
            state[i] = metal::select(old_state[i], quant_new, do_step);
          }}
          q_    += Hk * Dk;
          k_    += Hk * Dk;
          v_    += Hv * Dv;
          y     += Hv * Dv;
          tape_ += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """

    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_tape{suffix}",
        input_names=inputs,
        output_names=["y", "state_out", "innovation_tape"],
        source=source,
    )


_gated_delta_tape_kernel = _make_gated_delta_kernel_with_tape(
    has_mask=False, vectorized=False
)
_gated_delta_tape_kernel_masked = _make_gated_delta_kernel_with_tape(
    has_mask=True, vectorized=False
)
_gated_delta_tape_kernel_vec = _make_gated_delta_kernel_with_tape(
    has_mask=False, vectorized=True
)
_gated_delta_tape_kernel_vec_masked = _make_gated_delta_kernel_with_tape(
    has_mask=True, vectorized=True
)


def _gated_delta_ops_with_tape(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Hv % Hk != 0:
        raise ValueError(f"Cannot align K heads {Hk} to V heads {Hv}")
    repeat_factor = Hv // Hk
    if repeat_factor > 1:
        q = mx.repeat(q, repeat_factor, axis=2)
        k = mx.repeat(k, repeat_factor, axis=2)

    outputs = []
    tape = []
    for t in range(T):
        old_state = state
        if g.ndim == 4:
            decay = g[:, t, :, None, :]
        elif g.ndim == 3:
            decay = g[:, t, :, None, None]
        else:
            raise ValueError(f"Unsupported gating shape {g.shape}")
        decayed_state = state * decay
        kv_mem = (decayed_state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        new_state = decayed_state + k[:, t, :, None, :] * delta[..., None]
        y = (new_state * q[:, t, :, None, :]).sum(axis=-1)
        if mask is not None:
            step_mask = mask[:, t][:, None, None, None]
            y_mask = mask[:, t][:, None, None]
            new_state = mx.where(step_mask, new_state, old_state)
            delta = mx.where(y_mask, delta, mx.zeros_like(delta))
            y = mx.where(y_mask, y, mx.zeros_like(y))
        state = new_state
        outputs.append(y)
        tape.append(delta.astype(mx.float32))
    return mx.stack(outputs, axis=1), state, mx.stack(tape, axis=1)


def gated_delta_kernel_with_tape(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
):
    if not mx.metal.is_available():
        return _gated_delta_ops_with_tape(q, k, v, g, beta, state, mask)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk < 32 or Dk % 32 != 0:
        return _gated_delta_ops_with_tape(q, k, v, g, beta, state, mask)

    input_type = q.dtype

    if g.ndim == 4:
        kernel = _gated_delta_tape_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_tape_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_tape_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_tape_kernel_masked
            inputs.append(mask)

    if kernel is None:
        return _gated_delta_ops_with_tape(q, k, v, g, beta, state, mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, (B, T, Hv, Dv)],
        output_dtypes=[input_type, input_type, mx.float32],
    )


def _make_tape_replay_kernel(*, has_mask: bool = False, vectorized: bool = False):
    if not mx.metal.is_available():
        return None

    mask_load = (
        "bool do_step = static_cast<float>(mask[b_idx * T + t]) > 0.5f;"
        if has_mask
        else "constexpr bool do_step = true;"
    )

    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // tape: [B, T, Hv, Dv]
        auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        // k: [B, T, Hk, Dk]
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}

        for (int t = 0; t < T; ++t) {{
          {mask_load}
          float delta = static_cast<float>(tape_[dv_idx]);
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            float next = state[i] * {g_access} + k_[s_idx] * delta;
            next = static_cast<float>(static_cast<InT>(next));
            // Conditional move: old state when masked, next when accepted.
            // do_step is uniform across the simdgroup — no divergence.
            state[i] = metal::select(state[i], next, do_step);
          }}
          tape_ += Hv * Dv;
          k_    += Hk * Dk;
          {g_advance}
        }}

        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """

    inputs = ["tape", "k", "g", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"tape_replay{suffix}",
        input_names=inputs,
        output_names=["state_out"],
        source=source,
    )


_tape_replay_kernel = _make_tape_replay_kernel(has_mask=False, vectorized=False)
_tape_replay_kernel_masked = _make_tape_replay_kernel(has_mask=True, vectorized=False)
_tape_replay_kernel_vec = _make_tape_replay_kernel(has_mask=False, vectorized=True)
_tape_replay_kernel_vec_masked = _make_tape_replay_kernel(
    has_mask=True, vectorized=True
)


def _tape_replay_ops(
    tape: mx.array,
    k: mx.array,
    g: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> mx.array:
    _, _, hk, _ = k.shape
    hv = tape.shape[2]
    if hv % hk != 0:
        raise ValueError(f"Cannot align K heads {hk} to tape heads {hv}")
    repeat_factor = hv // hk
    if repeat_factor > 1:
        k = mx.repeat(k, repeat_factor, axis=2)

    for t in range(int(tape.shape[1])):
        prev_state = state
        if g.ndim == 4:
            decay = g[:, t, :, None, :]
        elif g.ndim == 3:
            decay = g[:, t, :, None, None]
        else:
            raise ValueError(f"Unsupported gating shape {g.shape}")
        delta = tape[:, t, :, :, None]
        k_t = k[:, t, :, None, :]
        state = state * decay
        state = state + delta * k_t
        if mask is not None:
            step_mask = mask[:, t][:, None, None, None]
            state = mx.where(step_mask, state, prev_state)
    return state


def tape_replay_kernel(
    tape: mx.array,
    k: mx.array,
    g: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> mx.array:
    if not mx.metal.is_available():
        return _tape_replay_ops(tape, k, g, state, mask)

    bsz, steps, hk, dk = k.shape
    hv, dv = tape.shape[2:]
    input_type = state.dtype
    if dk < 32 or dk % 32 != 0:
        return _tape_replay_ops(tape, k, g, state, mask)

    if g.ndim == 4:
        kernel = _tape_replay_kernel_vec
        inputs = [tape, k, g, state, steps]
        if mask is not None:
            kernel = _tape_replay_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _tape_replay_kernel
        inputs = [tape, k, g, state, steps]
        if mask is not None:
            kernel = _tape_replay_kernel_masked
            inputs.append(mask)

    if kernel is None:
        return _tape_replay_ops(tape, k, g, state, mask)

    (state_out,) = kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("Dk", dk),
            ("Dv", dv),
            ("Hk", hk),
            ("Hv", hv),
        ],
        grid=(32, dv, bsz * hv),
        threadgroup=(32, 4, 1),
        output_shapes=[state.shape],
        output_dtypes=[input_type],
    )
    return state_out


def _compute_sdpa_2pass_blocks(
    gqa_factor: int, n_kv: int, device_arch: Optional[str] = None
) -> int:
    arch = device_arch or str(mx.device_info().get("architecture", ""))
    devc = arch[-1] if arch else ""
    n_simds = int(gqa_factor)  # Match AR qL=1 dispatch heuristic.
    N = int(n_kv)

    if devc == "d":
        blocks = 128
        if n_simds <= 2 and N > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= N < 65536:
                blocks = 512
            elif N >= 65536:
                blocks = 1024
    elif devc == "s":
        blocks = 64
        if N > 1024 and n_simds > 4:
            if N <= 8192:
                blocks = 128
            elif N <= 32768:
                blocks = 256
            elif N <= 65536:
                blocks = 512
            else:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32

    return int(blocks)


def _make_batched_sdpa_2pass_partials_kernel(*, has_mask: bool = False):
    if not mx.metal.is_available():
        return None

    mask_setup = (
        "auto mask_ = mask + (((b_idx * Hq + q_head_idx) * M_FIXED + q_seq_idx) * N + block_idx);"
        if has_mask
        else ""
    )
    # Branchless mask: float gate fused into score; non-masked path is compile-time constant.
    mask_gate = (
        "float mask_gate = static_cast<float>(mask_[0]); use_key = use_key & (mask_gate > Limits<InT>::finite_min);"
        if has_mask
        else "constexpr float mask_gate = 0.0f; (void)mask_gate;"
    )
    mask_score = "score += mask_gate;" if has_mask else ""
    mask_advance = "mask_ += blocks;" if has_mask else ""
    inputs = [
        "queries",
        "keys",
        "values",
        "gqa_factor",
        "N",
        "k_head_stride",
        "k_seq_stride",
        "v_head_stride",
        "v_seq_stride",
        "scale",
        "blocks",
    ]
    if has_mask:
        inputs.append("mask")

    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        auto q_head_idx = threadgroup_position_in_grid.x;
        auto b_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto q_seq_idx = thread_position_in_threadgroup.z;
        auto simd_lid = thread_index_in_simdgroup;

        auto Hq = threadgroups_per_grid.x;
        auto hk_idx = q_head_idx / gqa_factor;
        auto q_batch_head_idx = b_idx * Hq + q_head_idx;
        auto o_offset = q_batch_head_idx * M_FIXED + q_seq_idx;

        auto q_ = queries + (o_offset * D) + simd_lid * qk_per_thread;
        auto k_ = keys + ((b_idx * Hk + hk_idx) * k_head_stride) + block_idx * k_seq_stride + simd_lid * qk_per_thread;
        auto v_ = values + ((b_idx * Hk + hk_idx) * v_head_stride) + block_idx * v_seq_stride + simd_lid * v_per_thread;

        partials += (o_offset * blocks + block_idx) * V + simd_lid * v_per_thread;
        sums += o_offset * blocks + block_idx;
        maxs += o_offset * blocks + block_idx;
        {mask_setup}

        thread float q[qk_per_thread];
        thread float o[v_per_thread];
        threadgroup InT tg_k[BD * qk_per_thread];
        threadgroup InT tg_v[BD * v_per_thread];

        for (int i = 0; i < qk_per_thread; ++i) {{
            q[i] = static_cast<float>(scale) * static_cast<float>(q_[i]);
        }}
        for (int i = 0; i < v_per_thread; ++i) {{
            o[i] = 0.0f;
        }}

        float max_score = Limits<float>::finite_min;
        float sum_exp_score = 0.0f;

        for (int n = block_idx; n < N; n += blocks) {{
            if (q_seq_idx == 0) {{
                for (int i = 0; i < qk_per_thread; ++i) {{
                    tg_k[simd_lid * qk_per_thread + i] = k_[i];
                }}
                for (int i = 0; i < v_per_thread; ++i) {{
                    tg_v[simd_lid * v_per_thread + i] = v_[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool use_key = (n <= (N - M_FIXED + q_seq_idx));
            {mask_gate}

            // Compute score unconditionally; select kills contribution when !use_key.
            float score = 0.0f;
            for (int i = 0; i < qk_per_thread; ++i) {{
                score += q[i] * static_cast<float>(tg_k[simd_lid * qk_per_thread + i]);
            }}
            score = simd_sum(score);
            {mask_score}
            // Blend to -inf when use_key==false — no branch in execution.
            score = metal::select(Limits<float>::finite_min, score, use_key);

            float new_max = metal::max(max_score, score);
            float factor = fast::exp(max_score - new_max);
            float exp_score = fast::exp(score - new_max);

            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int i = 0; i < v_per_thread; ++i) {{
                o[i] = o[i] * factor + exp_score * static_cast<float>(tg_v[simd_lid * v_per_thread + i]);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            k_ += blocks * int(k_seq_stride);
            v_ += blocks * int(v_seq_stride);
            {mask_advance}
        }}

        if (simd_lid == 0) {{
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }}
        for (int i = 0; i < v_per_thread; ++i) {{
            partials[i] = static_cast<InT>(o[i]);
        }}
    """

    return mx.fast.metal_kernel(
        name=f"batched_sdpa_2pass_partials{'_mask' if has_mask else ''}",
        input_names=inputs,
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def _make_batched_sdpa_2pass_reduce_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = V / BD;

        auto head_idx = threadgroup_position_in_grid.x;
        auto q_seq_idx = threadgroup_position_in_grid.y;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        auto q_offset = head_idx * M_FIXED + q_seq_idx;
        partials += (q_offset * blocks + simd_gid) * V + simd_lid * elem_per_thread;
        sums += q_offset * blocks;
        maxs += q_offset * blocks;
        out += q_offset * V + simd_gid * elem_per_thread;

        thread float o[elem_per_thread];
        threadgroup float outputs[BN * BD];

        for (int i = 0; i < elem_per_thread; ++i) {
            o[i] = 0.0f;
        }

        float sum_exp_score = 0.0f;
        float max_score = Limits<float>::finite_min;

        for (int b = 0; b < blocks / BN; ++b) {
            max_score = metal::max(max_score, maxs[simd_lid + BN * b]);
        }
        max_score = simd_max(max_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
            sum_exp_score += factor * sums[simd_lid + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_gid] - max_score);
            for (int i = 0; i < elem_per_thread; ++i) {
                o[i] += factor * static_cast<float>(partials[i]);
            }
            maxs += BN;
            partials += BN * V;
        }

        // Branchless reciprocal: avoid division-by-zero via max with epsilon.
        float inv_sum = 1.0f / metal::max(sum_exp_score, 1e-9f);

        for (int i = 0; i < elem_per_thread; ++i) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]) * inv_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; ++i) {
                out[i] = static_cast<InT>(o[i]);
            }
        }
    """

    return mx.fast.metal_kernel(
        name="batched_sdpa_2pass_reduce",
        input_names=["partials", "sums", "maxs", "blocks"],
        output_names=["out"],
        source=source,
    )


_batched_sdpa_2pass_partials_kernel = _make_batched_sdpa_2pass_partials_kernel(
    has_mask=False
)
_batched_sdpa_2pass_partials_kernel_masked = _make_batched_sdpa_2pass_partials_kernel(
    has_mask=True
)
_batched_sdpa_2pass_reduce_kernel = _make_batched_sdpa_2pass_reduce_kernel()


def batched_sdpa_2pass_exact(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    if not mx.metal.is_available():
        return None

    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None

    bsz, hq, q_len, d = queries.shape
    _, hk, n_kv, _ = keys.shape
    vdim = values.shape[-1]
    input_type = queries.dtype

    if q_len != 16:
        return None
    if input_type not in (mx.bfloat16, mx.float16):
        return None
    if d not in (128, 256) or vdim not in (128, 256) or d != vdim:
        return None
    if hk <= 0 or hq % hk != 0:
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)

    gqa_factor = hq // hk
    blocks = _compute_sdpa_2pass_blocks(gqa_factor, n_kv)
    if blocks <= 0 or blocks % 32 != 0:
        return None

    k_head_stride = keys.shape[2] * keys.shape[3]
    k_seq_stride = keys.shape[3]
    v_head_stride = values.shape[2] * values.shape[3]
    v_seq_stride = values.shape[3]

    mask_tensor = None
    kernel = _batched_sdpa_2pass_partials_kernel
    inputs = [
        queries,
        keys,
        values,
        gqa_factor,
        n_kv,
        k_head_stride,
        k_seq_stride,
        v_head_stride,
        v_seq_stride,
        float(scale),
        blocks,
    ]

    if mask is not None:
        if mask.dtype == mx.bool_:
            mask_tensor = mx.where(
                mask,
                mx.zeros(mask.shape, dtype=input_type),
                mx.full(mask.shape, mx.finfo(input_type).min, dtype=input_type),
            )
        else:
            mask_tensor = mask.astype(input_type) if mask.dtype != input_type else mask
        mask_tensor = mx.broadcast_to(mask_tensor, (bsz, hq, q_len, n_kv))
        mask_tensor = mx.contiguous(mask_tensor)
        kernel = _batched_sdpa_2pass_partials_kernel_masked
        inputs.append(mask_tensor)

    if kernel is None or _batched_sdpa_2pass_reduce_kernel is None:
        return None

    partial_shape = (bsz * hq, q_len, blocks, vdim)
    stats_shape = (bsz * hq, q_len, blocks)
    partials, sums, maxs = kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("D", d),
            ("V", vdim),
            ("Hk", hk),
            ("M_FIXED", q_len),
        ],
        grid=(hq * 32, bsz, blocks * q_len),
        threadgroup=(32, 1, q_len),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[input_type, mx.float32, mx.float32],
    )

    (out,) = _batched_sdpa_2pass_reduce_kernel(
        inputs=[partials, sums, maxs, blocks],
        template=[
            ("InT", input_type),
            ("V", vdim),
            ("M_FIXED", q_len),
        ],
        grid=((bsz * hq) * 1024, q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[input_type],
    )
    return out


def make_qwen3_no_cache_attn(
    q_proj,
    k_proj,
    v_proj,
    o_proj,
    q_norm,
    k_norm,
    rope,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    scale: float,
) -> Callable:
    """Return a compiled Qwen3 no-cache cross-attention function.

    Fuses Q/K/V projection, RMSNorm, RoPE, GQA head expansion, and SDPA into
    a single mx.compile trace. Verified by phew-mlx 0.1.6 to give ~1.2-1.35x
    vs the unfused path on M-series hardware.

    The returned callable has signature::

        fn(hidden_states, target_hidden, ctx_len) -> mx.array

    where ctx_len is a Python int treated as a compile-time constant; a
    different value triggers a retrace (acceptable for fixed-length inference).
    """
    rep = n_heads // n_kv_heads

    @mx.compile
    def _fwd(
        hidden_states: mx.array,  # (B, BL, D)
        target_hidden: mx.array,  # (B, CL, D)
        ctx_len: int,
    ) -> mx.array:
        B, BL, _ = hidden_states.shape
        CL = ctx_len

        # Q: project → reshape → RMSNorm → transpose → RoPE
        q = q_proj(hidden_states)
        q = q_norm(q.reshape(B, BL, n_heads, head_dim)).transpose(0, 2, 1, 3)
        q = rope(q, offset=CL)

        # Fused KV: single concat → two projections → norm/identity → transpose
        kv_in = mx.concatenate([target_hidden, hidden_states], axis=1)
        k_all = k_norm(
            k_proj(kv_in).reshape(B, CL + BL, n_kv_heads, head_dim)
        ).transpose(0, 2, 1, 3)
        v_all = v_proj(kv_in).reshape(B, CL + BL, n_kv_heads, head_dim).transpose(
            0, 2, 1, 3
        )

        # Split context / noise
        ck, nk = k_all[:, :, :CL, :], k_all[:, :, CL:, :]
        cv, nv = v_all[:, :, :CL, :], v_all[:, :, CL:, :]

        # RoPE on keys
        ck = rope(ck, offset=0)
        nk = rope(nk, offset=CL)

        # Concatenate and GQA head expansion
        keys = mx.concatenate([ck, nk], axis=2)
        values = mx.concatenate([cv, nv], axis=2)
        if rep > 1:
            keys = mx.repeat(keys, rep, axis=1)
            values = mx.repeat(values, rep, axis=1)

        # Flash attention + output projection
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, BL, -1)
        return o_proj(out)

    return _fwd

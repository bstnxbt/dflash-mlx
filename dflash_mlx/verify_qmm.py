# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import mlx.core as mx

from dflash_mlx.internal_debug import (
    verify_qmm_enabled as _debug_verify_qmm_enabled,
    verify_qmm_kparts as _debug_verify_qmm_kparts,
    verify_qmm_variant as _debug_verify_qmm_variant,
)

def is_enabled() -> bool:
    return _debug_verify_qmm_enabled()

def _variant() -> str:
    return _debug_verify_qmm_variant()

def _auto_variant(K: int, N: int) -> tuple[str, int]:
    if K >= 8192 or N <= 8192:
        return ("mma2big_pipe", 8)
    return ("mma2big", 1)

_VERIFY_KERNEL_CACHE: dict[tuple, object] = {}

def _build_kernel_mma2big(group_size: int, dtype: mx.Dtype):
    key = ("mma2big", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {group_size};

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;

        int sg_n_off = int(sg_id) * 16;

        for (int k0 = 0; k0 < K; k0 += BK) {{
            {{
                int n_global = n0 + dq_n_a;
                int k_base = k0 + dq_k_a * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b);
                }}
            }}
            {{
                int n_global = n0 + dq_n_b;
                int k_base = k0 + dq_k_b * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b);
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_matrix<T, 8, 8> c_tL_T, c_tR_T, c_bL_T, c_bR_T;
        c_tL_T.thread_elements()[0] = T(c_tL.thread_elements()[0]);
        c_tL_T.thread_elements()[1] = T(c_tL.thread_elements()[1]);
        c_tR_T.thread_elements()[0] = T(c_tR.thread_elements()[0]);
        c_tR_T.thread_elements()[1] = T(c_tR.thread_elements()[1]);
        c_bL_T.thread_elements()[0] = T(c_bL.thread_elements()[0]);
        c_bL_T.thread_elements()[1] = T(c_bL.thread_elements()[1]);
        c_bR_T.thread_elements()[0] = T(c_bR.thread_elements()[0]);
        c_bR_T.thread_elements()[1] = T(c_bR.thread_elements()[1]);
        simdgroup_store(c_tL_T, y + n0 + sg_n_off,                  N);
        simdgroup_store(c_tR_T, y + n0 + sg_n_off + 8,              N);
        simdgroup_store(c_bL_T, y + 8 * N + n0 + sg_n_off,          N);
        simdgroup_store(c_bR_T, y + 8 * N + n0 + sg_n_off + 8,      N);
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_mma2big_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_mma2big_8bit(group_size: int, dtype: mx.Dtype):
    key = ("mma2big_8bit", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {group_size};

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_4  = K / 4;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;

        threadgroup T B_tile[BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;

        int sg_n_off = int(sg_id) * 16;

        for (int k0 = 0; k0 < K; k0 += BK) {{
            {{
                int n_global = n0 + dq_n_a;
                int k_base   = k0 + dq_k_a * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];
                for (int ki = 0; ki < 4; ++ki)
                    B_tile[(dq_k_a * 8 + ki)     * BN + dq_n_a] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);
                for (int ki = 0; ki < 4; ++ki)
                    B_tile[(dq_k_a * 8 + 4 + ki) * BN + dq_n_a] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);
            }}
            {{
                int n_global = n0 + dq_n_b;
                int k_base   = k0 + dq_k_b * 8;
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];
                for (int ki = 0; ki < 4; ++ki)
                    B_tile[(dq_k_b * 8 + ki)     * BN + dq_n_b] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);
                for (int ki = 0; ki < 4; ++ki)
                    B_tile[(dq_k_b * 8 + 4 + ki) * BN + dq_n_b] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_matrix<T, 8, 8> c_tL_T, c_tR_T, c_bL_T, c_bR_T;
        c_tL_T.thread_elements()[0] = T(c_tL.thread_elements()[0]);
        c_tL_T.thread_elements()[1] = T(c_tL.thread_elements()[1]);
        c_tR_T.thread_elements()[0] = T(c_tR.thread_elements()[0]);
        c_tR_T.thread_elements()[1] = T(c_tR.thread_elements()[1]);
        c_bL_T.thread_elements()[0] = T(c_bL.thread_elements()[0]);
        c_bL_T.thread_elements()[1] = T(c_bL.thread_elements()[1]);
        c_bR_T.thread_elements()[0] = T(c_bR.thread_elements()[0]);
        c_bR_T.thread_elements()[1] = T(c_bR.thread_elements()[1]);
        simdgroup_store(c_tL_T, y + n0 + sg_n_off,             N);
        simdgroup_store(c_tR_T, y + n0 + sg_n_off + 8,         N);
        simdgroup_store(c_bL_T, y + 8 * N + n0 + sg_n_off,     N);
        simdgroup_store(c_bR_T, y + 8 * N + n0 + sg_n_off + 8, N);
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_mma2big_8bit_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_mma2big_pipe(group_size: int, dtype: mx.Dtype):
    key = ("mma2big_pipe", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {group_size};

        uint tid       = thread_position_in_threadgroup.x;
        uint sg_id     = tid / 32;
        uint tg_n      = threadgroup_position_in_grid.y;
        uint tg_k_part = threadgroup_position_in_grid.z;

        int K = int(K_size);
        int N = int(N_size);
        int KP = int(K_parts);
        int K_by_8  = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int k_slice = K / KP;
        int k_begin = k_slice * int(tg_k_part);
        int k_end   = k_begin + k_slice;

        threadgroup T B_tile[2][BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        #define STAGE_B(slot, k0_stage) {{                                              \\
            {{                                                                          \\
                int n_global = n0 + dq_n_a;                                             \\
                int k_base = (k0_stage) + dq_k_a * 8;                                   \\
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];               \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);            \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);            \\
                _Pragma("unroll")                                                       \\
                for (int ki = 0; ki < 8; ++ki) {{                                       \\
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;                         \\
                    B_tile[slot][(dq_k_a * 8 + ki) * BN + dq_n_a] = T(float(nib) * s + b); \\
                }}                                                                      \\
            }}                                                                          \\
            {{                                                                          \\
                int n_global = n0 + dq_n_b;                                             \\
                int k_base = (k0_stage) + dq_k_b * 8;                                   \\
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];               \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);            \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);            \\
                _Pragma("unroll")                                                       \\
                for (int ki = 0; ki < 8; ++ki) {{                                       \\
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;                         \\
                    B_tile[slot][(dq_k_b * 8 + ki) * BN + dq_n_b] = T(float(nib) * s + b); \\
                }}                                                                      \\
            }}                                                                          \\
        }}

        STAGE_B(0, k_begin);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int read_slot = 0;
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            int write_slot = 1 - read_slot;
            int k0_next = k0 + BK;

            if (k0_next < k_end) {{
                STAGE_B(write_slot, k0_next);
            }}

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            read_slot = write_slot;
        }}

        int part_off = int(tg_k_part) * BM * N;
        simdgroup_store(c_tL, partials + part_off + n0 + sg_n_off,                     N);
        simdgroup_store(c_tR, partials + part_off + n0 + sg_n_off + 8,                 N);
        simdgroup_store(c_bL, partials + part_off + 8 * N + n0 + sg_n_off,             N);
        simdgroup_store(c_bR, partials + part_off + 8 * N + n0 + sg_n_off + 8,         N);

        #undef STAGE_B
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_mma2big_pipe_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size", "K_parts"],
        output_names=["partials"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_mma2big_pipe_8bit(group_size: int, dtype: mx.Dtype):
    key = ("mma2big_pipe_8bit", group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int GS = {group_size};

        uint tid       = thread_position_in_threadgroup.x;
        uint sg_id     = tid / 32;
        uint tg_n      = threadgroup_position_in_grid.y;
        uint tg_k_part = threadgroup_position_in_grid.z;

        int K = int(K_size);
        int N = int(N_size);
        int KP = int(K_parts);
        int K_by_4  = K / 4;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int k_slice = K / KP;
        int k_begin = k_slice * int(tg_k_part);
        int k_end   = k_begin + k_slice;

        threadgroup T B_tile[2][BK * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int t_a = int(tid);
        int t_b = int(tid) + 64;
        int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
        int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
        int sg_n_off = int(sg_id) * 16;

        #define STAGE_B(slot, k0_stage) {{                                                                                  \\
            {{                                                                                                              \\
                int n_global = n0 + dq_n_a;                                                                                 \\
                int k_base   = (k0_stage) + dq_k_a * 8;                                                                     \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);                                                \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);                                                \\
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];                                                       \\
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];                                                   \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_a * 8 + ki)     * BN + dq_n_a] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);         \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_a * 8 + 4 + ki) * BN + dq_n_a] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);         \\
            }}                                                                                                              \\
            {{                                                                                                              \\
                int n_global = n0 + dq_n_b;                                                                                 \\
                int k_base   = (k0_stage) + dq_k_b * 8;                                                                     \\
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);                                                \\
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);                                                \\
                uint32_t p0 = w_q[n_global * K_by_4 + (k_base >> 2)];                                                       \\
                uint32_t p1 = w_q[n_global * K_by_4 + (k_base >> 2) + 1];                                                   \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_b * 8 + ki)     * BN + dq_n_b] = T(float((p0 >> (ki * 8)) & 0xFFu) * s + b);         \\
                _Pragma("unroll")                                                                                           \\
                for (int ki = 0; ki < 4; ++ki)                                                                              \\
                    B_tile[slot][(dq_k_b * 8 + 4 + ki) * BN + dq_n_b] = T(float((p1 >> (ki * 8)) & 0xFFu) * s + b);         \\
            }}                                                                                                              \\
        }}

        STAGE_B(0, k_begin);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int read_slot = 0;
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            int write_slot = 1 - read_slot;
            int k0_next = k0 + BK;

            if (k0_next < k_end) {{
                STAGE_B(write_slot, k0_next);
            }}

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB,                  K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB,          K);
                simdgroup_load(b_L, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off,         BN);
                simdgroup_load(b_R, B_tile[read_slot] + ks * BK_SUB * BN + sg_n_off + 8,     BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            read_slot = write_slot;
        }}

        int part_off = int(tg_k_part) * BM * N;
        simdgroup_store(c_tL, partials + part_off + n0 + sg_n_off,                     N);
        simdgroup_store(c_tR, partials + part_off + n0 + sg_n_off + 8,                 N);
        simdgroup_store(c_bL, partials + part_off + 8 * N + n0 + sg_n_off,             N);
        simdgroup_store(c_bR, partials + part_off + 8 * N + n0 + sg_n_off + 8,         N);

        #undef STAGE_B
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_mma2big_pipe_8bit_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size", "K_parts"],
        output_names=["partials"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _should_use_verify(
    x: mx.array,
    group_size: int,
    bits: int,
    transpose: bool,
) -> bool:
    if not is_enabled():
        return False
    if bits not in (4, 8) or group_size not in (32, 64, 128):
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if not transpose:
        return False
    m = 1
    for d in x.shape[:-1]:
        m *= d
    return m == 16

def verify_matmul(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    if not _should_use_verify(x, group_size, bits, transpose):
        return mx.quantized_matmul(
            x, w, scales=scales, biases=biases,
            transpose=transpose, group_size=group_size, bits=bits,
        )

    orig_shape = x.shape
    x2 = mx.contiguous(x.reshape(16, orig_shape[-1]))
    w_q = mx.contiguous(w)
    scales = mx.contiguous(scales)
    biases = mx.contiguous(biases)

    M = 16
    K = int(x2.shape[-1])
    N = int(w_q.shape[0])

    variant = _variant()
    auto_kp: int | None = None
    if variant == "auto":
        variant, auto_kp = _auto_variant(K, N)

    K_PARTS = auto_kp if auto_kp is not None else _debug_verify_qmm_kparts(4)

    if variant == "mma2big_pipe":
        if N % 32 != 0 or K % (32 * K_PARTS) != 0:
            return mx.quantized_matmul(
                x, w, scales=scales, biases=biases,
                transpose=transpose, group_size=group_size, bits=bits,
            )
        kernel = (
            _build_kernel_mma2big_pipe_8bit(group_size, x.dtype)
            if bits == 8 else
            _build_kernel_mma2big_pipe(group_size, x.dtype)
        )
        (partials,) = kernel(
            inputs=[x2, w_q, scales, biases, M, K, N, K_PARTS],
            template=[("T", x.dtype)],
            grid=(64, N // 32, K_PARTS),
            threadgroup=(64, 1, 1),
            output_shapes=[(K_PARTS, M, N)],
            output_dtypes=[mx.float32],
        )
        y = partials.sum(axis=0).astype(x.dtype)
        return y.reshape(*orig_shape[:-1], N)

    if N % 32 != 0 or K % 32 != 0:
        return mx.quantized_matmul(
            x, w, scales=scales, biases=biases,
            transpose=transpose, group_size=group_size, bits=bits,
        )
    kernel = (
        _build_kernel_mma2big_8bit(group_size, x.dtype)
        if bits == 8 else
        _build_kernel_mma2big(group_size, x.dtype)
    )
    (y,) = kernel(
        inputs=[x2, w_q, scales, biases, M, K, N],
        template=[("T", x.dtype)],
        grid=(64, N // 32, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*orig_shape[:-1], N)

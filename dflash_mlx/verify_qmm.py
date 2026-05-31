# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import platform

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

def _m4_ksplit_np_shape(K: int, N: int, bits: int) -> bool:
    return int(bits) == 4 and int(N) % 4 == 0 and int(K) % 32 == 0

def _m4_ksplit_np_kparts(N: int) -> int:
    return 2 if int(N) >= 4096 else 4

def _m16_ktmpl_variant(K: int, N: int, bits: int) -> str | None:
    if int(bits) != 4:
        return None
    if int(K) % 256 != 0 or int(N) % 16 != 0:
        return None
    if int(N) % 32 == 0 and _nax_verify_available():
        return "nax_ktmpl"
    if int(K) >= 8192 or int(N) <= 5120:
        return "combo_ktmpl"
    return "super_tree_fp16_ktmpl"

def _resolve_m16_ktmpl_variant(K: int, N: int, bits: int, variant: str | None = None) -> str | None:
    if int(bits) != 4 or int(K) % 256 != 0 or int(N) % 16 != 0:
        return None
    selected = _variant() if variant is None else str(variant)
    if selected == "auto":
        return _m16_ktmpl_variant(K, N, bits)
    if selected == "nax_ktmpl" and int(N) % 32 == 0 and _nax_verify_available():
        return selected
    if selected in ("combo_ktmpl", "super_tree_fp16_ktmpl"):
        return selected
    return None

def _nax_verify_available() -> bool:
    arch = str(mx.device_info().get("architecture", "")).lower()
    if not arch.startswith("applegpu_g17"):
        return False
    parts = platform.mac_ver()[0].split(".")
    try:
        major = int(parts[0]) if parts and parts[0] else 0
    except ValueError:
        major = 0
    try:
        minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    except ValueError:
        minor = 0
    return major > 26 or (major == 26 and minor >= 2)

def _build_kernel_m16_nax_ktmpl(k_val: int, group_size: int, dtype: mx.Dtype):
    key = ("m16_nax_ktmpl", int(k_val), group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        using namespace mpp::tensor_ops;

        constexpr int BM = 16;
        constexpr int BN = 32;
        constexpr int BK = 16;
        constexpr int NSG = 8;
        constexpr int GS = {group_size};
        constexpr int K = KCONST;
        constexpr int K_by_8 = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid = thread_position_in_threadgroup.x;
        uint sg_id = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;
        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        threadgroup T B_tile[NSG][BK * BN];
        threadgroup float partial[NSG][BM * BN];

        constexpr auto desc = matmul2d_descriptor(
            16,
            32,
            16,
            false,
            false,
            false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, metal::execution_simdgroup> op;

        tensor<device T, dextents<int, 2>, tensor_inline> A(
            (device T*)x,
            dextents<int, 2>{{K, BM}},
            array<int, 2>{{1, K}});
        tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
            B_tile[sg_id],
            dextents<int, 2>{{BN, BK}},
            array<int, 2>{{1, BN}});
        tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
            partial[sg_id],
            dextents<int, 2>{{BN, BM}},
            array<int, 2>{{1, BN}});

        auto ct_c = op.template get_destination_cooperative_tensor<
            tensor<device T, extents<int, 16, 16>, tensor_inline>,
            tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>,
            float>();
        _Pragma("unroll")
        for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {{
            ct_c[i] = 0.0f;
        }}

        int n_global = n0 + int(lane);
        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            uint32_t p0 = w_q[n_global * K_by_8 + ((k0 + 0) >> 3)];
            uint32_t p1 = w_q[n_global * K_by_8 + ((k0 + 8) >> 3)];
            float s0 = float(scales[n_global * K_by_gs + ((k0 + 0) / GS)]);
            float s1 = float(scales[n_global * K_by_gs + ((k0 + 8) / GS)]);
            float b0 = float(biases[n_global * K_by_gs + ((k0 + 0) / GS)]);
            float b1 = float(biases[n_global * K_by_gs + ((k0 + 8) / GS)]);

            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {{
                uint32_t nib = (p0 >> (ki * 4)) & 0xFu;
                B_tile[sg_id][ki * BN + int(lane)] = T(float(nib) * s0 + b0);
            }}
            _Pragma("unroll")
            for (int ki = 0; ki < 8; ++ki) {{
                uint32_t nib = (p1 >> (ki * 4)) & 0xFu;
                B_tile[sg_id][(8 + ki) * BN + int(lane)] = T(float(nib) * s1 + b1);
            }}
            simdgroup_barrier(mem_flags::mem_threadgroup);

            auto tA = A.template slice<16, 16>(k0, 0);
            auto tB = B.template slice<32, 16>(0, 0);
            op.run(tA, tB, ct_c);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        auto tC = C.template slice<32, 16>(0, 0);
        ct_c.store(tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * BN; off += NSG * 32) {{
            float acc = 0.0f;
            _Pragma("unroll")
            for (int g = 0; g < NSG; ++g) {{
                acc += partial[g][off];
            }}
            int row = off / BN;
            int col = off - row * BN;
            y[row * N + n0 + col] = T(acc);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_m16_nax_ktmpl_k{int(k_val)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        header="""
            #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        """,
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_m4_ksplit_np(
    group_size: int,
    dtype: mx.Dtype,
    *,
    k_parts: int = 4,
):
    key = ("m4_ksplit_np", group_size, dtype, int(k_parts))
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int M = 4;
        constexpr int BN = 4;
        constexpr int K_PARTS = {int(k_parts)};
        constexpr int GS = {group_size};

        uint part = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint tg_n = threadgroup_position_in_grid.y;

        int K = int(K_size);
        int N = int(N_size);
        int K_by_8 = K / 8;
        int K_by_gs = K / GS;
        int n0 = int(tg_n) * BN;
        int packs_per_part = K_by_8 / K_PARTS;
        int pack_start = int(part) * packs_per_part;
        int pack_end = (int(part) == K_PARTS - 1) ? K_by_8 : pack_start + packs_per_part;

        float acc[BN * M];
        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = 0.0f;
        }}

        using Vec8 = vec<T, 8>;
        const device Vec8 *xv = (const device Vec8*)x;

        for (int pack = pack_start + int(lane); pack < pack_end; pack += 32) {{
            int k_base = pack * 8;
            Vec8 v0 = xv[(0 * K + k_base) / 8];
            Vec8 v1 = xv[(1 * K + k_base) / 8];
            Vec8 v2 = xv[(2 * K + k_base) / 8];
            Vec8 v3 = xv[(3 * K + k_base) / 8];
            uint32_t p0 = w_q[(n0 + 0) * K_by_8 + pack];
            uint32_t p1 = w_q[(n0 + 1) * K_by_8 + pack];
            uint32_t p2 = w_q[(n0 + 2) * K_by_8 + pack];
            uint32_t p3 = w_q[(n0 + 3) * K_by_8 + pack];
            float s0 = float(scales[(n0 + 0) * K_by_gs + (k_base / GS)]);
            float s1 = float(scales[(n0 + 1) * K_by_gs + (k_base / GS)]);
            float s2 = float(scales[(n0 + 2) * K_by_gs + (k_base / GS)]);
            float s3 = float(scales[(n0 + 3) * K_by_gs + (k_base / GS)]);
            float b0 = float(biases[(n0 + 0) * K_by_gs + (k_base / GS)]);
            float b1 = float(biases[(n0 + 1) * K_by_gs + (k_base / GS)]);
            float b2 = float(biases[(n0 + 2) * K_by_gs + (k_base / GS)]);
            float b3 = float(biases[(n0 + 3) * K_by_gs + (k_base / GS)]);

            {{
                uint32_t packed = p0;
                float s = s0;
                float b = b0;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[0 * M + 0] += float(v0[ki]) * wv;
                    acc[0 * M + 1] += float(v1[ki]) * wv;
                    acc[0 * M + 2] += float(v2[ki]) * wv;
                    acc[0 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p1;
                float s = s1;
                float b = b1;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[1 * M + 0] += float(v0[ki]) * wv;
                    acc[1 * M + 1] += float(v1[ki]) * wv;
                    acc[1 * M + 2] += float(v2[ki]) * wv;
                    acc[1 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p2;
                float s = s2;
                float b = b2;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[2 * M + 0] += float(v0[ki]) * wv;
                    acc[2 * M + 1] += float(v1[ki]) * wv;
                    acc[2 * M + 2] += float(v2[ki]) * wv;
                    acc[2 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
            {{
                uint32_t packed = p3;
                float s = s3;
                float b = b3;
                for (int ki = 0; ki < 8; ++ki) {{
                    float wv = float((packed >> (ki * 4)) & 0xFu) * s + b;
                    acc[3 * M + 0] += float(v0[ki]) * wv;
                    acc[3 * M + 1] += float(v1[ki]) * wv;
                    acc[3 * M + 2] += float(v2[ki]) * wv;
                    acc[3 * M + 3] += float(v3[ki]) * wv;
                }}
            }}
        }}

        for (int i = 0; i < BN * M; ++i) {{
            acc[i] = simd_sum(acc[i]);
        }}

        threadgroup float partial[K_PARTS * BN * M];
        if (lane == 0) {{
            for (int i = 0; i < BN * M; ++i) {{
                partial[int(part) * BN * M + i] = acc[i];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (part == 0 && lane < BN * M) {{
            float total = 0.0f;
            for (int p = 0; p < K_PARTS; ++p) {{
                total += partial[p * BN * M + int(lane)];
            }}
            int j = int(lane) / M;
            int row = int(lane) - j * M;
            int n_global = n0 + j;
            if (n_global < N) {{
                y[row * N + n_global] = T(total);
            }}
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_m4_ksplit_np_kp{int(k_parts)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "K_size", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_m16_combo_ktmpl(k_val: int, group_size: int, dtype: mx.Dtype):
    key = ("m16_combo_ktmpl", int(k_val), group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 16;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int NSG = 8;
        constexpr int GS = {group_size};

        constexpr int K       = KCONST;
        constexpr int K_by_8  = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint lane  = tid % 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        threadgroup T B_tile[NSG][BK * BN];
        threadgroup float tg_partials[NSG][BM * BN];

        simdgroup_matrix<T, 8, 8> a_top, a_bot, b_L, b_R;
        simdgroup_matrix<float, 8, 8> c_tL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_tR = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bL = simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> c_bR = simdgroup_matrix<float, 8, 8>(0.0f);

        int dq_n = int(lane) % BN;
        int dq_k_lane = int(lane) / BN;

        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            _Pragma("unroll")
            for (int pack_idx = 0; pack_idx < 2; ++pack_idx) {{
                int dq_k = pack_idx * 2 + dq_k_lane;
                int n_global = n0 + dq_n;
                int k_base = k0 + dq_k * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] = T(float(nib) * s + b);
                }}
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top, x + k0 + ks * BK_SUB, K);
                simdgroup_load(a_bot, x + 8 * K + k0 + ks * BK_SUB, K);
                simdgroup_load(b_L, B_tile[sg_id] + ks * BK_SUB * BN, BN);
                simdgroup_load(b_R, B_tile[sg_id] + ks * BK_SUB * BN + 8, BN);
                simdgroup_multiply_accumulate(c_tL, a_top, b_L, c_tL);
                simdgroup_multiply_accumulate(c_tR, a_top, b_R, c_tR);
                simdgroup_multiply_accumulate(c_bL, a_bot, b_L, c_bL);
                simdgroup_multiply_accumulate(c_bR, a_bot, b_R, c_bR);
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_store(c_tL, tg_partials[sg_id], BN);
        simdgroup_store(c_tR, tg_partials[sg_id] + 8, BN);
        simdgroup_store(c_bL, tg_partials[sg_id] + 8 * BN, BN);
        simdgroup_store(c_bR, tg_partials[sg_id] + 8 * BN + 8, BN);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int off = int(tid); off < BM * BN; off += NSG * 32) {{
            float acc = 0.0f;
            _Pragma("unroll")
            for (int g = 0; g < NSG; ++g) {{
                acc += tg_partials[g][off];
            }}
            int row = off / BN;
            int col = off - row * BN;
            int n_global = n0 + col;
            y[row * N + n_global] = T(acc);
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_m16_combo_ktmpl_k{int(k_val)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

def _build_kernel_m16_super_tree_fp16_ktmpl(k_val: int, group_size: int, dtype: mx.Dtype):
    key = ("m16_super_tree_fp16_ktmpl", int(k_val), group_size, dtype)
    if key in _VERIFY_KERNEL_CACHE:
        return _VERIFY_KERNEL_CACHE[key]

    source = f"""
        using namespace metal;
        constexpr int BM = 16;
        constexpr int BN = 16;
        constexpr int BK = 32;
        constexpr int BK_SUB = 8;
        constexpr int NSG = 8;
        constexpr int GS = {group_size};

        constexpr int K       = KCONST;
        constexpr int K_by_8  = K / 8;
        constexpr int K_by_gs = K / GS;
        constexpr int K_chunk = K / NSG;

        uint tid   = thread_position_in_threadgroup.x;
        uint sg_id = tid / 32;
        uint lane  = tid % 32;
        uint tg_n  = threadgroup_position_in_grid.y;

        int N = int(N_size);
        int n0 = int(tg_n) * BN;
        int k_begin = int(sg_id) * K_chunk;
        int k_end = k_begin + K_chunk;

        threadgroup half B_tile[NSG][BK * BN];
        threadgroup half x_half[NSG][BM * BK];
        threadgroup half h_scratch[NSG][BM * BN];
        threadgroup float tg_partials[NSG][BM * BN];

        simdgroup_matrix<half, 8, 8> a_top_h, a_bot_h, b_L_h, b_R_h;
        simdgroup_matrix<half, 8, 8> c_tL_h = simdgroup_matrix<half, 8, 8>(half(0));
        simdgroup_matrix<half, 8, 8> c_tR_h = simdgroup_matrix<half, 8, 8>(half(0));
        simdgroup_matrix<half, 8, 8> c_bL_h = simdgroup_matrix<half, 8, 8>(half(0));
        simdgroup_matrix<half, 8, 8> c_bR_h = simdgroup_matrix<half, 8, 8>(half(0));

        int dq_n = int(lane) % BN;
        int dq_k_lane = int(lane) / BN;

        for (int k0 = k_begin; k0 < k_end; k0 += BK) {{
            _Pragma("unroll")
            for (int t = 0; t < BM * BK / 32; ++t) {{
                int slot = t * 32 + int(lane);
                int row = slot / BK;
                int kk = slot - row * BK;
                x_half[sg_id][row * BK + kk] = half(float(x[row * K + k0 + kk]));
            }}

            _Pragma("unroll")
            for (int pack_idx = 0; pack_idx < 2; ++pack_idx) {{
                int dq_k = pack_idx * 2 + dq_k_lane;
                int n_global = n0 + dq_n;
                int k_base = k0 + dq_k * 8;
                uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
                float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
                float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
                _Pragma("unroll")
                for (int ki = 0; ki < 8; ++ki) {{
                    uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                    B_tile[sg_id][(dq_k * 8 + ki) * BN + dq_n] = half(float(nib) * s + b);
                }}
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (int ks = 0; ks < BK / BK_SUB; ++ks) {{
                simdgroup_load(a_top_h, x_half[sg_id] + ks * BK_SUB, BK);
                simdgroup_load(a_bot_h, x_half[sg_id] + 8 * BK + ks * BK_SUB, BK);
                simdgroup_load(b_L_h, B_tile[sg_id] + ks * BK_SUB * BN, BN);
                simdgroup_load(b_R_h, B_tile[sg_id] + ks * BK_SUB * BN + 8, BN);
                simdgroup_multiply_accumulate(c_tL_h, a_top_h, b_L_h, c_tL_h);
                simdgroup_multiply_accumulate(c_tR_h, a_top_h, b_R_h, c_tR_h);
                simdgroup_multiply_accumulate(c_bL_h, a_bot_h, b_L_h, c_bL_h);
                simdgroup_multiply_accumulate(c_bR_h, a_bot_h, b_R_h, c_bR_h);
            }}

            simdgroup_barrier(mem_flags::mem_threadgroup);
        }}

        simdgroup_store(c_tL_h, h_scratch[sg_id], BN);
        simdgroup_store(c_tR_h, h_scratch[sg_id] + 8, BN);
        simdgroup_store(c_bL_h, h_scratch[sg_id] + 8 * BN, BN);
        simdgroup_store(c_bR_h, h_scratch[sg_id] + 8 * BN + 8, BN);

        simdgroup_barrier(mem_flags::mem_threadgroup);

        _Pragma("unroll")
        for (uint i = 0; i < BM * BN / 32; ++i) {{
            uint off = i * 32u + lane;
            tg_partials[sg_id][off] = float(h_scratch[sg_id][off]);
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((sg_id & 1u) == 0u) {{
            uint src_sg = sg_id + 1u;
            _Pragma("unroll")
            for (uint i = 0; i < BM * BN / 32; ++i) {{
                uint off = i * 32u + lane;
                tg_partials[sg_id][off] += tg_partials[src_sg][off];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((sg_id & 3u) == 0u) {{
            uint src_sg = sg_id + 2u;
            _Pragma("unroll")
            for (uint i = 0; i < BM * BN / 32; ++i) {{
                uint off = i * 32u + lane;
                tg_partials[sg_id][off] += tg_partials[src_sg][off];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {{
            _Pragma("unroll")
            for (uint i = 0; i < BM * BN / 32; ++i) {{
                uint off = i * 32u + lane;
                tg_partials[0][off] += tg_partials[4][off];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {{
            _Pragma("unroll")
            for (uint i = 0; i < BM * BN / 32; ++i) {{
                uint off = i * 32u + lane;
                int row = int(off) / BN;
                int col = int(off) - row * BN;
                int n_global = n0 + col;
                y[row * N + n_global] = T(tg_partials[0][off]);
            }}
        }}
    """

    dtype_tag = {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype, "unk")
    kernel = mx.fast.metal_kernel(
        name=f"verify_m16_super_tree_fp16_ktmpl_k{int(k_val)}_gs{group_size}_{dtype_tag}",
        input_names=["x", "w_q", "scales", "biases", "N_size"],
        output_names=["y"],
        source=source,
    )
    _VERIFY_KERNEL_CACHE[key] = kernel
    return kernel

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
    return m in (4, 16)

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
    m = 1
    for d in orig_shape[:-1]:
        m *= d
    x2 = mx.contiguous(x.reshape(m, orig_shape[-1]))
    w_q = mx.contiguous(w)
    scales = mx.contiguous(scales)
    biases = mx.contiguous(biases)

    M = int(m)
    K = int(x2.shape[-1])
    N = int(w_q.shape[0])

    if M == 4:
        if not _m4_ksplit_np_shape(K, N, bits):
            return mx.quantized_matmul(
                x, w, scales=scales, biases=biases,
                transpose=transpose, group_size=group_size, bits=bits,
            )
        k_parts = _m4_ksplit_np_kparts(N)
        kernel = _build_kernel_m4_ksplit_np(group_size, x.dtype, k_parts=k_parts)
        (y,) = kernel(
            inputs=[x2, w_q, scales, biases, K, N],
            template=[("T", x.dtype)],
            grid=(32 * k_parts, N // 4, 1),
            threadgroup=(32 * k_parts, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )
        return y.reshape(*orig_shape[:-1], N)

    variant = _variant()
    ktmpl_variant = _resolve_m16_ktmpl_variant(K, N, bits, variant)

    if ktmpl_variant is not None:
        if K % 256 != 0 or N % 16 != 0 or bits != 4:
            return mx.quantized_matmul(
                x, w, scales=scales, biases=biases,
                transpose=transpose, group_size=group_size, bits=bits,
            )
        if ktmpl_variant == "nax_ktmpl":
            kernel = _build_kernel_m16_nax_ktmpl(K, group_size, x.dtype)
            grid = (256, N // 32, 1)
        else:
            kernel = (
                _build_kernel_m16_combo_ktmpl(K, group_size, x.dtype)
                if ktmpl_variant == "combo_ktmpl" else
                _build_kernel_m16_super_tree_fp16_ktmpl(K, group_size, x.dtype)
            )
            grid = (256, N // 16, 1)
        (y,) = kernel(
            inputs=[x2, w_q, scales, biases, N],
            template=[("T", x.dtype), ("KCONST", K)],
            grid=grid,
            threadgroup=(256, 1, 1),
            output_shapes=[(M, N)],
            output_dtypes=[x.dtype],
        )
        return y.reshape(*orig_shape[:-1], N)

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

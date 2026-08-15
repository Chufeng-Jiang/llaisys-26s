#pragma once

#include "../../cuda/common.cuh"

#include "swiglu_cuda.hpp"
#include "../../../utils.hpp"
#include "../swiglu_config.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>

namespace llaisys::ops::cuda {
namespace {

// ============================================================
// Shared SwiGLU arithmetic
// ============================================================
//
// Preserve the FP32 evaluation order:
//
//     up * gate / (1 + exp(-gate))
//
// Do not rewrite this as:
//
//     up * (gate / denominator)
//
// Do not use __expf in the controlled baseline.
//
// NVIDIA and MetaX compile this exact arithmetic expression.
// ============================================================

__device__ __forceinline__ float swiglu_value(float gate_value, float up_value) {
    const float denominator = 1.0F + expf(-gate_value);

    return up_value * gate_value / denominator;
}

template <typename T> __device__ __forceinline__ T swiglu_element(T gate_value, T up_value) {
    const float gate_float = to_float<T>(gate_value);

    const float up_float = to_float<T>(up_value);

    return from_float<T>(swiglu_value(gate_float, up_float));
}

// ============================================================
// Shared scalar kernel
// ============================================================
//
// The grid-stride loop allows the same capped grid policy on all
// CUDA-compatible backends.
//
// Both operands are loaded before writing the result, preserving:
//
//     out == gate
//     out == up
// ============================================================

template <typename T>
__global__ void swiglu_scalar_kernel(
    T *__restrict__ out, const T *__restrict__ gate, const T *__restrict__ up, std::size_t numel) {
    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t grid_stride
        = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);

    for (std::size_t index = thread_index; index < numel; index += grid_stride) {
        const T gate_value = gate[index];
        const T up_value = up[index];

        out[index] = swiglu_element<T>(gate_value, up_value);
    }
}

// ============================================================
// Shared Packed128 kernel
// ============================================================
//
// Packed128 contains:
//
//     FP32 -> 4 elements
//     FP16 -> 8 elements
//     BF16 -> 8 elements
//
// Packing changes memory traversal only. Arithmetic remains FP32.
// The same eligibility condition and tail handling are used on
// NVIDIA and MetaX.
// ============================================================

template <typename T>
__global__ void swiglu_packed_kernel(
    T *__restrict__ out, const T *__restrict__ gate, const T *__restrict__ up, std::size_t numel) {
    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    static_assert(elements_per_pack > 0, "SwiGLU: Packed128 must contain at least one element.");

    const std::size_t pack_count = numel / elements_per_pack;

    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t grid_stride
        = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);

    const Packed128 *const gate_packs = reinterpret_cast<const Packed128 *>(gate);

    const Packed128 *const up_packs = reinterpret_cast<const Packed128 *>(up);

    Packed128 *const out_packs = reinterpret_cast<Packed128 *>(out);

    for (std::size_t pack_index = thread_index; pack_index < pack_count;
         pack_index += grid_stride) {
        // Load complete input packs before writing output.
        // This preserves exact in-place execution.
        const Packed128 gate_pack = gate_packs[pack_index];

        const Packed128 up_pack = up_packs[pack_index];

        Packed128 out_pack{};

        const T *const gate_values = reinterpret_cast<const T *>(&gate_pack);

        const T *const up_values = reinterpret_cast<const T *>(&up_pack);

        T *const out_values = reinterpret_cast<T *>(&out_pack);

#pragma unroll
        for (std::size_t lane = 0; lane < elements_per_pack; ++lane) {
            out_values[lane] = swiglu_element<T>(gate_values[lane], up_values[lane]);
        }

        out_packs[pack_index] = out_pack;
    }

    // ========================================================
    // Scalar tail
    // ========================================================

    const std::size_t tail_start = pack_count * elements_per_pack;

    for (std::size_t index = tail_start + thread_index; index < numel; index += grid_stride) {
        const T gate_value = gate[index];
        const T up_value = up[index];

        out[index] = swiglu_element<T>(gate_value, up_value);
    }
}

// ============================================================
// Shared Packed128 eligibility
// ============================================================

template <typename T>
inline bool can_use_packed_swiglu(const T *out, const T *gate, const T *up, std::size_t numel) {
    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    return numel >= elements_per_pack && are_aligned<PACKED_128_ALIGNMENT>(out, gate, up);
}

// ============================================================
// Shared logical work-item count
// ============================================================

template <typename T>
inline std::size_t get_swiglu_work_items(std::size_t numel, bool use_packed_kernel) {
    if (!use_packed_kernel) { return numel; }

    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    return div_ceil(numel, elements_per_pack);
}

// ============================================================
// Shared launch
// ============================================================

template <typename T>
void launch_swiglu(T *out, const T *gate, const T *up, std::size_t numel, cudaStream_t stream) {
    CHECK_ARGUMENT(numel == 0 || out != nullptr, "SwiGLU: output pointer must not be null.");

    CHECK_ARGUMENT(numel == 0 || gate != nullptr, "SwiGLU: gate pointer must not be null.");

    CHECK_ARGUMENT(numel == 0 || up != nullptr, "SwiGLU: up pointer must not be null.");

    if (numel == 0) { return; }

    const bool use_packed_kernel = can_use_packed_swiglu<T>(out, gate, up, numel);

    const std::size_t work_items = get_swiglu_work_items<T>(numel, use_packed_kernel);

    const std::size_t block_size = swiglu_config::block_size();

    CHECK_ARGUMENT(
        block_size > 0
            && block_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "SwiGLU: block size exceeds the supported launch range.");

    const std::size_t grid_size
        = get_capped_grid_size(work_items, block_size, swiglu_config::MAX_BLOCKS);

    CHECK_ARGUMENT(grid_size > 0, "SwiGLU: grid size must be greater than zero.");

    CHECK_ARGUMENT(
        grid_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "SwiGLU: grid size exceeds the supported launch range.");

    if (config::debug_enabled()) {
        std::fprintf(
            stderr,
            "[SwiGLU][%s] implementation=shared "
            "numel=%zu kernel=%s work_items=%zu "
            "block=%zu grid=%zu accumulation=f32\n",
            GPU_BACKEND_NAME, numel, use_packed_kernel ? "packed128" : "scalar", work_items,
            block_size, grid_size);
    }

    const dim3 grid_dimension(static_cast<unsigned int>(grid_size));

    const dim3 block_dimension(static_cast<unsigned int>(block_size));

    if (use_packed_kernel) {
        swiglu_packed_kernel<T>
            <<<grid_dimension, block_dimension, 0, stream>>>(out, gate, up, numel);
    } else {
        swiglu_scalar_kernel<T>
            <<<grid_dimension, block_dimension, 0, stream>>>(out, gate, up, numel);
    }

    check_kernel("SwiGLU kernel");
}

} // namespace

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    std::size_t numel,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    return dispatch_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_swiglu<T>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate),
            reinterpret_cast<const T *>(up), numel, cuda_stream);
    });
}

} // namespace llaisys::ops::cuda

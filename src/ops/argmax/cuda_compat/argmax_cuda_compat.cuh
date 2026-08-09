#pragma once

#include "../../cuda_compat/common.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace llaisys::ops::cuda_compat {

// ============================================================
// Argmax result semantics
// ============================================================
//
// UINT32_MAX is reserved as the invalid-result sentinel.
//
// With:
//
//     numel <= UINT32_MAX
//
// the largest valid input index is therefore:
//
//     UINT32_MAX - 1
// ============================================================

inline constexpr std::uint32_t ARGMAX_INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

struct ArgmaxResult {
    float value;
    std::uint32_t index;
};

static_assert(sizeof(std::int64_t) == 8, "Argmax requires a 64-bit output index.");

__host__ __device__ constexpr ArgmaxResult invalid_argmax_result() {
    return ArgmaxResult{0.0F, ARGMAX_INVALID_INDEX};
}

__host__ __device__ constexpr bool is_valid_argmax_result(const ArgmaxResult &result) {
    return result.index != ARGMAX_INVALID_INDEX;
}

// ============================================================
// Comparison semantics
// ============================================================
//
// Preserve the existing CPU/NVIDIA behavior:
//
// 1. NaN is considered greater than a non-NaN value.
// 2. Numerically greater values win.
// 3. Equal values use the smaller index.
//
// These rules belong to the operation semantics rather than
// any particular GPU backend.
// ============================================================

__device__ __forceinline__ bool is_better_argmax(
    float candidate_value,
    std::uint32_t candidate_index,
    float current_value,
    std::uint32_t current_index) {
    const bool candidate_is_nan = isnan(candidate_value);

    const bool current_is_nan = isnan(current_value);

    if (candidate_is_nan != current_is_nan) { return candidate_is_nan; }

    if (candidate_value > current_value) { return true; }

    if (candidate_value < current_value) { return false; }

    return candidate_index < current_index;
}

__device__ __forceinline__ void
update_argmax_result(ArgmaxResult &result, float candidate_value, std::uint32_t candidate_index) {
    if (!is_valid_argmax_result(result)
        || is_better_argmax(candidate_value, candidate_index, result.value, result.index)) {
        result.value = candidate_value;

        result.index = candidate_index;
    }
}

// ============================================================
// Portable shared-memory block reduction
// ============================================================
//
// This deliberately avoids:
//
//     __shfl_*
//     warp-size assumptions
//     atomicCAS
//
// The reduction also supports block sizes that are not exact
// powers of two.
// ============================================================

__device__ __forceinline__ ArgmaxResult
block_reduce_argmax_portable(ArgmaxResult thread_result, ArgmaxResult *shared_results) {
    const unsigned int thread_index = threadIdx.x;

    shared_results[thread_index] = thread_result;

    __syncthreads();

    unsigned int active_count = blockDim.x;

    while (active_count > 1) {
        const unsigned int next_count = (active_count + 1U) / 2U;

        const unsigned int pair_count = active_count / 2U;

        if (thread_index < pair_count) {
            const ArgmaxResult candidate = shared_results[thread_index + next_count];

            if (is_valid_argmax_result(candidate)) {
                update_argmax_result(
                    shared_results[thread_index], candidate.value, candidate.index);
            }
        }

        __syncthreads();

        active_count = next_count;
    }

    return shared_results[0];
}

// ============================================================
// CUDA-compatible portability baseline
// ============================================================
//
// A single block performs the full reduction.
//
// Threads use a block-stride loop, so arbitrarily large inputs
// can still be processed.
//
// This is intended as the portable correctness baseline.
// Vendor backends may provide optimized multi-block versions.
// ============================================================

template <typename T>
__global__ void argmax_portable_kernel(
    std::int64_t *__restrict__ max_idx,
    T *__restrict__ max_val,
    const T *__restrict__ vals,
    std::size_t numel) {
    extern __shared__ ArgmaxResult shared_results[];

    ArgmaxResult thread_result = invalid_argmax_result();

    for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < numel;
         index += static_cast<std::size_t>(blockDim.x)) {
        const float value = to_float<T>(vals[index]);

        update_argmax_result(thread_result, value, static_cast<std::uint32_t>(index));
    }

    const ArgmaxResult block_result = block_reduce_argmax_portable(thread_result, shared_results);

    if (threadIdx.x == 0 && is_valid_argmax_result(block_result)) {
        *max_idx = static_cast<std::int64_t>(block_result.index);

        *max_val = from_float<T>(block_result.value);
    }
}

// ============================================================
// CUDA-compatible launcher
// ============================================================
//
// The vendor adapter supplies:
//
//     block_size
//     stream
//
// The portable algorithm does not make assumptions about
// warp size or vendor-specific reduction primitives.
// ============================================================

template <typename T, typename StreamT>
inline void launch_argmax_portable(
    std::int64_t *max_idx,
    T *max_val,
    const T *vals,
    std::size_t numel,
    unsigned int block_size,
    StreamT stream) {
    if (numel == 0) { return; }

    const std::size_t shared_memory_bytes
        = static_cast<std::size_t>(block_size) * sizeof(ArgmaxResult);

    argmax_portable_kernel<T>
        <<<1, block_size, shared_memory_bytes, stream>>>(max_idx, max_val, vals, numel);
}

} // namespace llaisys::ops::cuda_compat
#pragma once

#include "../cuda_compat/argmax_cuda_compat.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llaisys::ops::metax::detail {

namespace cuda_compat = llaisys::ops::cuda_compat;

using cuda_compat::ArgmaxResult;
using cuda_compat::from_float;
using cuda_compat::invalid_argmax_result;
using cuda_compat::is_valid_argmax_result;
using cuda_compat::to_float;
using cuda_compat::update_argmax_result;

// ============================================================
// MetaX native execution model
// ============================================================
//
// Hardware/runtime probe:
//
//     warpSize               = 64
//     sizeof(__activemask()) = 8
//     full active mask       = 0xffffffffffffffff
//
// __shfl_down_sync with width=64 was verified to communicate
// across the lower/upper 32-lane halves.
//
// This is therefore deliberately MetaX-specific and must not
// be moved into cuda_compat.
// ============================================================

inline constexpr unsigned int METAX_ARGMAX_WARP_SIZE = 64;

// ============================================================
// Warp-level Argmax reduction
// ============================================================

__device__ __forceinline__ ArgmaxResult warp_reduce_argmax_shuffle64(ArgmaxResult result) {
    const auto active_mask = __activemask();

#pragma unroll
    for (int offset = static_cast<int>(METAX_ARGMAX_WARP_SIZE / 2); offset > 0; offset >>= 1) {
        const float other_value
            = __shfl_down_sync(active_mask, result.value, offset, METAX_ARGMAX_WARP_SIZE);

        const std::uint32_t other_index
            = __shfl_down_sync(active_mask, result.index, offset, METAX_ARGMAX_WARP_SIZE);

        if (other_index != cuda_compat::ARGMAX_INVALID_INDEX) {
            update_argmax_result(result, other_value, other_index);
        }
    }

    return result;
}

// ============================================================
// Two-level block reduction
//
// 256 threads on MetaX:
//
//     4 × 64-lane warps
//
// level 1:
//     each warp -> one ArgmaxResult
//
// level 2:
//     first warp reduces <= 4 warp results
//
// ============================================================

template <unsigned int BLOCK_SIZE>
__device__ __forceinline__ ArgmaxResult
block_reduce_argmax_shuffle64(ArgmaxResult thread_result, ArgmaxResult *warp_results) {
    static_assert(
        BLOCK_SIZE % METAX_ARGMAX_WARP_SIZE == 0,
        "MetaX shuffle Argmax requires block size "
        "to be a multiple of warpSize=64.");

    constexpr unsigned int NUM_WARPS = BLOCK_SIZE / METAX_ARGMAX_WARP_SIZE;

    const unsigned int lane_id = threadIdx.x & (METAX_ARGMAX_WARP_SIZE - 1U);

    const unsigned int warp_id = threadIdx.x / METAX_ARGMAX_WARP_SIZE;

    // --------------------------------------------------------
    // Level 1: one result per warp.
    // --------------------------------------------------------

    thread_result = warp_reduce_argmax_shuffle64(thread_result);

    if (lane_id == 0) { warp_results[warp_id] = thread_result; }

    __syncthreads();

    // --------------------------------------------------------
    // Level 2: first warp reduces warp leaders.
    //
    // All 64 lanes still participate in shuffle.
    // Lanes >= NUM_WARPS simply carry invalid results.
    // --------------------------------------------------------

    ArgmaxResult block_result = invalid_argmax_result();

    if (warp_id == 0) {
        if (lane_id < NUM_WARPS) { block_result = warp_results[lane_id]; }

        block_result = warp_reduce_argmax_shuffle64(block_result);
    }

    return block_result;
}

// ============================================================
// Stage 1
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
__global__ void argmax_multiblock_shuffle_stage1_kernel(
    ArgmaxResult *__restrict__ partial_results, const T *__restrict__ vals, std::size_t numel) {
    constexpr unsigned int NUM_WARPS = BLOCK_SIZE / METAX_ARGMAX_WARP_SIZE;

    __shared__ ArgmaxResult warp_results[NUM_WARPS];

    const std::size_t start
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(BLOCK_SIZE)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t stride
        = static_cast<std::size_t>(BLOCK_SIZE) * static_cast<std::size_t>(gridDim.x);

    ArgmaxResult thread_result = invalid_argmax_result();

    for (std::size_t index = start; index < numel; index += stride) {
        update_argmax_result(
            thread_result, to_float<T>(vals[index]), static_cast<std::uint32_t>(index));
    }

    const ArgmaxResult block_result
        = block_reduce_argmax_shuffle64<BLOCK_SIZE>(thread_result, warp_results);

    if (threadIdx.x == 0 && is_valid_argmax_result(block_result)) {
        partial_results[blockIdx.x] = block_result;
    }
}

// ============================================================
// Stage 2
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
__global__ void argmax_multiblock_shuffle_stage2_kernel(
    std::int64_t *__restrict__ max_idx,
    T *__restrict__ max_val,
    const ArgmaxResult *__restrict__ partial_results,
    std::size_t partial_count) {
    constexpr unsigned int NUM_WARPS = BLOCK_SIZE / METAX_ARGMAX_WARP_SIZE;

    __shared__ ArgmaxResult warp_results[NUM_WARPS];

    ArgmaxResult thread_result = invalid_argmax_result();

    for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < partial_count;
         index += static_cast<std::size_t>(BLOCK_SIZE)) {
        const ArgmaxResult candidate = partial_results[index];

        if (is_valid_argmax_result(candidate)) {
            update_argmax_result(thread_result, candidate.value, candidate.index);
        }
    }

    const ArgmaxResult final_result
        = block_reduce_argmax_shuffle64<BLOCK_SIZE>(thread_result, warp_results);

    if (threadIdx.x == 0 && is_valid_argmax_result(final_result)) {
        *max_idx = static_cast<std::int64_t>(final_result.index);

        *max_val = from_float<T>(final_result.value);
    }
}

// ============================================================
// Compile-time launcher
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
inline void launch_argmax_multiblock_shuffle_fixed(
    std::int64_t *max_idx,
    T *max_val,
    const T *vals,
    std::size_t numel,
    ArgmaxResult *partial_results,
    unsigned int grid_size,
    mcStream_t stream) {
    if (numel == 0 || grid_size == 0) { return; }

    argmax_multiblock_shuffle_stage1_kernel<T, BLOCK_SIZE>
        <<<grid_size, BLOCK_SIZE, 0, stream>>>(partial_results, vals, numel);

    MC_CHECK(mcGetLastError());

    argmax_multiblock_shuffle_stage2_kernel<T, BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(
        max_idx, max_val, partial_results, static_cast<std::size_t>(grid_size));

    MC_CHECK(mcGetLastError());
}

// ============================================================
// Runtime block-size dispatch
// ============================================================

template <typename T>
inline void launch_argmax_multiblock_shuffle(
    std::int64_t *max_idx,
    T *max_val,
    const T *vals,
    std::size_t numel,
    ArgmaxResult *partial_results,
    unsigned int block_size,
    unsigned int grid_size,
    mcStream_t stream) {
    switch (block_size) {
    case 64:
        return launch_argmax_multiblock_shuffle_fixed<T, 64>(
            max_idx, max_val, vals, numel, partial_results, grid_size, stream);

    case 128:
        return launch_argmax_multiblock_shuffle_fixed<T, 128>(
            max_idx, max_val, vals, numel, partial_results, grid_size, stream);

    case 256:
        return launch_argmax_multiblock_shuffle_fixed<T, 256>(
            max_idx, max_val, vals, numel, partial_results, grid_size, stream);

    default:
        throw std::invalid_argument(
            "MetaX shuffle Argmax requires "
            "block size 64, 128, or 256.");
    }
}

} // namespace llaisys::ops::metax::detail
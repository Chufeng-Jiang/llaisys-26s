#pragma once

#include "../cuda_compat/argmax_cuda_compat.cuh"
#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../cuda_compat/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::nvidia::detail {

namespace cuda_compat = llaisys::ops::cuda_compat;

using cuda_compat::ARGMAX_INVALID_INDEX;
using cuda_compat::ArgmaxResult;
using cuda_compat::invalid_argmax_result;
using cuda_compat::is_better_argmax;
using cuda_compat::is_valid_argmax_result;
using cuda_compat::update_argmax_result;

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::CUDA_MAX_WARPS_PER_BLOCK;
using llaisys::device::nvidia::CUDA_WARP_SIZE;
using llaisys::ops::cuda_compat::get_capped_grid_size;
using llaisys::device::nvidia::get_warp_aligned_block_size;

// ============================================================
// NVIDIA scheduling policy
// ============================================================
//
// Inputs no larger than this threshold are reduced using one
// CUDA block and require no global workspace.
// ============================================================

inline constexpr std::size_t NVIDIA_ARGMAX_SINGLE_BLOCK_THRESHOLD = 4096;

inline bool argmax_requires_workspace(std::size_t numel) {
    return numel > NVIDIA_ARGMAX_SINGLE_BLOCK_THRESHOLD;
}

static_assert(
    sizeof(unsigned long long) == 8, "Argmax requires unsigned long long to contain 64 bits.");

// ============================================================
// NVIDIA warp reduction
// ============================================================
//
// This is intentionally NVIDIA-specific:
//
//     CUDA warp size = 32
//     __shfl_down_sync()
//     full warp mask
// ============================================================

__device__ __forceinline__ ArgmaxResult warp_reduce_argmax(ArgmaxResult result) {
    constexpr unsigned int FULL_WARP_MASK = 0xFFFFFFFFU;

#pragma unroll
    for (int offset = static_cast<int>(CUDA_WARP_SIZE / 2); offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(FULL_WARP_MASK, result.value, offset);

        const std::uint32_t other_index = __shfl_down_sync(FULL_WARP_MASK, result.index, offset);

        if (other_index != ARGMAX_INVALID_INDEX) {
            update_argmax_result(result, other_value, other_index);
        }
    }

    return result;
}

// ============================================================
// NVIDIA block reduction
// ============================================================
//
// Stage 1:
//     reduce each warp using shuffle.
//
// Stage 2:
//     lane 0 of every warp writes into shared memory.
//
// Stage 3:
//     the first warp reduces all warp-level results.
// ============================================================

__device__ __forceinline__ ArgmaxResult
block_reduce_argmax(ArgmaxResult thread_result, ArgmaxResult *warp_results) {
    constexpr unsigned int warp_size = static_cast<unsigned int>(CUDA_WARP_SIZE);

    const unsigned int lane_id = threadIdx.x & (warp_size - 1U);

    const unsigned int warp_id = threadIdx.x / warp_size;

    const unsigned int num_warps = (blockDim.x + warp_size - 1U) / warp_size;

    thread_result = warp_reduce_argmax(thread_result);

    if (lane_id == 0) { warp_results[warp_id] = thread_result; }

    __syncthreads();

    ArgmaxResult block_result = invalid_argmax_result();

    if (warp_id == 0) {
        if (lane_id < num_warps) { block_result = warp_results[lane_id]; }

        block_result = warp_reduce_argmax(block_result);
    }

    return block_result;
}

// ============================================================
// Packed global result
// ============================================================
//
// High 32 bits:
//
//     raw FP32 value bits
//
// Low 32 bits:
//
//     uint32 index
//
// This permits value + index to be replaced atomically using
// one 64-bit atomicCAS operation.
// ============================================================

__device__ __forceinline__ unsigned long long pack_argmax_result(float value, std::uint32_t index) {
    const unsigned int value_bits = __float_as_uint(value);

    return (static_cast<unsigned long long>(value_bits) << 32)
         | static_cast<unsigned long long>(index);
}

__device__ __forceinline__ float unpack_argmax_value(unsigned long long packed) {
    const unsigned int value_bits = static_cast<unsigned int>(packed >> 32);

    return __uint_as_float(value_bits);
}

__device__ __forceinline__ std::uint32_t unpack_argmax_index(unsigned long long packed) {
    return static_cast<std::uint32_t>(packed & 0xFFFFFFFFULL);
}

// ============================================================
// Global atomic merge
// ============================================================

__device__ __forceinline__ void
atomic_update_argmax_result(unsigned long long *packed_workspace, const ArgmaxResult &candidate) {
    if (!is_valid_argmax_result(candidate)) { return; }

    // Atomic 64-bit read without changing the workspace.
    unsigned long long observed = atomicCAS(packed_workspace, 0ULL, 0ULL);

    while (true) {
        const float current_value = unpack_argmax_value(observed);

        const std::uint32_t current_index = unpack_argmax_index(observed);

        const bool current_valid = current_index != ARGMAX_INVALID_INDEX;

        if (current_valid
            && !is_better_argmax(candidate.value, candidate.index, current_value, current_index)) {
            return;
        }

        const unsigned long long desired = pack_argmax_result(candidate.value, candidate.index);

        const unsigned long long previous = atomicCAS(packed_workspace, observed, desired);

        if (previous == observed) { return; }

        observed = previous;
    }
}

// ============================================================
// NVIDIA single-block fast path
// ============================================================

template <typename T>
__global__ void argmax_single_block_kernel(
    std::int64_t *__restrict__ max_idx,
    T *__restrict__ max_val,
    const T *__restrict__ vals,
    std::size_t numel) {
    __shared__ ArgmaxResult warp_results[CUDA_MAX_WARPS_PER_BLOCK];

    ArgmaxResult thread_result = invalid_argmax_result();

    for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < numel;
         index += static_cast<std::size_t>(blockDim.x)) {
        const float value = cuda_compat::to_float<T>(vals[index]);

        update_argmax_result(thread_result, value, static_cast<std::uint32_t>(index));
    }

    const ArgmaxResult block_result = block_reduce_argmax(thread_result, warp_results);

    if (threadIdx.x == 0 && is_valid_argmax_result(block_result)) {
        *max_idx = static_cast<std::int64_t>(block_result.index);

        *max_val = cuda_compat::from_float<T>(block_result.value);
    }
}

// ============================================================
// Workspace initialization
// ============================================================

__global__ void initialize_argmax_workspace_kernel(unsigned long long *packed_workspace) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *packed_workspace = pack_argmax_result(0.0F, ARGMAX_INVALID_INDEX);
    }
}

// ============================================================
// NVIDIA multi-block reduction
// ============================================================

template <typename T>
__global__ void argmax_multi_block_kernel(
    unsigned long long *__restrict__ packed_workspace,
    const T *__restrict__ vals,
    std::size_t numel) {
    __shared__ ArgmaxResult warp_results[CUDA_MAX_WARPS_PER_BLOCK];

    const std::size_t start
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t stride
        = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    ArgmaxResult thread_result = invalid_argmax_result();

    for (std::size_t index = start; index < numel; index += stride) {
        const float value = cuda_compat::to_float<T>(vals[index]);

        update_argmax_result(thread_result, value, static_cast<std::uint32_t>(index));
    }

    const ArgmaxResult block_result = block_reduce_argmax(thread_result, warp_results);

    if (threadIdx.x == 0) { atomic_update_argmax_result(packed_workspace, block_result); }
}

// ============================================================
// Finalize packed result
// ============================================================

template <typename T>
__global__ void finalize_argmax_result_kernel(
    const unsigned long long *__restrict__ packed_workspace,
    std::int64_t *__restrict__ max_idx,
    T *__restrict__ max_val) {
    if (blockIdx.x != 0 || threadIdx.x != 0) { return; }

    const unsigned long long packed = *packed_workspace;

    const float value = unpack_argmax_value(packed);

    const std::uint32_t index = unpack_argmax_index(packed);

    if (index == ARGMAX_INVALID_INDEX) { return; }

    *max_idx = static_cast<std::int64_t>(index);

    *max_val = cuda_compat::from_float<T>(value);
}

// ============================================================
// NVIDIA optimized launcher
// ============================================================

template <typename T>
inline void launch_argmax_optimized(
    std::int64_t *max_idx,
    T *max_val,
    const T *vals,
    std::size_t numel,
    unsigned long long *packed_workspace,
    cudaStream_t stream) {
    // ========================================================
    // Small-input path
    // ========================================================

    if (numel <= NVIDIA_ARGMAX_SINGLE_BLOCK_THRESHOLD) {
        const unsigned int block_size = get_warp_aligned_block_size(numel);

        argmax_single_block_kernel<T><<<1, block_size, 0, stream>>>(max_idx, max_val, vals, numel);

        CUDA_CHECK(cudaGetLastError());

        return;
    }

    // ========================================================
    // Large-input path
    // ========================================================

    constexpr std::size_t block_size = CUDA_BLOCK_SIZE;

    const std::size_t grid_size = get_capped_grid_size(
    numel,
    block_size,
    llaisys::device::nvidia::CUDA_DEFAULT_MAX_GRID_SIZE);

    // Kernel 1:
    // reset reusable packed workspace.

    initialize_argmax_workspace_kernel<<<1, 1, 0, stream>>>(packed_workspace);

    CUDA_CHECK(cudaGetLastError());

    // Kernel 2:
    // multi-block local reduction + atomic global merge.

    argmax_multi_block_kernel<T>
        <<<static_cast<unsigned int>(grid_size), static_cast<unsigned int>(block_size), 0,
           stream>>>(packed_workspace, vals, numel);

    CUDA_CHECK(cudaGetLastError());

    // Kernel 3:
    // convert packed global result into requested outputs.

    finalize_argmax_result_kernel<T><<<1, 1, 0, stream>>>(packed_workspace, max_idx, max_val);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia::detail
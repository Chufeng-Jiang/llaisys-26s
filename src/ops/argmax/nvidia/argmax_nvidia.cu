#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"
#include "argmax_nvidia.cuh"

namespace {

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::CUDA_MAX_WARPS_PER_BLOCK;
using llaisys::device::nvidia::CUDA_WARP_SIZE;
using llaisys::device::nvidia::from_float;
using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::device::nvidia::get_warp_aligned_block_size;
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::to_cuda_stream;

// Inputs no larger than this threshold use one CUDA block and
// do not require the reusable packed workspace.
inline constexpr std::size_t SINGLE_BLOCK_THRESHOLD = 4096;

// UINT32_MAX is reserved as the invalid-result sentinel.
//
// Therefore, the largest valid index is UINT32_MAX - 1.
inline constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

// Argmax uses a 64-bit packed value for atomicCAS.
static_assert(sizeof(unsigned long long) == 8,
              "Argmax requires unsigned long long to contain 64 bits.");

// The output index is stored as int64.
static_assert(sizeof(std::int64_t) == 8,
              "Argmax requires a 64-bit index output.");

// ============================================================
// Argmax result
// ============================================================

struct MaxResult {
    float value;
    std::uint32_t index;
};

// An invalid result means that the current thread has not
// processed any valid input elements yet.
//
// The value field is only a placeholder. Validity is determined
// by the index field.
__host__ __device__ constexpr MaxResult invalid_result() {
    return MaxResult{
        0.0F,
        INVALID_INDEX,
    };
}

__host__ __device__ constexpr bool is_valid(const MaxResult &result) {
    return result.index != INVALID_INDEX;
}

// ============================================================
// Comparison
// ============================================================

// Comparison rules:
//
// 1. NaN is treated as greater than a non-NaN value.
// 2. A numerically greater value wins.
// 3. When values are equal, the smaller index wins.
//
// These rules keep NVIDIA behavior consistent with the CPU
// Argmax implementation.
__device__ __forceinline__ bool is_better(float candidate_value,
                                          std::uint32_t candidate_index,
                                          float current_value,
                                          std::uint32_t current_index) {
    const bool candidate_is_nan = isnan(candidate_value);

    const bool current_is_nan = isnan(current_value);

    if (candidate_is_nan != current_is_nan) {
        return candidate_is_nan;
    }

    if (candidate_value > current_value) {
        return true;
    }

    if (candidate_value < current_value) {
        return false;
    }

    return candidate_index < current_index;
}

// Update result when:
//
// 1. The current result is invalid.
// 2. The candidate is better according to is_better().
__device__ __forceinline__ void update_result(MaxResult &result,
                                              float candidate_value,
                                              std::uint32_t candidate_index) {
    if (!is_valid(result) || is_better(candidate_value, candidate_index, result.value, result.index)) {
        result.value = candidate_value;
        result.index = candidate_index;
    }
}

// ============================================================
// Warp reduction
// ============================================================

// Reduce one MaxResult per thread inside a CUDA warp.
//
// After completion, lane 0 contains the complete result for
// the current warp. Other lanes may contain partial results.
__device__ __forceinline__ MaxResult warp_reduce_argmax(MaxResult result) {
    // All supported block sizes are complete multiples of
    // CUDA_WARP_SIZE, so every lane participates.
    constexpr unsigned int FULL_WARP_MASK = 0xFFFFFFFFU;

#pragma unroll
    for (int offset = static_cast<int>(CUDA_WARP_SIZE / 2); offset > 0;
         offset >>= 1) {
        const float other_value = __shfl_down_sync(FULL_WARP_MASK, result.value, offset);

        const std::uint32_t other_index = __shfl_down_sync(FULL_WARP_MASK, result.index, offset);

        if (other_index != INVALID_INDEX) {
            update_result(result, other_value, other_index);
        }
    }

    return result;
}

// ============================================================
// Reusable block reduction
//
// This helper is used by both:
//
// - argmax_single_block_kernel
// - argmax_multi_block_kernel
//
// Every thread in the block must call this helper because it
// contains __syncthreads().
//
// After completion, thread 0 contains the complete block result.
// ============================================================

__device__ __forceinline__ MaxResult
block_reduce_argmax(MaxResult thread_result, MaxResult *warp_results) {
    constexpr unsigned int warp_size = static_cast<unsigned int>(CUDA_WARP_SIZE);

    const unsigned int lane_id = threadIdx.x & (warp_size - 1U);

    const unsigned int warp_id = threadIdx.x / warp_size;

    const unsigned int num_warps = (blockDim.x + warp_size - 1U) / warp_size;

    // First reduction level:
    // reduce the thread results inside each warp.
    thread_result = warp_reduce_argmax(thread_result);

    // Lane 0 of each warp writes that warp's result into
    // shared memory.
    if (lane_id == 0) {
        warp_results[warp_id] = thread_result;
    }

    // Ensure all warp leaders have written their results before
    // the first warp starts reading shared memory.
    __syncthreads();

    MaxResult block_result = invalid_result();

    // Second reduction level:
    // the first warp reduces all warp results.
    if (warp_id == 0) {
        if (lane_id < num_warps) {
            block_result = warp_results[lane_id];
        }

        block_result = warp_reduce_argmax(block_result);
    }

    return block_result;
}

// ============================================================
// Packed result helpers
//
// High 32 bits:
//     raw FP32 bit representation.
//
// Low 32 bits:
//     uint32 index.
//
// Packing the value and index into one 64-bit value allows them
// to be updated together using one atomicCAS operation.
// ============================================================

__device__ __forceinline__ unsigned long long pack_result(float value,
                                                          std::uint32_t index) {
    const unsigned int value_bits = __float_as_uint(value);

    return (static_cast<unsigned long long>(value_bits) << 32) | static_cast<unsigned long long>(index);
}

__device__ __forceinline__ float unpack_value(unsigned long long packed) {
    const unsigned int value_bits = static_cast<unsigned int>(packed >> 32);

    return __uint_as_float(value_bits);
}

__device__ __forceinline__ std::uint32_t unpack_index(
    unsigned long long packed) {
    return static_cast<std::uint32_t>(packed & 0xFFFFFFFFULL);
}

// ============================================================
// Global atomic merge
// ============================================================

// Atomically merge one block-level Argmax candidate into the
// global packed workspace.
__device__ __forceinline__ void atomic_update_result(
    unsigned long long *packed_workspace, const MaxResult &candidate) {
    if (!is_valid(candidate)) {
        return;
    }

    // Perform an atomic 64-bit read without changing the value.
    unsigned long long observed = atomicCAS(packed_workspace, 0ULL, 0ULL);

    while (true) {
        const float current_value = unpack_value(observed);

        const std::uint32_t current_index = unpack_index(observed);

        const bool current_valid = current_index != INVALID_INDEX;

        // The candidate cannot improve the current global result.
        if (current_valid && !is_better(candidate.value, candidate.index, current_value, current_index)) {
            return;
        }

        const unsigned long long desired = pack_result(candidate.value, candidate.index);

        const unsigned long long previous = atomicCAS(packed_workspace, observed, desired);

        // The workspace still contained observed, so this thread
        // successfully replaced it with desired.
        if (previous == observed) {
            return;
        }

        // Another block updated the workspace before this CAS.
        // Compare the candidate against the newer result.
        observed = previous;
    }
}

// ============================================================
// Small-input kernel
// ============================================================

// One CUDA block processes the entire input.
//
// This path directly writes the final output and does not use
// the reusable packed workspace.
template <typename T>
__global__ void argmax_single_block_kernel(std::int64_t *__restrict__ max_idx,
                                           T *__restrict__ max_val,
                                           const T *__restrict__ vals,
                                           std::size_t numel) {
    // One CUDA block contains at most 32 warps.
    __shared__ MaxResult warp_results[CUDA_MAX_WARPS_PER_BLOCK];

    MaxResult thread_result = invalid_result();

    // Block-stride loop:
    // every thread processes zero or more input elements.
    for (std::size_t i = threadIdx.x; i < numel; i += blockDim.x) {
        const float value = to_float(vals[i]);

        update_result(thread_result, value, static_cast<std::uint32_t>(i));
    }

    const MaxResult block_result = block_reduce_argmax(thread_result, warp_results);

    // Thread 0 contains the complete block result.
    if (threadIdx.x == 0 && is_valid(block_result)) {
        *max_idx = static_cast<std::int64_t>(block_result.index);

        *max_val = from_float<T>(block_result.value);
    }
}

// ============================================================
// Workspace initialization
// ============================================================

// Reset the reusable workspace before the multi-block reduction.
//
// INVALID_INDEX means that no block has submitted a valid result yet.
__global__ void initialize_argmax_workspace_kernel(
    unsigned long long *packed_workspace) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *packed_workspace = pack_result(0.0F, INVALID_INDEX);
    }
}

// ============================================================
// Large-input kernel
// ============================================================

// Multiple CUDA blocks process the input.
//
// Every block first performs a local reduction and then submits
// one result to the shared global workspace.
template <typename T>
__global__ void argmax_multi_block_kernel(
    unsigned long long *__restrict__ packed_workspace,
    const T *__restrict__ vals, std::size_t numel) {
    __shared__ MaxResult warp_results[CUDA_MAX_WARPS_PER_BLOCK];

    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    MaxResult thread_result = invalid_result();

    // Grid-stride loop:
    // every thread can process multiple input elements.
    for (std::size_t i = start; i < numel; i += stride) {
        const float value = to_float(vals[i]);

        update_result(thread_result, value, static_cast<std::uint32_t>(i));
    }

    const MaxResult block_result = block_reduce_argmax(thread_result, warp_results);

    // Submit only one global atomic update per block.
    if (threadIdx.x == 0) {
        atomic_update_result(packed_workspace, block_result);
    }
}

// ============================================================
// Final output kernel
// ============================================================

// Convert the packed global result into:
//
// - an int64 output index;
// - a maximum value using the requested output dtype.
template <typename T>
__global__ void finalize_argmax_result_kernel(
    const unsigned long long *__restrict__ packed_workspace,
    std::int64_t *__restrict__ max_idx, T *__restrict__ max_val) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const unsigned long long packed = *packed_workspace;

    const float value = unpack_value(packed);

    const std::uint32_t index = unpack_index(packed);

    if (index == INVALID_INDEX) {
        return;
    }

    *max_idx = static_cast<std::int64_t>(index);

    *max_val = from_float<T>(value);
}

// ============================================================
// Launcher
// ============================================================

template <typename T>
void launch_argmax_kernel(std::int64_t *max_idx, T *max_val, const T *vals,
                          std::size_t numel,
                          unsigned long long *packed_workspace,
                          cudaStream_t stream) {
    // Small-input fast path:
    //
    // - one block;
    // - one kernel;
    // - no packed workspace;
    // - no global atomic operation;
    // - no finalize kernel.
    if (numel <= SINGLE_BLOCK_THRESHOLD) {
        const unsigned int block_size = get_warp_aligned_block_size(numel);

        argmax_single_block_kernel<T>
            <<<1, block_size, 0, stream>>>(max_idx, max_val, vals, numel);

        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Large-input path.
    constexpr std::size_t block_size = CUDA_BLOCK_SIZE;

    const std::size_t grid_size = get_capped_grid_size(numel, block_size);

    const dim3 block_dimension(static_cast<unsigned int>(block_size));

    const dim3 grid_dimension(static_cast<unsigned int>(grid_size));

    // Kernel 1:
    // initialize the reusable packed workspace.
    initialize_argmax_workspace_kernel<<<1, 1, 0, stream>>>(packed_workspace);

    CUDA_CHECK(cudaGetLastError());

    // Kernel 2:
    // run the multi-block reduction.
    argmax_multi_block_kernel<T><<<grid_dimension, block_dimension, 0, stream>>>(
        packed_workspace, vals, numel);

    CUDA_CHECK(cudaGetLastError());

    // Kernel 3:
    // write the packed result into the requested output tensors.
    finalize_argmax_result_kernel<T>
        <<<1, 1, 0, stream>>>(packed_workspace, max_idx, max_val);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// ============================================================
// Public NVIDIA backend interface
// ============================================================

namespace llaisys::ops::nvidia {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, std::size_t numel,
            unsigned long long *packed_workspace, llaisysStream_t stream) {
    CHECK_ARGUMENT(max_idx != nullptr,
                   "Argmax: max_idx pointer must not be null.");

    CHECK_ARGUMENT(max_val != nullptr,
                   "Argmax: max_val pointer must not be null.");

    CHECK_ARGUMENT(vals != nullptr, "Argmax: vals pointer must not be null.");

    CHECK_ARGUMENT(numel > 0, "Argmax: input tensor must not be empty.");

    CHECK_ARGUMENT(numel <= static_cast<std::size_t>(
                       std::numeric_limits<std::uint32_t>::max()),
                   "Argmax: NVIDIA implementation supports at most "
                   "UINT32_MAX elements.");

    // Small inputs do not use the workspace.
    CHECK_ARGUMENT(numel <= SINGLE_BLOCK_THRESHOLD || packed_workspace != nullptr,
                   "Argmax: NVIDIA packed workspace must not be null.");

    const cudaStream_t cuda_stream = llaisys::device::nvidia::to_cuda_stream(stream);

    return llaisys::device::nvidia::dispatch_cuda_dtype(
        type,
        [&](auto tag) {
            using T = typename decltype(tag)::type;

            return launch_argmax_kernel<T>(
                reinterpret_cast<std::int64_t *>(max_idx),
                reinterpret_cast<T *>(max_val),
                reinterpret_cast<const T *>(vals),
                numel,
                packed_workspace,
                cuda_stream
            );
        }
    );
}

} // namespace llaisys::ops::nvidia
#include "argmax_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::div_ceil;

// Large-input kernels use grid-stride loops, so the grid
// does not need to grow without limit.
inline constexpr std::size_t MAX_GRID_SIZE = 4096;

// Small inputs are completed by one block and one kernel.
inline constexpr std::size_t SINGLE_BLOCK_THRESHOLD = 4096;

inline constexpr unsigned int WARP_SIZE = 32;

// Reserved as the invalid-result sentinel.
//
// Therefore, the largest valid index is UINT32_MAX - 1.
inline constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

static_assert(CUDA_BLOCK_SIZE % WARP_SIZE == 0,
              "CUDA_BLOCK_SIZE must be divisible by 32.");

static_assert(CUDA_BLOCK_SIZE <= 1024, "CUDA_BLOCK_SIZE must not exceed 1024.");

static_assert(sizeof(llaisys::fp16_t) == sizeof(half),
              "llaisys::fp16_t and CUDA half must have the same size.");

static_assert(
    sizeof(llaisys::bf16_t) == sizeof(__nv_bfloat16),
    "llaisys::bf16_t and CUDA __nv_bfloat16 must have the same size.");

static_assert(sizeof(std::int64_t) == sizeof(unsigned long long),
              "Argmax requires a 64-bit index output.");

template <typename>
inline constexpr bool ALWAYS_FALSE = false;

// ============================================================
// Argmax result
// ============================================================

struct MaxResult {
    float value;
    std::uint32_t index;
};

__host__ __device__ constexpr MaxResult invalid_result() {
    return MaxResult{0.0F, INVALID_INDEX};
}

__host__ __device__ constexpr bool is_valid(const MaxResult &result) {
    return result.index != INVALID_INDEX;
}

// ============================================================
// Type conversion
// ============================================================

template <typename T>
__device__ __forceinline__ float to_accumulator(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;

    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(value);

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);

    } else {
        static_assert(ALWAYS_FALSE<T>, "Unsupported NVIDIA Argmax input type.");
    }
}

template <typename T>
__device__ __forceinline__ T from_accumulator(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;

    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(value);

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(value);

    } else {
        static_assert(ALWAYS_FALSE<T>, "Unsupported NVIDIA Argmax output type.");
    }
}

// ============================================================
// Comparison
// ============================================================

// NaN is treated as greater than a non-NaN value.
//
// When values are equal, the smaller index is selected.
// This makes CUDA behavior consistent with the CPU version.
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

__device__ __forceinline__ MaxResult warp_reduce_argmax(MaxResult result) {
    // All supported block sizes are multiples of 32.
    constexpr unsigned int mask = 0xFFFFFFFFU;

#pragma unroll
    for (int offset = static_cast<int>(WARP_SIZE / 2); offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(mask, result.value, offset);

        const std::uint32_t other_index = __shfl_down_sync(mask, result.index, offset);

        if (other_index != INVALID_INDEX) {
            update_result(result, other_value, other_index);
        }
    }

    return result;
}

// ============================================================
// Reusable block reduction
// ============================================================

// This helper is used by both:
// - argmax_single_block_kernel;
// - argmax_multi_block_kernel.
//
// All threads in the block must call this function because it
// contains __syncthreads().
//
// After completion, thread 0 contains the block result.
__device__ __forceinline__ MaxResult
block_reduce_argmax(MaxResult thread_result, MaxResult *warp_results) {
    const unsigned int lane_id = threadIdx.x & (WARP_SIZE - 1);

    const unsigned int warp_id = threadIdx.x / WARP_SIZE;

    const unsigned int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // First level: reduce inside each warp.
    thread_result = warp_reduce_argmax(thread_result);

    // Each warp leader writes one result.
    if (lane_id == 0) {
        warp_results[warp_id] = thread_result;
    }

    __syncthreads();

    MaxResult block_result = invalid_result();

    // Second level: the first warp reduces all warp results.
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
// ============================================================

// High 32 bits: raw FP32 bits.
// Low 32 bits: uint32 index.
__device__ __forceinline__ unsigned long long pack_result(float value,
                                                          std::uint32_t index) {
    const unsigned int value_bits = __float_as_uint(value);

    return (static_cast<unsigned long long>(value_bits) << 32) | static_cast<unsigned long long>(index);
}

__device__ __forceinline__ float unpack_value(unsigned long long packed) {
    const unsigned int value_bits = static_cast<unsigned int>(packed >> 32);

    return __uint_as_float(value_bits);
}

__device__ __forceinline__ std::uint32_t
unpack_index(unsigned long long packed) {
    return static_cast<std::uint32_t>(packed & 0xFFFFFFFFULL);
}

// ============================================================
// Global atomic merge
// ============================================================

__device__ __forceinline__ void
atomic_update_result(unsigned long long *packed_workspace,
                     const MaxResult &candidate) {
    if (!is_valid(candidate)) {
        return;
    }

    // Atomic 64-bit read.
    unsigned long long observed = atomicCAS(packed_workspace, 0ULL, 0ULL);

    while (true) {
        const float current_value = unpack_value(observed);

        const std::uint32_t current_index = unpack_index(observed);

        const bool current_valid = current_index != INVALID_INDEX;

        if (current_valid && !is_better(candidate.value, candidate.index, current_value, current_index)) {
            return;
        }

        const unsigned long long desired = pack_result(candidate.value, candidate.index);

        const unsigned long long previous = atomicCAS(packed_workspace, observed, desired);

        if (previous == observed) {
            return;
        }

        // Another block updated the workspace.
        observed = previous;
    }
}

// ============================================================
// Small-input kernel
// ============================================================

template <typename T>
__global__ void argmax_single_block_kernel(std::int64_t *__restrict__ max_idx,
                                           T *__restrict__ max_val,
                                           const T *__restrict__ vals,
                                           std::size_t numel) {
    // Maximum CUDA block size is 1024 threads:
    // 1024 / 32 = 32 warps.
    __shared__ MaxResult warp_results[32];

    MaxResult thread_result = invalid_result();

    for (std::size_t i = threadIdx.x; i < numel; i += blockDim.x) {
        const float value = to_accumulator(vals[i]);

        update_result(thread_result, value, static_cast<std::uint32_t>(i));
    }

    const MaxResult block_result = block_reduce_argmax(thread_result, warp_results);

    if (threadIdx.x == 0 && is_valid(block_result)) {
        *max_idx = static_cast<std::int64_t>(block_result.index);

        *max_val = from_accumulator<T>(block_result.value);
    }
}

// ============================================================
// Workspace initialization
// ============================================================

__global__ void
initialize_argmax_workspace_kernel(unsigned long long *packed_workspace) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *packed_workspace = pack_result(0.0F, INVALID_INDEX);
    }
}

// ============================================================
// Large-input kernel
// ============================================================

template <typename T>
__global__ void
argmax_multi_block_kernel(unsigned long long *__restrict__ packed_workspace,
                          const T *__restrict__ vals, std::size_t numel) {
    __shared__ MaxResult warp_results[32];

    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    MaxResult thread_result = invalid_result();

    for (std::size_t i = start; i < numel; i += stride) {
        const float value = to_accumulator(vals[i]);

        update_result(thread_result, value, static_cast<std::uint32_t>(i));
    }

    const MaxResult block_result = block_reduce_argmax(thread_result, warp_results);

    // Only one global atomic update per block.
    if (threadIdx.x == 0) {
        atomic_update_result(packed_workspace, block_result);
    }
}

// ============================================================
// Final output kernel
// ============================================================

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

    *max_val = from_accumulator<T>(value);
}

// ============================================================
// Launch configuration
// ============================================================

unsigned int get_single_block_size(std::size_t numel) {
    const std::size_t rounded_size = div_ceil(numel, static_cast<std::size_t>(WARP_SIZE)) * WARP_SIZE;

    const std::size_t block_size = std::min(static_cast<std::size_t>(CUDA_BLOCK_SIZE), std::max(static_cast<std::size_t>(WARP_SIZE), rounded_size));

    return static_cast<unsigned int>(block_size);
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
    // one block, one kernel, no workspace needed.
    if (numel <= SINGLE_BLOCK_THRESHOLD) {
        const unsigned int block_size = get_single_block_size(numel);

        argmax_single_block_kernel<T>
            <<<1, block_size, 0, stream>>>(max_idx, max_val, vals, numel);

        CUDA_CHECK(cudaGetLastError());
        return;
    }

    constexpr std::size_t block_size = CUDA_BLOCK_SIZE;

    const std::size_t required_blocks = div_ceil(numel, block_size);

    const std::size_t grid_size = std::min(required_blocks, MAX_GRID_SIZE);

    const dim3 block_dim(static_cast<unsigned int>(block_size));

    const dim3 grid_dim(static_cast<unsigned int>(grid_size));

    // Kernel 1: initialize reusable workspace.
    initialize_argmax_workspace_kernel<<<1, 1, 0, stream>>>(packed_workspace);

    CUDA_CHECK(cudaGetLastError());

    // Kernel 2: parallel multi-block reduction.
    argmax_multi_block_kernel<T>
        <<<grid_dim, block_dim, 0, stream>>>(packed_workspace, vals, numel);

    CUDA_CHECK(cudaGetLastError());

    // Kernel 3: convert packed result to output dtype.
    finalize_argmax_result_kernel<T>
        <<<1, 1, 0, stream>>>(packed_workspace, max_idx, max_val);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace

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

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_argmax_kernel<float>(
            reinterpret_cast<std::int64_t *>(max_idx),
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals), numel, packed_workspace,
            cuda_stream);

    case LLAISYS_DTYPE_F16:
        return launch_argmax_kernel<half>(reinterpret_cast<std::int64_t *>(max_idx),
                                          reinterpret_cast<half *>(max_val),
                                          reinterpret_cast<const half *>(vals),
                                          numel, packed_workspace, cuda_stream);

    case LLAISYS_DTYPE_BF16:
        return launch_argmax_kernel<__nv_bfloat16>(
            reinterpret_cast<std::int64_t *>(max_idx),
            reinterpret_cast<__nv_bfloat16 *>(max_val),
            reinterpret_cast<const __nv_bfloat16 *>(vals), numel, packed_workspace,
            cuda_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
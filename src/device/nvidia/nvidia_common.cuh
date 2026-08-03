#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace llaisys::device::nvidia {

// ============================================================
// CUDA execution constants
// ============================================================

// Number of threads in one CUDA warp.
inline constexpr std::size_t CUDA_WARP_SIZE = 32;

// Default number of threads used by CUDA kernels.
inline constexpr std::size_t CUDA_BLOCK_SIZE = 256;

// CUDA currently supports at most 1024 threads in one block.
inline constexpr std::size_t CUDA_MAX_THREADS_PER_BLOCK = 1024;

// Maximum possible number of warps in one CUDA block.
inline constexpr std::size_t CUDA_MAX_WARPS_PER_BLOCK = CUDA_MAX_THREADS_PER_BLOCK / CUDA_WARP_SIZE;

// Software grid-size cap used by grid-stride kernels.
//
// This is not the CUDA hardware grid limit. It prevents kernels from
// launching an unnecessarily large number of blocks.
inline constexpr std::size_t CUDA_DEFAULT_MAX_GRID_SIZE = 4096;

static_assert(CUDA_BLOCK_SIZE % CUDA_WARP_SIZE == 0,
              "CUDA_BLOCK_SIZE must be divisible by CUDA_WARP_SIZE.");

static_assert(CUDA_BLOCK_SIZE <= CUDA_MAX_THREADS_PER_BLOCK,
              "CUDA_BLOCK_SIZE exceeds the CUDA threads-per-block limit.");

// ============================================================
// Template utilities
// ============================================================

// Used in unsupported branches of if constexpr.
//
// Unlike a normal false value, this expression depends on T, so the
// static_assert is evaluated only when that branch is instantiated.
template <typename>
inline constexpr bool DEPENDENT_FALSE = false;

// ============================================================
// Integer helpers
// ============================================================

// Integer ceiling division.
//
// Examples:
// div_ceil(10, 4) == 3
// div_ceil(8, 4)  == 2
//
// The divisor must be greater than zero.
__host__ __device__ constexpr std::size_t div_ceil(std::size_t value,
                                                   std::size_t divisor) {
    return value / divisor + static_cast<std::size_t>(value % divisor != 0);
}

// Return a block size that:
// 1. is at least one warp;
// 2. is a multiple of the warp size;
// 3. does not exceed CUDA_BLOCK_SIZE.
inline unsigned int get_warp_aligned_block_size(std::size_t work_items) {
    std::size_t block_size = div_ceil(work_items, CUDA_WARP_SIZE) * CUDA_WARP_SIZE;

    if (block_size < CUDA_WARP_SIZE) {
        block_size = CUDA_WARP_SIZE;
    }

    if (block_size > CUDA_BLOCK_SIZE) {
        block_size = CUDA_BLOCK_SIZE;
    }

    return static_cast<unsigned int>(block_size);
}

// Calculate the number of blocks required by a grid-stride kernel,
// capped by an operator-selected maximum grid size.
inline std::size_t get_capped_grid_size(
    std::size_t work_items, std::size_t block_size,
    std::size_t max_grid_size = CUDA_DEFAULT_MAX_GRID_SIZE) {
    const std::size_t required_blocks = div_ceil(work_items, block_size);

    return required_blocks < max_grid_size ? required_blocks : max_grid_size;
}

// ============================================================
// CUDA data-type conversion
// ============================================================

// Convert a supported CUDA scalar type to FP32.
//
// This helper is useful for operations that compare or accumulate
// FP16 and BF16 values using FP32.
template <typename T>
__device__ __forceinline__ float to_float(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;

    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(value);

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);

    } else {
        static_assert(DEPENDENT_FALSE<T>,
                      "Unsupported CUDA type for conversion to float.");
    }
}

// Convert an FP32 result back to a supported CUDA scalar type.
template <typename T>
__device__ __forceinline__ T from_float(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;

    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half(value);

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(value);

    } else {
        static_assert(DEPENDENT_FALSE<T>,
                      "Unsupported CUDA type for conversion from float.");
    }
}

// ============================================================
// Address-alignment helpers
// ============================================================

// Return true when a pointer satisfies the requested byte alignment.
template <std::size_t Alignment, typename T>
inline bool is_aligned(const T *pointer) {
    static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
                  "Alignment must be a nonzero power of two.");

    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);

    return address % Alignment == 0;
}

// Check multiple pointers against the same alignment requirement.
//
// Example:
// are_aligned<16>(out, input_a, input_b)
template <std::size_t Alignment, typename... PointerTypes>
inline bool are_aligned(PointerTypes... pointers) {
    return (is_aligned<Alignment>(pointers) && ...);
}

// ============================================================
// 128-bit raw-memory pack
// ============================================================

// CUDA uint4 occupies 128 bits and requires 16-byte alignment.
//
// It is suitable for raw copies such as Embedding, where the bit pattern
// should be transferred without numerical conversion.
using Packed128 = uint4;

inline constexpr std::size_t PACKED_128_BYTES = sizeof(Packed128);
inline constexpr std::size_t PACKED_128_ALIGNMENT = alignof(Packed128);

static_assert(PACKED_128_BYTES == 16,
              "Packed128 must occupy exactly 16 bytes.");

static_assert(PACKED_128_ALIGNMENT == 16,
              "Packed128 must require 16-byte alignment.");

// Number of T elements contained in one 128-bit pack.
//
// float         -> 4 elements
// half          -> 8 elements
// __nv_bfloat16 -> 8 elements
template <typename T>
inline constexpr std::size_t PACKED_128_ELEMENTS = PACKED_128_BYTES / sizeof(T);

// ============================================================
// CUDA Runtime error handling
// ============================================================

inline void check_cuda(cudaError_t error, const char *expression,
                       const char *file, int line, const char *function) {
    if (error == cudaSuccess) {
        return;
    }

    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error) + " while executing " + expression + " in " + function + " at " + file + ":" + std::to_string(line));
}

} // namespace llaisys::device::nvidia

#define CUDA_CHECK(CALL) ::llaisys::device::nvidia::check_cuda((CALL), #CALL, __FILE__, __LINE__, __func__)
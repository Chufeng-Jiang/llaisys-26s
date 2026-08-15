#pragma once

#include "../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace llaisys::ops::cuda {

// ============================================================
// CUDA-compatible backend identification
// ============================================================
//
// This layer intentionally uses the CUDA-compatible API surface
// for both NVIDIA CUDA and MetaX MACA builds.
//
// The backend compiler/runtime provides the concrete implementation.
// Exactly one GPU backend is expected in one build.
// ============================================================

#if defined(ENABLE_NVIDIA_API) && defined(ENABLE_METAX_API)
#error "The shared CUDA-compatible operator path currently expects one GPU backend per build."
#endif

#if defined(ENABLE_METAX_API)
inline constexpr const char *GPU_BACKEND_NAME = "MetaX";
#elif defined(ENABLE_NVIDIA_API)
inline constexpr const char *GPU_BACKEND_NAME = "NVIDIA";
#else
inline constexpr const char *GPU_BACKEND_NAME = "CUDA-compatible";
#endif

// ============================================================
// CUDA-compatible scalar/vector types
// ============================================================

using fp16_t = __half;
using fp16x2_t = __half2;

using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;

static_assert(sizeof(fp16_t) == 2, "CUDA-compatible FP16 type must occupy 2 bytes.");

static_assert(sizeof(fp16x2_t) == 4, "CUDA-compatible FP16x2 type must occupy 4 bytes.");

static_assert(sizeof(bf16_t) == 2, "CUDA-compatible BF16 type must occupy 2 bytes.");

static_assert(sizeof(bf16x2_t) == 4, "CUDA-compatible BF16x2 type must occupy 4 bytes.");

static_assert(sizeof(float4) == 16, "CUDA-compatible float4 must occupy 16 bytes.");

static_assert(sizeof(uint4) == 16, "CUDA-compatible uint4 must occupy 16 bytes.");

// LLAISYS storage types must be layout-compatible with the
// CUDA-compatible scalar types used by shared GPU kernels.

static_assert(
    sizeof(llaisys::fp16_t) == sizeof(fp16_t),
    "LLAISYS FP16 storage must match CUDA-compatible FP16 storage.");

static_assert(
    sizeof(llaisys::bf16_t) == sizeof(bf16_t),
    "LLAISYS BF16 storage must match CUDA-compatible BF16 storage.");

// ============================================================
// Template utilities
// ============================================================

template <typename> inline constexpr bool DEPENDENT_FALSE = false;

// ============================================================
// CUDA-compatible dtype dispatch
// ============================================================

template <typename T> struct DTypeTag {
    using type = T;
};

template <typename Function>
decltype(auto) dispatch_dtype(llaisysDataType_t type, Function &&function) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return std::forward<Function>(function)(DTypeTag<float>{});

    case LLAISYS_DTYPE_F16:
        return std::forward<Function>(function)(DTypeTag<fp16_t>{});

    case LLAISYS_DTYPE_BF16:
        return std::forward<Function>(function)(DTypeTag<bf16_t>{});

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

// ============================================================
// CUDA-compatible runtime error handling
// ============================================================
//
// Shared CUDA-style operators are compiled by either:
//   - nvcc for NVIDIA
//   - mxcc for MetaX
//
// Both expose the CUDA-compatible runtime API used by this layer.
// ============================================================

inline void check_cuda(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) { return; }

    std::ostringstream message;

    message << operation << " failed: " << cudaGetErrorString(status);

    throw std::runtime_error(message.str());
}

inline void check_kernel(const char *operation) { check_cuda(cudaGetLastError(), operation); }

// ============================================================
// Integer helpers
// ============================================================

__host__ __device__ constexpr std::size_t div_ceil(std::size_t value, std::size_t divisor) {
    return value / divisor + static_cast<std::size_t>(value % divisor != 0);
}

inline std::size_t cap_grid_size(std::size_t required_blocks, std::size_t max_grid_size) {
    return required_blocks < max_grid_size ? required_blocks : max_grid_size;
}

/**
 * @brief Computes the number of thread blocks needed for a workload,
 *        limited by a caller-provided grid-size cap.
 *
 * The required number of blocks is:
 *
 *     ceil(work_items / block_size)
 *
 * If that value exceeds max_grid_size, the returned grid size is capped.
 *
 * This helper only implements the shared grid-size calculation.
 * The block size and maximum grid size remain operator/backend policies.
 */
inline std::size_t
get_capped_grid_size(std::size_t work_items, std::size_t block_size, std::size_t max_grid_size) {
    return cap_grid_size(div_ceil(work_items, block_size), max_grid_size);
}

// ============================================================
// CUDA-compatible data-type conversion
// ============================================================

template <typename T> __device__ __forceinline__ float to_float(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, fp16_t>) {
        return __half2float(value);
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        return __bfloat162float(value);
    } else {
        static_assert(
            DEPENDENT_FALSE<T>, "Unsupported CUDA-compatible type for conversion to float.");
    }
}

template <typename T> __device__ __forceinline__ T from_float(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, fp16_t>) {
        return __float2half(value);
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        return __float2bfloat16(value);
    } else {
        static_assert(
            DEPENDENT_FALSE<T>, "Unsupported CUDA-compatible type for conversion from float.");
    }
}

// ============================================================
// Address-alignment helpers
// ============================================================

template <std::size_t Alignment, typename T> inline bool is_aligned(const T *pointer) {
    static_assert(
        Alignment > 0 && (Alignment & (Alignment - 1)) == 0,
        "Alignment must be a nonzero power of two.");

    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);

    return address % Alignment == 0;
}

template <std::size_t Alignment, typename... PointerTypes>
inline bool are_aligned(PointerTypes... pointers) {
    return (is_aligned<Alignment>(pointers) && ...);
}

// ============================================================
// 128-bit raw-memory pack
// ============================================================

using Packed128 = uint4;

inline constexpr std::size_t PACKED_128_BYTES = sizeof(Packed128);

inline constexpr std::size_t PACKED_128_ALIGNMENT = alignof(Packed128);

static_assert(PACKED_128_BYTES == 16, "Packed128 must occupy exactly 16 bytes.");

static_assert(PACKED_128_ALIGNMENT == 16, "Packed128 must require 16-byte alignment.");

template <typename T>
inline constexpr std::size_t PACKED_128_ELEMENTS = PACKED_128_BYTES / sizeof(T);

} // namespace llaisys::ops::cuda
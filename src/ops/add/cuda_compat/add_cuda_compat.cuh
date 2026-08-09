#pragma once

#include "../../cuda_compat/common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

namespace llaisys::ops::cuda_compat {

// ============================================================
// Scalar addition
// ============================================================

template <typename T> __device__ __forceinline__ T add_value(T a, T b) {
    if constexpr (std::is_same_v<T, float>) {
        return a + b;
    } else if constexpr (std::is_same_v<T, half>) {
        return __hadd(a, b);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __hadd(a, b);
    } else {
        static_assert(DEPENDENT_FALSE<T>, "Unsupported CUDA-compatible Add data type.");
    }
}

// ============================================================
// Scalar kernel
// ============================================================

template <typename T>
__global__ void
add_kernel(T *__restrict__ c, const T *__restrict__ a, const T *__restrict__ b, std::size_t numel) {
    const std::size_t start
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t stride
        = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t i = start; i < numel; i += stride) { c[i] = add_value<T>(a[i], b[i]); }
}

// ============================================================
// Vector traits
// ============================================================
//
// Add keeps its own vector traits because the vectorized memory
// access pattern depends on the data type.
//
// These describe the shared CUDA-compatible algorithm, not the
// NVIDIA Runtime.
// ============================================================

template <typename T> struct VectorTraits;

template <> struct VectorTraits<float> {
    // 4 FP32 values = 16 bytes.
    static constexpr std::size_t ELEMENTS = 4;

    static constexpr std::size_t ALIGNMENT = alignof(float4);
};

template <> struct VectorTraits<half> {
    // 8 FP16 values = 16 bytes.
    static constexpr std::size_t ELEMENTS = 8;

    static constexpr std::size_t ALIGNMENT = alignof(half2);
};

template <> struct VectorTraits<__nv_bfloat16> {
    // 8 BF16 values = 16 bytes.
    static constexpr std::size_t ELEMENTS = 8;

    static constexpr std::size_t ALIGNMENT = alignof(__nv_bfloat162);
};

// ============================================================
// Vectorized-kernel eligibility
// ============================================================

template <typename T>
inline bool can_use_vectorized_add(const T *c, const T *a, const T *b, std::size_t numel) {
    constexpr std::size_t vector_size = VectorTraits<T>::ELEMENTS;

    constexpr std::size_t vector_alignment = VectorTraits<T>::ALIGNMENT;

    return numel >= vector_size && are_aligned<vector_alignment>(c, a, b);
}

// ============================================================
// Logical work-item calculation
//
// Shared layer decides how much work exists.
//
// Vendor backend decides how that work maps to its launch grid.
// ============================================================

template <typename T>
inline std::size_t get_add_work_items(std::size_t numel, bool use_vectorized_kernel) {
    if (use_vectorized_kernel) { return numel / VectorTraits<T>::ELEMENTS; }

    return numel;
}

// ============================================================
// Vectorized kernel
// ============================================================

template <typename T>
__global__ void add_kernel_vectorized(
    T *__restrict__ c, const T *__restrict__ a, const T *__restrict__ b, std::size_t numel) {
    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t thread_stride
        = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    constexpr std::size_t vector_size = VectorTraits<T>::ELEMENTS;

    const std::size_t vector_count = numel / vector_size;

    for (std::size_t vector_index = thread_index; vector_index < vector_count;
         vector_index += thread_stride) {
        const std::size_t element_index = vector_index * vector_size;

        if constexpr (std::is_same_v<T, float>) {
            const float4 a_vector = *reinterpret_cast<const float4 *>(a + element_index);

            const float4 b_vector = *reinterpret_cast<const float4 *>(b + element_index);

            float4 c_vector;

            c_vector.x = a_vector.x + b_vector.x;

            c_vector.y = a_vector.y + b_vector.y;

            c_vector.z = a_vector.z + b_vector.z;

            c_vector.w = a_vector.w + b_vector.w;

            *reinterpret_cast<float4 *>(c + element_index) = c_vector;

        } else if constexpr (std::is_same_v<T, half>) {
            const half2 *const a_vector = reinterpret_cast<const half2 *>(a + element_index);

            const half2 *const b_vector = reinterpret_cast<const half2 *>(b + element_index);

            half2 *const c_vector = reinterpret_cast<half2 *>(c + element_index);

#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                c_vector[pair] = __hadd2(a_vector[pair], b_vector[pair]);
            }

        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            const __nv_bfloat162 *const a_vector
                = reinterpret_cast<const __nv_bfloat162 *>(a + element_index);

            const __nv_bfloat162 *const b_vector
                = reinterpret_cast<const __nv_bfloat162 *>(b + element_index);

            __nv_bfloat162 *const c_vector = reinterpret_cast<__nv_bfloat162 *>(c + element_index);

#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                c_vector[pair] = __hadd2(a_vector[pair], b_vector[pair]);
            }

        } else {
            static_assert(DEPENDENT_FALSE<T>, "Unsupported CUDA-compatible Add vector type.");
        }
    }

    // ========================================================
    // Scalar tail
    // ========================================================

    const std::size_t tail_start = vector_count * vector_size;

    for (std::size_t i = tail_start + thread_index; i < numel; i += thread_stride) {
        c[i] = add_value<T>(a[i], b[i]);
    }
}

// ============================================================
// Shared kernel launcher
//
// This function intentionally does NOT:
//
//   - choose vendor block size
//   - choose vendor grid limit
//   - convert llaisysStream_t
//   - call cudaGetLastError()
//   - call CUDA_CHECK
//
// Those are backend-specific responsibilities.
// ============================================================

template <typename T, typename StreamT>
inline void launch_add_kernel(
    T *c,
    const T *a,
    const T *b,
    std::size_t numel,
    std::size_t block_size,
    std::size_t grid_size,
    bool use_vectorized_kernel,
    StreamT stream) {
    if (numel == 0) { return; }

    const dim3 block_dimension(static_cast<unsigned int>(block_size));

    const dim3 grid_dimension(static_cast<unsigned int>(grid_size));

    if (use_vectorized_kernel) {
        add_kernel_vectorized<T><<<grid_dimension, block_dimension, 0, stream>>>(c, a, b, numel);
    } else {
        add_kernel<T><<<grid_dimension, block_dimension, 0, stream>>>(c, a, b, numel);
    }
}

} // namespace llaisys::ops::cuda_compat
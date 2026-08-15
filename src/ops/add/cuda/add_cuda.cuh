#pragma once

#include "add_cuda.hpp"

#include "../../../utils.hpp"
#include "../../cuda/common.cuh"
#include "../add_config.hpp"

#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace llaisys::ops::cuda {
namespace {

struct AddLaunchPlan {
    std::size_t block_size;
    std::size_t grid_size;
    std::size_t work_items;
    bool use_vectorized_kernel;
};

template <typename T> __device__ __forceinline__ T add_value(T a, T b) {
    if constexpr (std::is_same_v<T, float>) {
        return a + b;
    } else if constexpr (std::is_same_v<T, fp16_t>) {
        return __hadd(a, b);
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        return __hadd(a, b);
    } else {
        static_assert(DEPENDENT_FALSE<T>, "Unsupported CUDA-compatible Add data type.");
    }
}

template <typename T>
__global__ void add_scalar_kernel(
    T *__restrict__ c, const T *__restrict__ a, const T *__restrict__ b, std::size_t numel) {
    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid_stride
        = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    for (std::size_t index = thread_index; index < numel; index += grid_stride) {
        c[index] = add_value<T>(a[index], b[index]);
    }
}

template <typename T> struct VectorTraits;

template <> struct VectorTraits<float> {
    static constexpr std::size_t ELEMENTS = 4;
    static constexpr std::size_t ALIGNMENT = alignof(float4);
};

template <> struct VectorTraits<fp16_t> {
    static constexpr std::size_t ELEMENTS = 8;
    static constexpr std::size_t ALIGNMENT = alignof(fp16x2_t);
};

template <> struct VectorTraits<bf16_t> {
    static constexpr std::size_t ELEMENTS = 8;
    static constexpr std::size_t ALIGNMENT = alignof(bf16x2_t);
};

template <typename T>
inline bool can_use_vectorized_add(const T *c, const T *a, const T *b, std::size_t numel) {
    constexpr std::size_t vector_size = VectorTraits<T>::ELEMENTS;
    constexpr std::size_t vector_alignment = VectorTraits<T>::ALIGNMENT;
    return numel >= vector_size && are_aligned<vector_alignment>(c, a, b);
}

template <typename T>
inline AddLaunchPlan get_launch_plan(
    const T *c,
    const T *a,
    const T *b,
    std::size_t numel,
    std::size_t block_size,
    std::size_t max_blocks) {
    const bool use_vectorized_kernel = can_use_vectorized_add<T>(c, a, b, numel);
    const std::size_t work_items
        = use_vectorized_kernel ? numel / VectorTraits<T>::ELEMENTS : numel;

    return AddLaunchPlan{
        block_size,
        get_capped_grid_size(work_items, block_size, max_blocks),
        work_items,
        use_vectorized_kernel,
    };
}

template <typename T>
__global__ void add_vectorized_kernel(
    T *__restrict__ c, const T *__restrict__ a, const T *__restrict__ b, std::size_t numel) {
    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t grid_stride
        = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    constexpr std::size_t vector_size = VectorTraits<T>::ELEMENTS;
    const std::size_t vector_count = numel / vector_size;

    for (std::size_t vector_index = thread_index; vector_index < vector_count;
         vector_index += grid_stride) {
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
        } else if constexpr (std::is_same_v<T, fp16_t>) {
            const fp16x2_t *const a_vector = reinterpret_cast<const fp16x2_t *>(a + element_index);
            const fp16x2_t *const b_vector = reinterpret_cast<const fp16x2_t *>(b + element_index);
            fp16x2_t *const c_vector = reinterpret_cast<fp16x2_t *>(c + element_index);

#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                c_vector[pair] = __hadd2(a_vector[pair], b_vector[pair]);
            }
        } else if constexpr (std::is_same_v<T, bf16_t>) {
            const bf16x2_t *const a_vector = reinterpret_cast<const bf16x2_t *>(a + element_index);
            const bf16x2_t *const b_vector = reinterpret_cast<const bf16x2_t *>(b + element_index);
            bf16x2_t *const c_vector = reinterpret_cast<bf16x2_t *>(c + element_index);

#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                c_vector[pair] = __hadd2(a_vector[pair], b_vector[pair]);
            }
        } else {
            static_assert(DEPENDENT_FALSE<T>, "Unsupported CUDA-compatible Add vector type.");
        }
    }

    const std::size_t tail_start = vector_count * vector_size;
    for (std::size_t index = tail_start + thread_index; index < numel; index += grid_stride) {
        c[index] = add_value<T>(a[index], b[index]);
    }
}

template <typename T>
void launch_add(T *c, const T *a, const T *b, std::size_t numel, cudaStream_t stream) {
    const add_config::LaunchPolicy &policy = add_config::get_launch_policy();
    const AddLaunchPlan plan
        = get_launch_plan<T>(c, a, b, numel, policy.block_size, policy.max_blocks);

    if (config::debug_enabled()) {
        std::fprintf(
            stderr,
            "[Add][%s] implementation=shared numel=%zu kernel=%s work_items=%zu "
            "block=%zu grid=%zu max_blocks=%zu\n",
            GPU_BACKEND_NAME, numel, plan.use_vectorized_kernel ? "vectorized" : "scalar",
            plan.work_items, plan.block_size, plan.grid_size, policy.max_blocks);
    }

    const dim3 block_dimension(static_cast<unsigned int>(plan.block_size));
    const dim3 grid_dimension(static_cast<unsigned int>(plan.grid_size));

    if (plan.use_vectorized_kernel) {
        add_vectorized_kernel<T><<<grid_dimension, block_dimension, 0, stream>>>(c, a, b, numel);
    } else {
        add_scalar_kernel<T><<<grid_dimension, block_dimension, 0, stream>>>(c, a, b, numel);
    }

    check_kernel("Add kernel");
}

} // namespace

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    std::size_t numel,
    llaisysStream_t stream) {
    CHECK_ARGUMENT(numel == 0 || c != nullptr, "Add: output pointer c must not be null.");
    CHECK_ARGUMENT(numel == 0 || a != nullptr, "Add: input pointer a must not be null.");
    CHECK_ARGUMENT(numel == 0 || b != nullptr, "Add: input pointer b must not be null.");

    if (numel == 0) { return; }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    return dispatch_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_add<T>(
            reinterpret_cast<T *>(c), reinterpret_cast<const T *>(a),
            reinterpret_cast<const T *>(b), numel, cuda_stream);
    });
}

} // namespace llaisys::ops::cuda

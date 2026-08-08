#include <cstddef>
#include <type_traits>

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"
#include "add_nvidia.cuh"


namespace {

using llaisys::device::nvidia::are_aligned;
using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::DEPENDENT_FALSE;
using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::device::nvidia::to_cuda_stream;


// ============================================================
// Scalar addition
// ============================================================

template <typename T>
__device__ __forceinline__ T add_value(T a, T b) {
    if constexpr (std::is_same_v<T, float>) {
        return a + b;

    } else if constexpr (std::is_same_v<T, half>) {
        return __hadd(a, b);

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __hadd(a, b);

    } else {
        static_assert(DEPENDENT_FALSE<T>, "Unsupported CUDA Add data type.");
    }
}

// ============================================================
// Scalar kernel
// ============================================================

template <typename T>
__global__ void add_kernel(T *__restrict__ c, const T *__restrict__ a,
                           const T *__restrict__ b, std::size_t numel) {
    // Global index of the current CUDA thread.
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Total number of CUDA threads in the grid.
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    // Grid-stride loop: one CUDA thread can process multiple elements.
    for (std::size_t i = start; i < numel; i += stride) {
        c[i] = add_value<T>(a[i], b[i]);
    }
}

// ============================================================
// Vector traits
// ============================================================

// Add keeps its own vector traits because the vectorized
// memory-access pattern depends on the data type.
template <typename T>
struct VectorTraits;

template <>
struct VectorTraits<float> {
    // Four FP32 values:
    // 4 elements * 4 bytes = 16 bytes.
    static constexpr std::size_t ELEMENTS = 4;

    // float4 access requires float4-compatible alignment.
    static constexpr std::size_t ALIGNMENT = alignof(float4);
};

template <>
struct VectorTraits<half> {
    // Eight FP16 values:
    // 8 elements * 2 bytes = 16 bytes.
    static constexpr std::size_t ELEMENTS = 8;

    // FP16 values are accessed through half2 pointers.
    static constexpr std::size_t ALIGNMENT = alignof(half2);
};

template <>
struct VectorTraits<__nv_bfloat16> {
    // Eight BF16 values:
    // 8 elements * 2 bytes = 16 bytes.
    static constexpr std::size_t ELEMENTS = 8;

    // BF16 values are accessed through __nv_bfloat162 pointers.
    static constexpr std::size_t ALIGNMENT = alignof(__nv_bfloat162);
};

// ============================================================
// Vectorized kernel
// ============================================================

template <typename T>
__global__ void add_kernel_vectorized(T *__restrict__ c,
                                      const T *__restrict__ a,
                                      const T *__restrict__ b,
                                      std::size_t numel) {
    const std::size_t thread_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t thread_stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    // float:
    //     4 elements per vector group.
    //
    // half and BF16:
    //     8 elements per vector group.
    constexpr std::size_t vector_size = VectorTraits<T>::ELEMENTS;
    const std::size_t vector_count = numel / vector_size;

    for (std::size_t vector_index = thread_index; vector_index < vector_count; vector_index += thread_stride) {
        const std::size_t element_index = vector_index * vector_size;

        if constexpr (std::is_same_v<T, float>) {
            const float4 a_vector = *reinterpret_cast<const float4 *>(a + element_index);
            const float4 b_vector = *reinterpret_cast<const float4 *>(b + element_index);
            float4 c_vector;

            c_vector.x = a_vector.x + b_vector.x;
            c_vector.y = a_vector.y + b_vector.y;
            c_vector.z = a_vector.z + b_vector.z;
            c_vector.w = a_vector.w + b_vector.w;

            // Write four FP32 results.
            *reinterpret_cast<float4 *>(c + element_index) = c_vector;
        } else if constexpr (std::is_same_v<T, half>) {
            // Eight FP16 values are treated as four half2 pairs.
            const half2 *const a_vector = reinterpret_cast<const half2 *>(a + element_index);
            const half2 *const b_vector = reinterpret_cast<const half2 *>(b + element_index);
            half2 *const c_vector = reinterpret_cast<half2 *>(c + element_index);

#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                c_vector[pair] = __hadd2(a_vector[pair], b_vector[pair]);
            }

        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // Eight BF16 values are treated as four __nv_bfloat162 pairs.
            const __nv_bfloat162 *const a_vector = reinterpret_cast<const __nv_bfloat162 *>(a + element_index);
            const __nv_bfloat162 *const b_vector = reinterpret_cast<const __nv_bfloat162 *>(b + element_index);
            __nv_bfloat162 *const c_vector = reinterpret_cast<__nv_bfloat162 *>(c + element_index);

#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                c_vector[pair] = __hadd2(a_vector[pair], b_vector[pair]);
            }
        }
    }

    // Process the remaining elements that cannot form
    // a complete vector group.
    const std::size_t tail_start = vector_count * vector_size;

    for (std::size_t i = tail_start + thread_index; i < numel;
         i += thread_stride) {
        c[i] = add_value<T>(a[i], b[i]);
    }
}

// ============================================================
// Kernel launcher
// ============================================================

template <typename T>
void launch_add_kernel(T *c, const T *a, const T *b, std::size_t numel,
                       cudaStream_t stream) {

    if (numel == 0) {
        return;
    }

    constexpr std::size_t vector_size = VectorTraits<T>::ELEMENTS;
    constexpr std::size_t vector_alignment = VectorTraits<T>::ALIGNMENT;
    const bool use_vectorized_kernel = numel >= vector_size && are_aligned<vector_alignment>(c, a, b);
    constexpr std::size_t block_size = CUDA_BLOCK_SIZE;
    const std::size_t work_items = use_vectorized_kernel ? numel / vector_size : numel;
    const std::size_t grid_size = get_capped_grid_size(work_items, block_size);
    const dim3 block_dimension(static_cast<unsigned int>(block_size));
    const dim3 grid_dimension(static_cast<unsigned int>(grid_size));

    if (use_vectorized_kernel) {
        add_kernel_vectorized<T>
            <<<grid_dimension, block_dimension, 0, stream>>>(c, a, b, numel);
    } else {
        add_kernel<T>
            <<<grid_dimension, block_dimension, 0, stream>>>(c, a, b, numel);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// ============================================================
// Public NVIDIA backend interface
// ============================================================
namespace llaisys::ops::nvidia {
    
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, std::size_t numel, llaisysStream_t stream) {
    CHECK_ARGUMENT(numel == 0 || c != nullptr, "Add: output pointer c must not be null.");
    CHECK_ARGUMENT(numel == 0 || a != nullptr, "Add: input pointer a must not be null.");
    CHECK_ARGUMENT(numel == 0 || b != nullptr, "Add: input pointer b must not be null.");

    const cudaStream_t cuda_stream = llaisys::device::nvidia::to_cuda_stream(stream);

    return llaisys::device::nvidia::dispatch_cuda_dtype(
        type,
        [&](auto tag) {
            using T = typename decltype(tag)::type;

            return launch_add_kernel<T>(
                reinterpret_cast<T *>(c),
                reinterpret_cast<const T *>(a),
                reinterpret_cast<const T *>(b),
                numel,
                cuda_stream
            );
        }
    );
}

} // namespace llaisys::ops::nvidia
#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "add_nvidia.cuh"

namespace llaisys::ops::nvidia {

namespace {

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::div_ceil;

// Add-specific maximum grid size.
// Other operators may use different limits.
inline constexpr std::size_t MAX_GRID_SIZE = 4096;

// Confirm that the LLAISYS custom 16-bit types have the same
// storage size as the corresponding CUDA types.
static_assert(sizeof(llaisys::fp16_t) == sizeof(half),
              "llaisys::fp16_t and CUDA half must have the same size.");

static_assert(
    sizeof(llaisys::bf16_t) == sizeof(__nv_bfloat16),
    "llaisys::bf16_t and CUDA __nv_bfloat16 must have the same size.");

// Used to produce a compile-time error if an unsupported
// template type reaches add_value().
template <typename>
inline constexpr bool ALWAYS_FALSE = false;

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
    static_assert(ALWAYS_FALSE<T>, "Unsupported CUDA Add data type.");
  }
}

// ============================================================
// Scalar kernel
// ============================================================

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, std::size_t numel) {
  // Global index of the current CUDA thread.
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  // Total number of CUDA threads in the grid.
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  // Grid-stride loop:
  // one thread can process multiple elements.
  for (std::size_t i = start; i < numel; i += stride) {
    c[i] = add_value<T>(a[i], b[i]);
  }
}

// ============================================================
// Vector traits
// ============================================================

template <typename T>
struct VectorTraits;

template <>
struct VectorTraits<float> {
  // Four float values:
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

  // FP16 values are processed in half2 pairs.
  static constexpr std::size_t ALIGNMENT = alignof(half2);
};

template <>
struct VectorTraits<__nv_bfloat16> {
  // Eight BF16 values:
  // 8 elements * 2 bytes = 16 bytes.
  static constexpr std::size_t ELEMENTS = 8;

  // BF16 values are processed in __nv_bfloat162 pairs.
  static constexpr std::size_t ALIGNMENT = alignof(__nv_bfloat162);
};

// ============================================================
// Vectorized kernel
// ============================================================

template <typename T>
__global__ void add_kernel_vectorized(T *c, const T *a, const T *b,
                                      std::size_t numel) {
  const std::size_t thread_idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  const std::size_t thread_stride =
      static_cast<std::size_t>(blockDim.x) * gridDim.x;

  // float: 4 elements
  // half: 8 elements
  // BF16: 8 elements
  constexpr std::size_t vector_size = VectorTraits<T>::ELEMENTS;

  // Number of complete vector groups.
  const std::size_t vector_count = numel / vector_size;

  for (std::size_t vector_idx = thread_idx; vector_idx < vector_count;
       vector_idx += thread_stride) {
    const std::size_t index = vector_idx * vector_size;

    if constexpr (std::is_same_v<T, float>) {
      // Read four float values from a.
      const float4 a_vec = *reinterpret_cast<const float4 *>(a + index);

      // Read four float values from b.
      const float4 b_vec = *reinterpret_cast<const float4 *>(b + index);

      float4 c_vec;

      c_vec.x = a_vec.x + b_vec.x;
      c_vec.y = a_vec.y + b_vec.y;
      c_vec.z = a_vec.z + b_vec.z;
      c_vec.w = a_vec.w + b_vec.w;

      // Write four float values to c.
      *reinterpret_cast<float4 *>(c + index) = c_vec;

    } else if constexpr (std::is_same_v<T, half>) {
      // Eight FP16 values are treated as
      // four half2 pairs.
      const half2 *a_vec = reinterpret_cast<const half2 *>(a + index);

      const half2 *b_vec = reinterpret_cast<const half2 *>(b + index);

      half2 *c_vec = reinterpret_cast<half2 *>(c + index);

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        c_vec[i] = __hadd2(a_vec[i], b_vec[i]);
      }

    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      // Eight BF16 values are treated as
      // four __nv_bfloat162 pairs.
      const __nv_bfloat162 *a_vec =
          reinterpret_cast<const __nv_bfloat162 *>(a + index);

      const __nv_bfloat162 *b_vec =
          reinterpret_cast<const __nv_bfloat162 *>(b + index);

      __nv_bfloat162 *c_vec = reinterpret_cast<__nv_bfloat162 *>(c + index);

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        c_vec[i] = __hadd2(a_vec[i], b_vec[i]);
      }
    }
  }

  // Process remaining elements that cannot form
  // a complete vector.
  const std::size_t tail_start = vector_count * vector_size;

  for (std::size_t i = tail_start + thread_idx; i < numel; i += thread_stride) {
    c[i] = add_value<T>(a[i], b[i]);
  }
}

// ============================================================
// Alignment check
// ============================================================

template <typename T>
bool is_vector_aligned(const T *c, const T *a, const T *b) {
  constexpr std::uintptr_t alignment = VectorTraits<T>::ALIGNMENT;

  const std::uintptr_t c_address = reinterpret_cast<std::uintptr_t>(c);

  const std::uintptr_t a_address = reinterpret_cast<std::uintptr_t>(a);

  const std::uintptr_t b_address = reinterpret_cast<std::uintptr_t>(b);

  return (c_address % alignment == 0 && a_address % alignment == 0 &&
          b_address % alignment == 0);
}

// ============================================================
// Kernel launcher
// ============================================================

template <typename T>
void launch_add_kernel(T *c, const T *a, const T *b, std::size_t numel, cudaStream_t stream) {
  // CUDA does not allow a kernel launch with zero blocks.
  if (numel == 0) {
    return;
  }

  constexpr std::size_t vector_size = VectorTraits<T>::ELEMENTS;

  // Vectorized execution requires enough elements and
  // correctly aligned addresses.
  const bool use_vectorized_kernel =
      numel >= vector_size && is_vector_aligned(c, a, b);

  constexpr std::size_t block_size = CUDA_BLOCK_SIZE;

  std::size_t work_items;

  if (use_vectorized_kernel) {
    // One work item represents one complete vector.
    work_items = numel / vector_size;
  } else {
    // One work item represents one scalar element.
    work_items = numel;
  }

  const std::size_t required_blocks = div_ceil(work_items, block_size);

  // The kernels use grid-stride loops, so the grid
  // does not need to grow without limit.
  const std::size_t grid_size = std::min(required_blocks, MAX_GRID_SIZE);

  const dim3 block_dim(static_cast<unsigned int>(block_size));

  const dim3 grid_dim(static_cast<unsigned int>(grid_size));

  if (use_vectorized_kernel) {
    add_kernel_vectorized<T><<<grid_dim, block_dim, 0, stream>>>(c, a, b, numel);
  } else {
    add_kernel<T><<<grid_dim, block_dim, 0, stream>>>(c, a, b, numel);
  }

  // Check kernel launch and configuration errors.
  // This does not synchronize the entire device.
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

// ============================================================
// Public NVIDIA backend interface
// ============================================================

void add(std::byte *c, const std::byte *a, const std::byte *b,
         llaisysDataType_t type, std::size_t numel, llaisysStream_t stream) {
  CHECK_ARGUMENT(numel == 0 || c != nullptr,
                 "Add: output pointer c must not be null.");

  CHECK_ARGUMENT(numel == 0 || a != nullptr,
                 "Add: input pointer a must not be null.");

  CHECK_ARGUMENT(numel == 0 || b != nullptr,
                 "Add: input pointer b must not be null.");

  // llaisysStream_t is void* in the generic LLAISYS API.
	// In the NVIDIA backend, it stores a cudaStream_t.
	//
	// nullptr is also valid and represents the CUDA default stream.
	const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);


  switch (type) {
    case LLAISYS_DTYPE_F32:
      return launch_add_kernel<float>(
          reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
          reinterpret_cast<const float *>(b), numel, cuda_stream);

    case LLAISYS_DTYPE_BF16:
      return launch_add_kernel<__nv_bfloat16>(
          reinterpret_cast<__nv_bfloat16 *>(c),
          reinterpret_cast<const __nv_bfloat16 *>(a),
          reinterpret_cast<const __nv_bfloat16 *>(b), numel, cuda_stream);

    case LLAISYS_DTYPE_F16:
      return launch_add_kernel<half>(reinterpret_cast<half *>(c),
                                     reinterpret_cast<const half *>(a),
                                     reinterpret_cast<const half *>(b), numel, cuda_stream);

    default:
      EXCEPTION_UNSUPPORTED_DATATYPE(type);
  }
}

}  // namespace llaisys::ops::nvidia
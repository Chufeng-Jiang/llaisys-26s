#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"
#include "embedding_nvidia.cuh"

namespace {

using llaisys::device::nvidia::are_aligned;
using llaisys::device::nvidia::CUDA_DEFAULT_MAX_GRID_SIZE;
using llaisys::device::nvidia::get_warp_aligned_block_size;
using llaisys::device::nvidia::Packed128;
using llaisys::device::nvidia::PACKED_128_ALIGNMENT;
using llaisys::device::nvidia::PACKED_128_BYTES;
using llaisys::device::nvidia::PACKED_128_ELEMENTS;

// ============================================================
// Scalar kernel
// ============================================================
template <typename T>
__global__ void embedding_scalar_kernel(T *__restrict__ out,
                                        const T *__restrict__ weight,
                                        const std::int64_t *__restrict__ index,
                                        std::size_t index_count,
                                        std::size_t len,
                                        std::size_t vocabulary_size) {

    for (std::size_t output_row = blockIdx.x; output_row < index_count; output_row += gridDim.x) {
        const std::int64_t signed_weight_row = index[output_row];

        if (signed_weight_row < 0 || static_cast<std::uint64_t>(signed_weight_row) >= static_cast<std::uint64_t>(vocabulary_size)) {
            continue;
        }

        const std::size_t weight_row = static_cast<std::size_t>(signed_weight_row);
        const std::size_t output_offset = output_row * len;
        const std::size_t weight_offset = weight_row * len;

        for (std::size_t column = threadIdx.x; column < len; column += blockDim.x) {
            out[output_offset + column] = weight[weight_offset + column];
        }
    }
}

// ============================================================
// Vectorized kernel
// Vectorized Embedding implementation.
// Every thread copies one 128-bit value at a time.
// This kernel must only be launched when:
// 1. out is aligned to PACKED_128_ALIGNMENT;
// 2. weight is aligned to PACKED_128_ALIGNMENT;
// 3. every row contains a multiple of PACKED_128_BYTES.
// ============================================================

template <typename T>
__global__ void embedding_vectorized_kernel(T *__restrict__ out, const T *__restrict__ weight,
                                            const std::int64_t *__restrict__ index, std::size_t index_count,
                                            std::size_t len, std::size_t vocabulary_size) {
                                                
    static_assert(PACKED_128_BYTES % sizeof(T) == 0, "Embedding: data type size must divide the 128-bit pack size.");

    // Number of T elements stored in one 128-bit pack.
    //
    // float:
    //     16 / 4 = 4 elements
    //
    // half and BF16:
    //     16 / 2 = 8 elements
    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;
    const std::size_t packs_per_row = len / elements_per_pack;

    for (std::size_t output_row = blockIdx.x; output_row < index_count; output_row += gridDim.x) {
        const std::int64_t signed_weight_row = index[output_row];

        if (signed_weight_row < 0 || static_cast<std::uint64_t>(signed_weight_row) >= static_cast<std::uint64_t>(vocabulary_size)) {
            continue;
        }

        const std::size_t weight_row = static_cast<std::size_t>(signed_weight_row);
        T *const output_row_pointer = out + output_row * len;
        const T *const weight_row_pointer = weight + weight_row * len;

        // The launcher guarantees that both pointers and every
        // row start satisfy the 128-bit alignment requirement.
        auto *const packed_output = reinterpret_cast<Packed128 *>(output_row_pointer);

        const auto *const packed_weight = reinterpret_cast<const Packed128 *>(weight_row_pointer);

        // Each thread copies one or more 128-bit packs.
        for (std::size_t pack_index = threadIdx.x; pack_index < packs_per_row;
             pack_index += blockDim.x) {
            packed_output[pack_index] = packed_weight[pack_index];
        }
    }
}

// ============================================================
// Kernel launcher
// ============================================================

template <typename T>
void launch_embedding(T *out, const std::int64_t *index, const T *weight,
                      std::size_t numel, std::size_t len,
                      std::size_t vocabulary_size, cudaStream_t stream) {
    CHECK_ARGUMENT(len > 0,
                   "Embedding: embedding length must be greater than zero.");

    CHECK_ARGUMENT(numel % len == 0,
                   "Embedding: output element count must be divisible by "
                   "embedding length.");

    CHECK_ARGUMENT(numel == 0 || out != nullptr,
                   "Embedding: output pointer must not be null.");

    CHECK_ARGUMENT(numel == 0 || index != nullptr,
                   "Embedding: index pointer must not be null.");

    CHECK_ARGUMENT(numel == 0 || weight != nullptr,
                   "Embedding: weight pointer must not be null.");

    CHECK_ARGUMENT(numel == 0 || vocabulary_size > 0,
                   "Embedding: vocabulary size must be greater than zero.");

    CHECK_ARGUMENT(len <= std::numeric_limits<std::size_t>::max() / sizeof(T),
                   "Embedding: row byte size overflows size_t.");

    CHECK_ARGUMENT(
        vocabulary_size == 0 || len <= std::numeric_limits<std::size_t>::max() / vocabulary_size,
        "Embedding: weight element count overflows size_t.");

    // CUDA cannot launch a kernel with zero blocks.
    if (numel == 0) {
        return;
    }

    const std::size_t index_count = numel / len;

    const std::size_t row_bytes = len * sizeof(T);

    // This kernel assigns one logical output row to each block.
    //
    // The row-level grid-stride loop allows the grid size to be
    // capped without leaving any rows unprocessed.
    const std::size_t grid_size = index_count < CUDA_DEFAULT_MAX_GRID_SIZE
                                    ? index_count
                                    : CUDA_DEFAULT_MAX_GRID_SIZE;

    // Base pointer alignment.
    const bool base_addresses_aligned = are_aligned<PACKED_128_ALIGNMENT>(out, weight);

    // Even when the base pointer is aligned, every subsequent
    // row must also start at an aligned address.
    //
    // Therefore, the row stride in bytes must be a multiple of
    // the 128-bit pack size.
    const bool every_row_aligned = row_bytes % PACKED_128_BYTES == 0;

    const bool use_vectorized_kernel = base_addresses_aligned && every_row_aligned;

    if (use_vectorized_kernel) {
        const std::size_t packs_per_row = row_bytes / PACKED_128_BYTES;

        const unsigned int block_size = get_warp_aligned_block_size(packs_per_row);

        embedding_vectorized_kernel<T>
            <<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
                out, weight, index, index_count, len, vocabulary_size);

    } else {
        const unsigned int block_size = get_warp_aligned_block_size(len);

        embedding_scalar_kernel<T>
            <<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
                out, weight, index, index_count, len, vocabulary_size);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// ============================================================
// Public NVIDIA backend interface
// ============================================================

namespace llaisys::ops::nvidia {

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, std::size_t numel, std::size_t len,
               std::size_t vocabulary_size, llaisysStream_t stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launch_embedding<float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const std::int64_t *>(index),
            reinterpret_cast<const float *>(weight), numel, len, vocabulary_size,
            cuda_stream);

    case LLAISYS_DTYPE_F16:
        return launch_embedding<half>(
            reinterpret_cast<half *>(out),
            reinterpret_cast<const std::int64_t *>(index),
            reinterpret_cast<const half *>(weight), numel, len, vocabulary_size,
            cuda_stream);

    case LLAISYS_DTYPE_BF16:
        return launch_embedding<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16 *>(out),
            reinterpret_cast<const std::int64_t *>(index),
            reinterpret_cast<const __nv_bfloat16 *>(weight), numel, len,
            vocabulary_size, cuda_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
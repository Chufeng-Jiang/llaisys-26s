#include "embedding_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include "../cuda_compat/embedding_cuda_compat.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

namespace cuda_compat = llaisys::ops::cuda_compat;

using llaisys::device::nvidia::CUDA_DEFAULT_MAX_GRID_SIZE;

using llaisys::device::nvidia::get_warp_aligned_block_size;

using llaisys::device::nvidia::to_cuda_stream;

// ============================================================
// NVIDIA Embedding adapter
// ============================================================
//
// Shared CUDA-compatible layer owns:
//
//   - scalar Embedding kernel
//   - Packed128 Embedding kernel
//   - alignment requirements
//   - per-row logical work
//
// NVIDIA adapter owns:
//
//   - grid-size cap
//   - warp-aligned block-size policy
//   - CUDA stream conversion
//   - CUDA launch error handling
// ============================================================

template <typename T>
void launch_nvidia_embedding(
    T *out,
    const std::int64_t *index,
    const T *weight,
    std::size_t numel,
    std::size_t embedding_length,
    std::size_t vocabulary_size,
    cudaStream_t stream) {
    // ========================================================
    // Validation
    // ========================================================

    CHECK_ARGUMENT(embedding_length > 0, "Embedding: embedding length must be greater than zero.");

    CHECK_ARGUMENT(
        numel % embedding_length == 0, "Embedding: output element count must be divisible by "
                                       "embedding length.");

    CHECK_ARGUMENT(numel == 0 || out != nullptr, "Embedding: output pointer must not be null.");

    CHECK_ARGUMENT(numel == 0 || index != nullptr, "Embedding: index pointer must not be null.");

    CHECK_ARGUMENT(numel == 0 || weight != nullptr, "Embedding: weight pointer must not be null.");

    CHECK_ARGUMENT(
        numel == 0 || vocabulary_size > 0, "Embedding: vocabulary size must be greater than zero.");

    CHECK_ARGUMENT(
        embedding_length <= std::numeric_limits<std::size_t>::max() / sizeof(T),
        "Embedding: row byte size overflows size_t.");

    CHECK_ARGUMENT(
        vocabulary_size == 0
            || embedding_length <= std::numeric_limits<std::size_t>::max() / vocabulary_size,
        "Embedding: weight element count overflows size_t.");

    if (numel == 0) { return; }

    const std::size_t index_count = numel / embedding_length;

    // ========================================================
    // Shared path selection
    // ========================================================

    const bool use_vectorized_kernel
        = cuda_compat::can_use_vectorized_embedding<T>(out, weight, embedding_length);

    const std::size_t row_work_items
        = cuda_compat::get_embedding_row_work_items<T>(embedding_length, use_vectorized_kernel);

    // ========================================================
    // NVIDIA-specific launch tuning
    // ========================================================

    // One logical output row is initially assigned to one block.
    //
    // The shared kernel contains a row-level grid-stride loop,
    // so the physical grid can safely be capped.

    const std::size_t grid_size
        = index_count < CUDA_DEFAULT_MAX_GRID_SIZE ? index_count : CUDA_DEFAULT_MAX_GRID_SIZE;

    const unsigned int block_size = get_warp_aligned_block_size(row_work_items);

    // ========================================================
    // Shared CUDA-compatible kernel
    // ========================================================

    cuda_compat::launch_embedding_kernel<T>(
        out, index, weight, index_count, embedding_length, vocabulary_size, block_size, grid_size,
        use_vectorized_kernel, stream);

    // ========================================================
    // NVIDIA-specific launch error handling
    // ========================================================

    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// ============================================================
// Public NVIDIA backend
// ============================================================

namespace llaisys::ops::nvidia {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    std::size_t numel,
    std::size_t len,
    std::size_t vocabulary_size,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = to_cuda_stream(stream);

    return llaisys::device::nvidia::dispatch_cuda_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_nvidia_embedding<T>(
            reinterpret_cast<T *>(out), reinterpret_cast<const std::int64_t *>(index),
            reinterpret_cast<const T *>(weight), numel, len, vocabulary_size, cuda_stream);
    });
}

} // namespace llaisys::ops::nvidia
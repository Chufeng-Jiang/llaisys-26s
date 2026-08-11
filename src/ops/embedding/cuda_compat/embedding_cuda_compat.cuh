#pragma once

#include "../../cuda_compat/common.cuh"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cuda_compat {

// ============================================================
// Scalar Embedding kernel
// ============================================================
//
// One logical output row is assigned to each block.
//
// A row-level grid-stride loop allows the backend to cap the
// physical grid size without leaving rows unprocessed.
// ============================================================

template <typename T>
__global__ void embedding_scalar_kernel(
    T *__restrict__ out,
    const T *__restrict__ weight,
    const std::int64_t *__restrict__ index,
    std::size_t index_count,
    std::size_t embedding_length,
    std::size_t vocabulary_size) {
    for (std::size_t output_row = static_cast<std::size_t>(blockIdx.x); output_row < index_count;
         output_row += static_cast<std::size_t>(gridDim.x)) {
        const std::int64_t signed_weight_row = index[output_row];

        // Preserve the current backend behavior:
        // invalid indices leave the corresponding output row
        // untouched.
        if (signed_weight_row < 0
            || static_cast<std::uint64_t>(signed_weight_row)
                   >= static_cast<std::uint64_t>(vocabulary_size)) {
            continue;
        }

        const std::size_t weight_row = static_cast<std::size_t>(signed_weight_row);

        const std::size_t output_offset = output_row * embedding_length;

        const std::size_t weight_offset = weight_row * embedding_length;

        for (std::size_t column = static_cast<std::size_t>(threadIdx.x); column < embedding_length;
             column += static_cast<std::size_t>(blockDim.x)) {
            out[output_offset + column] = weight[weight_offset + column];
        }
    }
}

// ============================================================
// 128-bit vectorized Embedding kernel
// ============================================================
//
// Each thread copies one 128-bit value at a time.
//
// This kernel may only be launched when:
//
//   1. out is PACKED_128_ALIGNMENT aligned;
//   2. weight is PACKED_128_ALIGNMENT aligned;
//   3. every embedding row contains a whole number of
//      Packed128 objects.
//
// Elements per 128-bit pack:
//
//   FP32 -> 4
//   FP16 -> 8
//   BF16 -> 8
// ============================================================

template <typename T>
__global__ void embedding_vectorized_kernel(
    T *__restrict__ out,
    const T *__restrict__ weight,
    const std::int64_t *__restrict__ index,
    std::size_t index_count,
    std::size_t embedding_length,
    std::size_t vocabulary_size) {
    static_assert(
        PACKED_128_BYTES % sizeof(T) == 0,
        "Embedding: data type size must divide the 128-bit pack size.");

    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    const std::size_t packs_per_row = embedding_length / elements_per_pack;

    for (std::size_t output_row = static_cast<std::size_t>(blockIdx.x); output_row < index_count;
         output_row += static_cast<std::size_t>(gridDim.x)) {
        const std::int64_t signed_weight_row = index[output_row];

        if (signed_weight_row < 0
            || static_cast<std::uint64_t>(signed_weight_row)
                   >= static_cast<std::uint64_t>(vocabulary_size)) {
            continue;
        }

        const std::size_t weight_row = static_cast<std::size_t>(signed_weight_row);

        T *const output_row_pointer = out + output_row * embedding_length;

        const T *const weight_row_pointer = weight + weight_row * embedding_length;

        auto *const packed_output = reinterpret_cast<Packed128 *>(output_row_pointer);

        const auto *const packed_weight = reinterpret_cast<const Packed128 *>(weight_row_pointer);

        for (std::size_t pack_index = static_cast<std::size_t>(threadIdx.x);
             pack_index < packs_per_row;
             pack_index += static_cast<std::size_t>(blockDim.x)) {
            packed_output[pack_index] = packed_weight[pack_index];
        }
    }
}

// ============================================================
// Vectorized-path eligibility
// ============================================================
//
// Base alignment alone is insufficient.
//
// If one row occupies a non-multiple of 16 bytes, later rows
// will no longer begin at a Packed128-aligned address.
// ============================================================

template <typename T>
inline bool
can_use_vectorized_embedding(const T *out, const T *weight, std::size_t embedding_length) {
    const std::size_t row_bytes = embedding_length * sizeof(T);

    const bool base_addresses_aligned = are_aligned<PACKED_128_ALIGNMENT>(out, weight);

    const bool every_row_aligned = row_bytes % PACKED_128_BYTES == 0;

    return base_addresses_aligned && every_row_aligned;
}

// ============================================================
// Per-row logical work items
// ============================================================
//
// Shared algorithm determines how much logical work exists
// inside each row.
//
// Vendor adapter determines the physical block size.
// ============================================================

template <typename T>
inline std::size_t
get_embedding_row_work_items(std::size_t embedding_length, bool use_vectorized_kernel) {
    if (!use_vectorized_kernel) { return embedding_length; }

    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    return embedding_length / elements_per_pack;
}

// ============================================================
// Shared CUDA-compatible launcher
// ============================================================
//
// Shared layer owns:
//
//   - scalar copy algorithm
//   - Packed128 copy algorithm
//   - index lookup
//   - vectorized-path requirements
//
// Vendor adapter owns:
//
//   - grid-size policy
//   - block-size policy
//   - stream conversion
//   - launch error handling
// ============================================================

template <typename T, typename StreamT>
inline void launch_embedding_kernel(
    T *out,
    const std::int64_t *index,
    const T *weight,
    std::size_t index_count,
    std::size_t embedding_length,
    std::size_t vocabulary_size,
    unsigned int block_size,
    std::size_t grid_size,
    bool use_vectorized_kernel,
    StreamT stream) {
    if (index_count == 0) { return; }

    if (use_vectorized_kernel) {
        embedding_vectorized_kernel<T>
            <<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
                out, weight, index, index_count, embedding_length, vocabulary_size);

        return;
    }

    embedding_scalar_kernel<T><<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
        out, weight, index, index_count, embedding_length, vocabulary_size);
}

} // namespace llaisys::ops::cuda_compat
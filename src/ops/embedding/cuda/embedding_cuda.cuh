#pragma once

#include "../../cuda/common.cuh"

#include "embedding_cuda.hpp"
#include "../../../utils.hpp"
#include "../embedding_config.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace llaisys::ops::cuda {
namespace {

struct EmbeddingLaunchPlan {
    std::size_t block_size;
    std::size_t grid_size;
    std::size_t index_count;
    std::size_t row_work_items;
    bool use_vectorized_kernel;
};

template <typename T>
inline void validate_embedding_arguments(
    T *out,
    const std::int64_t *index,
    const T *weight,
    std::size_t numel,
    std::size_t embedding_length,
    std::size_t vocabulary_size) {
    CHECK_ARGUMENT(embedding_length > 0, "Embedding: embedding length must be greater than zero.");

    CHECK_ARGUMENT(
        numel % embedding_length == 0,
        "Embedding: output element count must be divisible by embedding length.");

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
}

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
             pack_index < packs_per_row; pack_index += static_cast<std::size_t>(blockDim.x)) {
            packed_output[pack_index] = packed_weight[pack_index];
        }
    }
}

template <typename T>
inline bool
can_use_vectorized_embedding(const T *out, const T *weight, std::size_t embedding_length) {
    const std::size_t row_bytes = embedding_length * sizeof(T);

    return are_aligned<PACKED_128_ALIGNMENT>(out, weight) && row_bytes % PACKED_128_BYTES == 0;
}

template <typename T>
inline std::size_t get_row_work_items(std::size_t embedding_length, bool use_vectorized_kernel) {
    if (!use_vectorized_kernel) { return embedding_length; }

    return embedding_length / PACKED_128_ELEMENTS<T>;
}

template <typename T>
inline EmbeddingLaunchPlan get_launch_plan(
    T *out,
    const T *weight,
    std::size_t numel,
    std::size_t embedding_length,
    std::size_t block_size,
    std::size_t max_blocks,
    bool allow_vectorized) {
    if (numel == 0) {
        return EmbeddingLaunchPlan{
            block_size, 0, 0, 0, false,
        };
    }

    const std::size_t index_count = numel / embedding_length;

    const bool use_vectorized_kernel
        = allow_vectorized && can_use_vectorized_embedding<T>(out, weight, embedding_length);

    const std::size_t row_work_items
        = get_row_work_items<T>(embedding_length, use_vectorized_kernel);

    CHECK_ARGUMENT(row_work_items > 0, "Embedding: row work item count must be greater than zero.");

    const std::size_t grid_size = cap_grid_size(index_count, max_blocks);

    CHECK_ARGUMENT(
        grid_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "Embedding: grid size exceeds the supported launch range.");

    return EmbeddingLaunchPlan{
        block_size, grid_size, index_count, row_work_items, use_vectorized_kernel,
    };
}

template <typename T>
void launch_embedding(
    T *out,
    const std::int64_t *index,
    const T *weight,
    std::size_t numel,
    std::size_t embedding_length,
    std::size_t vocabulary_size,
    cudaStream_t stream) {
    validate_embedding_arguments<T>(out, index, weight, numel, embedding_length, vocabulary_size);

    if (numel == 0) { return; }

    const embedding_config::LaunchPolicy &policy = embedding_config::get_launch_policy();

    const EmbeddingLaunchPlan plan = get_launch_plan<T>(
        out, weight, numel, embedding_length, policy.block_size, policy.max_blocks,
        policy.allow_vectorized);

    if (config::debug_enabled()) {
        std::fprintf(
            stderr,
            "[Embedding][%s] implementation=shared numel=%zu index_count=%zu "
            "embedding_length=%zu kernel=%s row_work_items=%zu block=%zu grid=%zu "
            "max_blocks=%zu\n",
            GPU_BACKEND_NAME, numel, plan.index_count, embedding_length,
            plan.use_vectorized_kernel ? "vectorized128" : "scalar", plan.row_work_items,
            plan.block_size, plan.grid_size, policy.max_blocks);
    }

    const dim3 block_dimension(static_cast<unsigned int>(plan.block_size));
    const dim3 grid_dimension(static_cast<unsigned int>(plan.grid_size));

    if (plan.use_vectorized_kernel) {
        embedding_vectorized_kernel<T><<<grid_dimension, block_dimension, 0, stream>>>(
            out, weight, index, plan.index_count, embedding_length, vocabulary_size);
    } else {
        embedding_scalar_kernel<T><<<grid_dimension, block_dimension, 0, stream>>>(
            out, weight, index, plan.index_count, embedding_length, vocabulary_size);
    }

    check_kernel("Embedding kernel");
}

} // namespace

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    std::size_t numel,
    std::size_t embedding_length,
    std::size_t vocabulary_size,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    return dispatch_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_embedding<T>(
            reinterpret_cast<T *>(out), reinterpret_cast<const std::int64_t *>(index),
            reinterpret_cast<const T *>(weight), numel, embedding_length, vocabulary_size,
            cuda_stream);
    });
}

} // namespace llaisys::ops::cuda

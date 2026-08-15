#pragma once

#include "../../cuda/common.cuh"

#include "rope_cuda.hpp"
#include "../../../utils.hpp"
#include "../rope_config.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace llaisys::ops::cuda {
namespace {

// ============================================================
// RoPE angle
// ============================================================
//
// Preserve the original Float32 evaluation order:
//
//     exponent    = 2 * pair / dimension
//     denominator = theta ^ exponent
//     angle       = position / denominator
//
// Do not replace this with a reciprocal, negative exponent, or
// an exp/log reformulation. Those variants can round differently.
// ============================================================

__device__ __forceinline__ float
rope_angle(float position, std::size_t pair_index, std::size_t dimension, float theta) {
    const float exponent = 2.0F * static_cast<float>(pair_index) / static_cast<float>(dimension);

    const float denominator = powf(theta, exponent);

    return position / denominator;
}

// ============================================================
// Shared direct kernel
// ============================================================
//
// This is the primary controlled cross-GPU baseline.
//
// One block processes one token at a time. Each thread owns one
// or more dimension pairs, computes sin/cos once for that pair,
// then reuses it across all heads.
//
// NVIDIA and MetaX compile exactly this source with the same:
//   - block size
//   - grid cap
//   - arithmetic order
//   - sincosf implementation request
//   - FP32 intermediate arithmetic
//
// out == in is supported because both pair values are loaded
// before either output location is written.
// ============================================================

template <typename T>
__global__ void rope_direct_kernel(
    T *out,
    const T *in,
    const std::int64_t *__restrict__ position_ids,
    float theta,
    std::size_t sequence_length,
    std::size_t head_count,
    std::size_t dimension) {
    const std::size_t half_dimension = dimension / 2;

    for (std::size_t token = static_cast<std::size_t>(blockIdx.x); token < sequence_length;
         token += static_cast<std::size_t>(gridDim.x)) {
        const float position = static_cast<float>(position_ids[token]);

        const std::size_t token_offset = token * head_count * dimension;

        for (std::size_t pair = static_cast<std::size_t>(threadIdx.x); pair < half_dimension;
             pair += static_cast<std::size_t>(blockDim.x)) {
            const float angle = rope_angle(position, pair, dimension, theta);

            float sine;
            float cosine;

            sincosf(angle, &sine, &cosine);

            for (std::size_t head = 0; head < head_count; ++head) {
                const std::size_t vector_offset = token_offset + head * dimension;

                const std::size_t low_index = vector_offset + pair;

                const std::size_t high_index = low_index + half_dimension;

                const float low = to_float<T>(in[low_index]);
                const float high = to_float<T>(in[high_index]);

                const float rotated_low = low * cosine - high * sine;

                const float rotated_high = high * cosine + low * sine;

                out[low_index] = from_float<T>(rotated_low);
                out[high_index] = from_float<T>(rotated_high);
            }
        }
    }
}

// ============================================================
// Shared cached kernel
// ============================================================
//
// This is an explicit same-source ablation:
//
//     LLAISYS_ROPE_IMPL=cached
//
// It is NOT selected automatically.
//
// Each block caches one token's cosine/sine table in dynamic
// shared memory and reuses it across heads.
//
// For a fair cross-GPU cached experiment, only compare shapes
// whose shared-memory requirement is supported by both platforms.
// ============================================================

template <typename T>
__global__ void rope_cached_kernel(
    T *out,
    const T *in,
    const std::int64_t *__restrict__ position_ids,
    float theta,
    std::size_t sequence_length,
    std::size_t head_count,
    std::size_t dimension) {
    extern __shared__ float trigonometric_cache[];

    const std::size_t half_dimension = dimension / 2;

    float *const cosine_cache = trigonometric_cache;
    float *const sine_cache = trigonometric_cache + half_dimension;

    for (std::size_t token = static_cast<std::size_t>(blockIdx.x); token < sequence_length;
         token += static_cast<std::size_t>(gridDim.x)) {
        const float position = static_cast<float>(position_ids[token]);

        for (std::size_t pair = static_cast<std::size_t>(threadIdx.x); pair < half_dimension;
             pair += static_cast<std::size_t>(blockDim.x)) {
            const float angle = rope_angle(position, pair, dimension, theta);

            float sine;
            float cosine;

            sincosf(angle, &sine, &cosine);

            cosine_cache[pair] = cosine;
            sine_cache[pair] = sine;
        }

        __syncthreads();

        const std::size_t pair_count = head_count * half_dimension;

        const std::size_t token_offset = token * head_count * dimension;

        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < pair_count;
             index += static_cast<std::size_t>(blockDim.x)) {
            const std::size_t head = index / half_dimension;

            const std::size_t pair = index - head * half_dimension;

            const std::size_t vector_offset = token_offset + head * dimension;

            const std::size_t low_index = vector_offset + pair;

            const std::size_t high_index = low_index + half_dimension;

            const float low = to_float<T>(in[low_index]);
            const float high = to_float<T>(in[high_index]);

            const float cosine = cosine_cache[pair];
            const float sine = sine_cache[pair];

            const float rotated_low = low * cosine - high * sine;

            const float rotated_high = high * cosine + low * sine;

            out[low_index] = from_float<T>(rotated_low);
            out[high_index] = from_float<T>(rotated_high);
        }

        // This block may process another token because the grid is capped.
        // Ensure the current cache is no longer being read before reuse.
        __syncthreads();
    }
}

inline std::size_t get_cache_bytes(std::size_t half_dimension) {
    CHECK_ARGUMENT(
        half_dimension <= std::numeric_limits<std::size_t>::max() / (2 * sizeof(float)),
        "RoPE: cached shared-memory size overflows size_t.");

    return 2 * half_dimension * sizeof(float);
}

// ============================================================
// Shared launcher
// ============================================================

template <typename T>
void launch_rope(
    T *out,
    const T *in,
    const std::int64_t *position_ids,
    float theta,
    std::size_t sequence_length,
    std::size_t head_count,
    std::size_t dimension,
    cudaStream_t stream) {
    const std::size_t vector_count = utils::checked_product(
        sequence_length, head_count, "RoPE: sequence/head count overflows size_t.");

    const std::size_t element_count = utils::checked_product(
        vector_count, dimension, "RoPE: tensor element count overflows size_t.");

    CHECK_ARGUMENT(element_count == 0 || out != nullptr, "RoPE: output pointer must not be null.");

    CHECK_ARGUMENT(element_count == 0 || in != nullptr, "RoPE: input pointer must not be null.");

    CHECK_ARGUMENT(
        sequence_length == 0 || position_ids != nullptr,
        "RoPE: position-id pointer must not be null.");

    CHECK_ARGUMENT(
        sequence_length == 0 || head_count > 0,
        "RoPE: head count must be greater than zero for a nonempty sequence.");

    CHECK_ARGUMENT(dimension > 0, "RoPE: head dimension must be greater than zero.");

    CHECK_ARGUMENT(dimension % 2 == 0, "RoPE: head dimension must be even.");

    CHECK_ARGUMENT(
        std::isfinite(theta) && theta > 0.0F, "RoPE: theta must be finite and greater than zero.");

    if (element_count == 0) { return; }

    const std::size_t half_dimension = dimension / 2;

    CHECK_ARGUMENT(half_dimension > 0, "RoPE: half dimension must be greater than zero.");

    const std::size_t block_size = rope_config::block_size();

    CHECK_ARGUMENT(
        block_size > 0
            && block_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "RoPE: block size exceeds the supported launch range.");

    const std::size_t grid_size = cap_grid_size(sequence_length, rope_config::MAX_BLOCKS);

    CHECK_ARGUMENT(grid_size > 0, "RoPE: grid size must be greater than zero.");

    CHECK_ARGUMENT(
        grid_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "RoPE: grid size exceeds the supported launch range.");

    const rope_config::Implementation implementation = rope_config::implementation();

    if (config::debug_enabled()) {
        std::fprintf(
            stderr,
            "[RoPE][%s] implementation=%s seq=%zu heads=%zu dim=%zu "
            "block=%zu grid=%zu shared_bytes=%zu\n",
            GPU_BACKEND_NAME,
            implementation == rope_config::Implementation::DIRECT ? "shared_direct"
                                                                  : "shared_cached",
            sequence_length, head_count, dimension, block_size, grid_size,
            implementation == rope_config::Implementation::CACHED ? get_cache_bytes(half_dimension)
                                                                  : 0);
    }

    const dim3 grid_dimension(static_cast<unsigned int>(grid_size));

    const dim3 block_dimension(static_cast<unsigned int>(block_size));

    if (implementation == rope_config::Implementation::CACHED) {
        const std::size_t shared_memory_bytes = get_cache_bytes(half_dimension);

        rope_cached_kernel<T><<<grid_dimension, block_dimension, shared_memory_bytes, stream>>>(
            out, in, position_ids, theta, sequence_length, head_count, dimension);
    } else {
        rope_direct_kernel<T><<<grid_dimension, block_dimension, 0, stream>>>(
            out, in, position_ids, theta, sequence_length, head_count, dimension);
    }

    check_kernel("RoPE kernel");
}

} // namespace

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    float theta,
    llaisysDataType_t type,
    std::size_t sequence_length,
    std::size_t head_count,
    std::size_t head_dimension,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    const auto *const position_ids = reinterpret_cast<const std::int64_t *>(pos_ids);

    return dispatch_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_rope<T>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), position_ids, theta,
            sequence_length, head_count, head_dimension, cuda_stream);
    });
}

} // namespace llaisys::ops::cuda

#pragma once

#include "../../cuda_compat/common.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cuda_compat {

// ============================================================
// RoPE angle
// ============================================================
//
// Preserve the Float32 evaluation order:
//
//     exponent    = 2 * pair / dimension
//     denominator = theta ^ exponent
//     angle       = position / denominator
//
// Do not rewrite this using:
//
//     position * reciprocal
//     powf(theta, -exponent)
//     exp2f(...)
//
// Mathematically equivalent expressions may round differently.
// ============================================================

__device__ __forceinline__ float
rope_angle(float position, std::size_t pair_index, std::size_t dimension, float theta) {
    const float exponent = 2.0F * static_cast<float>(pair_index) / static_cast<float>(dimension);

    const float denominator = powf(theta, exponent);

    return position / denominator;
}

// ============================================================
// Shared-memory cached kernel
// ============================================================
//
// One block processes one token at a time.
//
// Trigonometric values are calculated once for each dimension
// pair and reused by every attention head.
//
// out and in intentionally do not use __restrict__ because
// exact in-place execution (out == in) is supported.
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

    float *cosine_cache = trigonometric_cache;

    float *sine_cache = trigonometric_cache + half_dimension;

    for (std::size_t token = static_cast<std::size_t>(blockIdx.x); token < sequence_length;
         token += static_cast<std::size_t>(gridDim.x)) {
        const float position = static_cast<float>(position_ids[token]);

        // ====================================================
        // Build trigonometric cache
        // ====================================================

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

        // ====================================================
        // Rotate all [head, pair] values
        // ====================================================

        const std::size_t pair_count = head_count * half_dimension;

        const std::size_t token_offset = token * head_count * dimension;

        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < pair_count;
             index += static_cast<std::size_t>(blockDim.x)) {
            const std::size_t head = index / half_dimension;

            const std::size_t pair = index - head * half_dimension;

            const std::size_t vector_offset = token_offset + head * dimension;

            const std::size_t low_index = vector_offset + pair;

            const std::size_t high_index = low_index + half_dimension;

            // Load both values before either output is written.
            // This preserves out == in.

            const float low = to_float<T>(in[low_index]);

            const float high = to_float<T>(in[high_index]);

            const float cosine = cosine_cache[pair];

            const float sine = sine_cache[pair];

            const float rotated_low = low * cosine - high * sine;

            const float rotated_high = high * cosine + low * sine;

            out[low_index] = from_float<T>(rotated_low);

            out[high_index] = from_float<T>(rotated_high);
        }

        // This block may process another token because the grid
        // can be capped. Make sure nobody is still reading the
        // current cache before it is overwritten.
        __syncthreads();
    }
}

// ============================================================
// Direct kernel
// ============================================================
//
// No shared-memory trigonometric cache.
//
// Each thread owns one or more dimension pairs, calculates the
// corresponding sine/cosine pair, and reuses it across heads.
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

                // Load before write for in-place safety.

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
// Dynamic shared-memory requirement
// ============================================================

inline std::size_t get_rope_cache_bytes(std::size_t half_dimension) {
    return 2 * half_dimension * sizeof(float);
}

// ============================================================
// Shared CUDA-compatible launcher
// ============================================================
//
// Algorithm layer owns:
//
//   - RoPE arithmetic
//   - cached/direct implementations
//   - dynamic shared-memory layout
//
// Vendor adapter owns:
//
//   - cached/direct selection policy
//   - block-size selection
//   - grid-size selection
//   - stream conversion
//   - launch error handling
// ============================================================

template <typename T, typename StreamT>
inline void launch_rope_kernel(
    T *out,
    const T *in,
    const std::int64_t *position_ids,
    float theta,
    std::size_t sequence_length,
    std::size_t head_count,
    std::size_t dimension,
    unsigned int block_size,
    std::size_t grid_size,
    bool use_cached_kernel,
    StreamT stream) {
    if (sequence_length == 0) { return; }

    if (use_cached_kernel) {
        const std::size_t half_dimension = dimension / 2;

        const std::size_t shared_memory_bytes = get_rope_cache_bytes(half_dimension);

        rope_cached_kernel<T>
            <<<static_cast<unsigned int>(grid_size), block_size, shared_memory_bytes, stream>>>(
                out, in, position_ids, theta, sequence_length, head_count, dimension);

        return;
    }

    rope_direct_kernel<T><<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
        out, in, position_ids, theta, sequence_length, head_count, dimension);
}

} // namespace llaisys::ops::cuda_compat
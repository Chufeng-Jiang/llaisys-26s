#pragma once

#include "../../cuda_compat/common.cuh"

#include <cfloat>
#include <cmath>
#include <cstddef>

namespace llaisys::ops::cuda_compat {

// ============================================================
// Portable block reduction
// ============================================================
//
// Deliberately avoids:
//
//     __shfl_*
//     warp-size assumptions
//     vendor reduction libraries
//
// The reduction works for arbitrary positive block sizes.
// ============================================================

__device__ __forceinline__ float
self_attention_block_reduce_sum(float value, float *shared_reduction) {
    const unsigned int thread_index = threadIdx.x;

    shared_reduction[thread_index] = value;

    __syncthreads();

    unsigned int active_count = blockDim.x;

    while (active_count > 1) {
        const unsigned int next_count = (active_count + 1U) / 2U;

        const unsigned int pair_count = active_count / 2U;

        if (thread_index < pair_count) {
            shared_reduction[thread_index] += shared_reduction[thread_index + next_count];
        }

        __syncthreads();

        active_count = next_count;
    }

    return shared_reduction[0];
}

// ============================================================
// CUDA-compatible SelfAttention baseline
// ============================================================
//
// Tensor layout:
//
//     output: [seqlen, nhead, dv]
//     Q:      [seqlen, nhead, d]
//     K:      [total_len, nkvhead, d]
//     V:      [total_len, nkvhead, dv]
//
// Supports:
//     - causal attention
//     - GQA
//     - MQA
//     - FP32 accumulation
//     - online softmax
//
// One block processes one (query, query-head) task at a time.
//
// Unlike the NVIDIA optimized implementation, this baseline:
//     - does not assume warp size 32;
//     - does not use __shfl_down_sync;
//     - does not use cuDNN;
//     - does not require a temporary score buffer.
//
// It prioritizes portability and correctness rather than
// maximum performance.
// ============================================================

template <typename T>
__global__ void self_attention_portable_kernel(
    T *__restrict__ attn_val,
    const T *__restrict__ q,
    const T *__restrict__ k,
    const T *__restrict__ v,
    float scale,
    std::size_t seqlen,
    std::size_t nhead,
    std::size_t dv,
    std::size_t total_len,
    std::size_t nkvhead,
    std::size_t d) {
    extern __shared__ float shared[];

    // Shared-memory layout:
    //
    //     shared_q[d]
    //     shared_output[dv]
    //     shared_reduction[blockDim.x]
    //     shared_state[4]
    //
    // shared_state:
    //
    //     [0] running maximum
    //     [1] running softmax denominator
    //     [2] old accumulator scale
    //     [3] current-key weight

    float *shared_q = shared;

    float *shared_output = shared_q + d;

    float *shared_reduction = shared_output + dv;

    float *shared_state = shared_reduction + blockDim.x;

    const std::size_t task_count = seqlen * nhead;

    const std::size_t group_size = nhead / nkvhead;

    const std::size_t prefix_length = total_len - seqlen;

    for (std::size_t task = static_cast<std::size_t>(blockIdx.x); task < task_count;
         task += static_cast<std::size_t>(gridDim.x)) {
        const std::size_t query_index = task / nhead;

        const std::size_t query_head = task - query_index * nhead;

        const std::size_t kv_head = query_head / group_size;

        // Bottom-right causal alignment:
        //
        // query 0 sees:
        //
        //     prefix_length + 1
        //
        // keys.
        const std::size_t causal_length = prefix_length + query_index + 1;

        const T *query = q + (query_index * nhead + query_head) * d;

        // ====================================================
        // Cache Q in FP32
        // ====================================================

        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < d;
             index += static_cast<std::size_t>(blockDim.x)) {
            shared_q[index] = to_float<T>(query[index]);
        }

        // ====================================================
        // Initialize output accumulator
        // ====================================================

        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dv;
             index += static_cast<std::size_t>(blockDim.x)) {
            shared_output[index] = 0.0F;
        }

        if (threadIdx.x == 0) {
            shared_state[0] = -FLT_MAX;

            shared_state[1] = 0.0F;

            shared_state[2] = 0.0F;

            shared_state[3] = 0.0F;
        }

        __syncthreads();

        // ====================================================
        // Online causal attention
        // ====================================================
        //
        // For every visible key:
        //
        //     score = Q dot K * scale
        //
        // and update the softmax/output accumulator using the
        // numerically stable online-softmax recurrence.
        // ====================================================

        for (std::size_t key_index = 0; key_index < causal_length; ++key_index) {
            const T *key = k + (key_index * nkvhead + kv_head) * d;

            // =================================================
            // Q dot K
            // =================================================

            float partial_dot = 0.0F;

            for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < d;
                 index += static_cast<std::size_t>(blockDim.x)) {
                partial_dot += shared_q[index] * to_float<T>(key[index]);
            }

            const float dot = self_attention_block_reduce_sum(partial_dot, shared_reduction);

            // =================================================
            // Online softmax state
            // =================================================

            if (threadIdx.x == 0) {
                const float score = dot * scale;

                const float running_max = shared_state[0];

                const float new_max = fmaxf(running_max, score);

                const float old_scale
                    = running_max == -FLT_MAX ? 0.0F : expf(running_max - new_max);

                const float key_weight = expf(score - new_max);

                shared_state[0] = new_max;

                shared_state[2] = old_scale;

                shared_state[3] = key_weight;

                shared_state[1] = shared_state[1] * old_scale + key_weight;
            }

            __syncthreads();

            const float old_scale = shared_state[2];

            const float key_weight = shared_state[3];

            // =================================================
            // Weighted V accumulation
            // =================================================

            const T *value = v + (key_index * nkvhead + kv_head) * dv;

            for (std::size_t value_index = static_cast<std::size_t>(threadIdx.x); value_index < dv;
                 value_index += static_cast<std::size_t>(blockDim.x)) {
                shared_output[value_index] = shared_output[value_index] * old_scale
                                           + key_weight * to_float<T>(value[value_index]);
            }

            __syncthreads();
        }

        // ====================================================
        // Final normalization
        // ====================================================

        T *output = attn_val + (query_index * nhead + query_head) * dv;

        const float denominator = shared_state[1];

        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dv;
             index += static_cast<std::size_t>(blockDim.x)) {
            output[index] = from_float<T>(shared_output[index] / denominator);
        }

        // The same physical block may process another logical
        // attention task because the grid can be capped.
        __syncthreads();
    }
}

// ============================================================
// Portable launcher
// ============================================================
//
// Vendor adapter supplies:
//
//     block size
//     grid size
//     dynamic shared-memory size
//     stream
//
// Device capability checks remain vendor-specific.
// ============================================================

template <typename T, typename StreamT>
inline void launch_self_attention_portable(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    float scale,
    std::size_t seqlen,
    std::size_t nhead,
    std::size_t dv,
    std::size_t total_len,
    std::size_t nkvhead,
    std::size_t d,
    unsigned int block_size,
    std::size_t grid_size,
    std::size_t shared_memory_bytes,
    StreamT stream) {
    if (seqlen == 0) { return; }

    self_attention_portable_kernel<T>
        <<<static_cast<unsigned int>(grid_size), block_size, shared_memory_bytes, stream>>>(
            attn_val, q, k, v, scale, seqlen, nhead, dv, total_len, nkvhead, d);
}

} // namespace llaisys::ops::cuda_compat

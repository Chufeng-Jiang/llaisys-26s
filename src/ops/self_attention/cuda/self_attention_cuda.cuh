#pragma once

#include "../../cuda/common.cuh"
#include "self_attention_cuda.hpp"
#include "../../../utils.hpp"
#include "../self_attention_config.hpp"

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>

namespace llaisys::ops::cuda {
namespace {

// ============================================================
// Shared block reduction
// ============================================================
//
// Deliberately avoids:
//
//     __shfl_*
//     warp-size assumptions
//     CUB
//     vendor attention libraries
//
// This reduction works for any positive block size and is compiled
// unchanged by NVCC and MXCC.
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
// Shared causal GQA/MQA Self-Attention kernel
// ============================================================
//
// Tensor layout:
//
//     output: [seqlen, nhead, dv]
//     Q:      [seqlen, nhead, d]
//     K:      [total_len, nkvhead, d]
//     V:      [total_len, nkvhead, dv]
//
// Semantics:
//
//     output = causal_softmax(Q K^T * scale) V
//
// Supports:
//
//     causal attention
//     bottom-right causal alignment
//     GQA
//     MQA
//     FP32 accumulation
//     numerically stable online softmax
//
// One block processes one (query, query-head) task at a time.
//
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
//
// The exact same kernel source, reduction order, block size, grid
// cap, arithmetic expressions, and shared-memory layout are used
// for NVIDIA and MetaX in the controlled baseline.
// ============================================================

template <typename T>
__global__ void self_attention_kernel(
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
        //     visible keys
        //       =
        //     total_len - seqlen + query_index + 1
        const std::size_t causal_length = prefix_length + query_index + 1;

        const T *const query = q + (query_index * nhead + query_head) * d;

        // ----------------------------------------------------
        // Cache Q in FP32.
        // ----------------------------------------------------

        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < d;
             index += static_cast<std::size_t>(blockDim.x)) {
            shared_q[index] = to_float<T>(query[index]);
        }

        // ----------------------------------------------------
        // Initialize output accumulator.
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // Online causal attention.
        // ----------------------------------------------------

        for (std::size_t key_index = 0; key_index < causal_length; ++key_index) {
            const T *const key = k + (key_index * nkvhead + kv_head) * d;

            float partial_dot = 0.0F;

            for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < d;
                 index += static_cast<std::size_t>(blockDim.x)) {
                partial_dot += shared_q[index] * to_float<T>(key[index]);
            }

            const float dot = self_attention_block_reduce_sum(partial_dot, shared_reduction);

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

            const T *const value = v + (key_index * nkvhead + kv_head) * dv;

            for (std::size_t value_index = static_cast<std::size_t>(threadIdx.x); value_index < dv;
                 value_index += static_cast<std::size_t>(blockDim.x)) {
                shared_output[value_index] = shared_output[value_index] * old_scale
                                           + key_weight * to_float<T>(value[value_index]);
            }

            __syncthreads();
        }

        // ----------------------------------------------------
        // Final normalization.
        // ----------------------------------------------------

        T *const output = attn_val + (query_index * nhead + query_head) * dv;

        const float denominator = shared_state[1];

        for (std::size_t index = static_cast<std::size_t>(threadIdx.x); index < dv;
             index += static_cast<std::size_t>(blockDim.x)) {
            output[index] = from_float<T>(shared_output[index] / denominator);
        }

        // The physical block may process another logical task
        // because the grid can be capped.
        __syncthreads();
    }
}

// ============================================================
// Shared validation helpers
// ============================================================

struct AttentionElementCounts {
    std::size_t output;
    std::size_t query;
    std::size_t key;
    std::size_t value;
    std::size_t tasks;
};

inline AttentionElementCounts get_attention_element_counts(
    std::size_t seqlen,
    std::size_t nhead,
    std::size_t dv,
    std::size_t total_len,
    std::size_t nkvhead,
    std::size_t d) {
    const std::size_t task_count
        = utils::checked_product(seqlen, nhead, "SelfAttention: task count overflows size_t.");

    const std::size_t output_elements = utils::checked_product(
        task_count, dv, "SelfAttention: output element count overflows size_t.");

    const std::size_t query_elements = utils::checked_product(
        task_count, d, "SelfAttention: query element count overflows size_t.");

    const std::size_t kv_vector_count = utils::checked_product(
        total_len, nkvhead, "SelfAttention: KV vector count overflows size_t.");

    const std::size_t key_elements = utils::checked_product(
        kv_vector_count, d, "SelfAttention: key element count overflows size_t.");

    const std::size_t value_elements = utils::checked_product(
        kv_vector_count, dv, "SelfAttention: value element count overflows size_t.");

    return AttentionElementCounts{
        output_elements, query_elements, key_elements, value_elements, task_count,
    };
}

inline std::size_t get_shared_memory_bytes(std::size_t d, std::size_t dv, std::size_t block_size) {
    CHECK_ARGUMENT(
        d <= std::numeric_limits<std::size_t>::max() - dv,
        "SelfAttention: shared-memory element count overflows size_t.");

    const std::size_t q_output_elements = d + dv;

    CHECK_ARGUMENT(
        block_size <= std::numeric_limits<std::size_t>::max() - q_output_elements - 4,
        "SelfAttention: shared-memory element count overflows size_t.");

    const std::size_t shared_elements = q_output_elements + block_size + 4;

    CHECK_ARGUMENT(
        shared_elements <= std::numeric_limits<std::size_t>::max() / sizeof(float),
        "SelfAttention: shared-memory byte count overflows size_t.");

    return shared_elements * sizeof(float);
}

// ============================================================
// Shared launcher
// ============================================================
//
// No backend-specific device capability query is performed in the
// controlled baseline. If a shape exceeds the dynamic shared-memory
// limit of either platform, the launch failure remains explicit.
//
// This avoids comparing:
//     NVIDIA opt-in shared-memory policy
// against
//     MetaX default launch behavior.
//
// Capability-specific tuning can be studied separately.
// ============================================================

template <typename T>
void launch_self_attention(
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
    cudaStream_t stream) {
    CHECK_ARGUMENT(nhead > 0, "SelfAttention: query head count must be greater than zero.");

    CHECK_ARGUMENT(nkvhead > 0, "SelfAttention: KV head count must be greater than zero.");

    CHECK_ARGUMENT(
        nhead % nkvhead == 0,
        "SelfAttention: query head count must be a multiple of KV head count.");

    CHECK_ARGUMENT(
        total_len >= seqlen,
        "SelfAttention: total KV length must not be smaller than query length.");

    CHECK_ARGUMENT(d > 0, "SelfAttention: query/key head dimension must be greater than zero.");

    CHECK_ARGUMENT(dv > 0, "SelfAttention: value head dimension must be greater than zero.");

    CHECK_ARGUMENT(std::isfinite(scale), "SelfAttention: scale must be finite.");

    const AttentionElementCounts counts
        = get_attention_element_counts(seqlen, nhead, dv, total_len, nkvhead, d);

    CHECK_ARGUMENT(
        counts.output == 0 || attn_val != nullptr,
        "SelfAttention: output pointer must not be null.");

    CHECK_ARGUMENT(
        counts.query == 0 || q != nullptr, "SelfAttention: query pointer must not be null.");

    CHECK_ARGUMENT(counts.key == 0 || k != nullptr, "SelfAttention: key pointer must not be null.");

    CHECK_ARGUMENT(
        counts.value == 0 || v != nullptr, "SelfAttention: value pointer must not be null.");

    if (counts.output == 0) { return; }

    const std::size_t block_size = self_attention_config::block_size();

    CHECK_ARGUMENT(
        block_size > 0
            && block_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "SelfAttention: block size exceeds the supported launch range.");

    const std::size_t grid_size = cap_grid_size(counts.tasks, self_attention_config::MAX_BLOCKS);

    CHECK_ARGUMENT(grid_size > 0, "SelfAttention: grid size must be greater than zero.");

    CHECK_ARGUMENT(
        grid_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "SelfAttention: grid size exceeds the supported launch range.");

    const std::size_t shared_memory_bytes = get_shared_memory_bytes(d, dv, block_size);

    if (config::debug_enabled()) {
        std::fprintf(
            stderr,
            "[SelfAttention][%s] implementation=shared_online "
            "seqlen=%zu nhead=%zu nkvhead=%zu d=%zu dv=%zu "
            "total_len=%zu block=%zu grid=%zu shared_bytes=%zu "
            "accumulation=f32\n",
            GPU_BACKEND_NAME, seqlen, nhead, nkvhead, d, dv, total_len, block_size, grid_size,
            shared_memory_bytes);
    }

    self_attention_kernel<T>
        <<<static_cast<unsigned int>(grid_size), static_cast<unsigned int>(block_size),
           shared_memory_bytes, stream>>>(
            attn_val, q, k, v, scale, seqlen, nhead, dv, total_len, nkvhead, d);

    check_kernel("SelfAttention kernel");
}

} // namespace

void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    float scale,
    llaisysDataType_t type,
    std::size_t seqlen,
    std::size_t nhead,
    std::size_t dv,
    std::size_t total_len,
    std::size_t nkvhead,
    std::size_t d,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    return dispatch_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_self_attention<T>(
            reinterpret_cast<T *>(attn_val), reinterpret_cast<const T *>(q),
            reinterpret_cast<const T *>(k), reinterpret_cast<const T *>(v), scale, seqlen, nhead,
            dv, total_len, nkvhead, d, cuda_stream);
    });
}

} // namespace llaisys::ops::cuda

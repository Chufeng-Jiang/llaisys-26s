#pragma once

#include <cstddef>

namespace llaisys::ops::cuda_compat {

// ============================================================
// Linear bias broadcast
// ============================================================
//
// Input:
//
//     bias: [output_features]
//
// Output:
//
//     out: [row_count, output_features]
//
// For every output row:
//
//     out[row, column] = bias[column]
//
// This is independent of the vendor BLAS implementation and
// can therefore be shared across CUDA-compatible backends.
// ============================================================

template <typename T>
__global__ void linear_broadcast_bias_kernel(
    T *__restrict__ out,
    const T *__restrict__ bias,
    std::size_t output_elements,
    std::size_t output_features) {
    const std::size_t first
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t stride
        = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);

    for (std::size_t index = first; index < output_elements; index += stride) {
        out[index] = bias[index % output_features];
    }
}

// ============================================================
// Shared CUDA-compatible launcher
// ============================================================
//
// The vendor backend supplies:
//
//     block_size
//     grid_size
//     stream
//
// because those are backend-specific execution choices.
// ============================================================

template <typename T, typename StreamT>
inline void launch_linear_bias_broadcast(
    T *out,
    const T *bias,
    std::size_t output_elements,
    std::size_t output_features,
    unsigned int block_size,
    std::size_t grid_size,
    StreamT stream) {
    if (output_elements == 0) { return; }

    linear_broadcast_bias_kernel<T>
        <<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
            out, bias, output_elements, output_features);
}

} // namespace llaisys::ops::cuda_compat
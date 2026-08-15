#pragma once

#include "../../cuda/common.cuh"
#include "../../../utils.hpp"

#include "linear_cuda.hpp"
#include "../linear_config.hpp"

#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace llaisys::ops::cuda {
namespace {

struct LinearProblem {
    std::size_t output_elements;
    std::size_t input_elements;
    std::size_t weight_elements;
};

inline LinearProblem validate_linear_arguments(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    std::size_t row_count,
    std::size_t output_features,
    std::size_t input_features) {
    const std::size_t output_elements = llaisys::utils::checked_product(
        row_count, output_features, "Linear: output element count overflows size_t.");

    const std::size_t input_elements = llaisys::utils::checked_product(
        row_count, input_features, "Linear: input element count overflows size_t.");

    const std::size_t weight_elements = llaisys::utils::checked_product(
        output_features, input_features, "Linear: weight element count overflows size_t.");

    CHECK_ARGUMENT(
        output_elements == 0 || out != nullptr, "Linear: output pointer must not be null.");

    CHECK_ARGUMENT(input_elements == 0 || in != nullptr, "Linear: input pointer must not be null.");

    CHECK_ARGUMENT(
        weight_elements == 0 || weight != nullptr, "Linear: weight pointer must not be null.");

    return LinearProblem{
        output_elements,
        input_elements,
        weight_elements,
    };
}

// ============================================================
// Shared tiled Linear kernel
// ============================================================
//
// LLAISYS tensor layout:
//
//     in:     [M, K]
//     weight: [N, K]
//     out:    [M, N]
//
// Computes:
//
//     out[m, n] = sum_k in[m, k] * weight[n, k] + bias[n]
//
// Cross-GPU controlled baseline:
//   - identical source on NVIDIA and MetaX
//   - identical TILE_SIZE
//   - identical block geometry TILE_SIZE x TILE_SIZE
//   - identical shared-memory tiling
//   - identical reduction order
//   - FP32 accumulation for F32/F16/BF16
//   - bias fused into the same kernel
//
// The weight tile is stored transposed in shared memory so the
// per-output-column accesses in the inner product are contiguous.
// ============================================================

template <typename T, unsigned int TILE_SIZE>
__global__ void linear_tiled_kernel(
    T *__restrict__ out,
    const T *__restrict__ in,
    const T *__restrict__ weight,
    const T *__restrict__ bias,
    std::size_t row_count,
    std::size_t output_features,
    std::size_t input_features) {
    static_assert(TILE_SIZE == 8 || TILE_SIZE == 16, "Unsupported shared Linear tile size.");

    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];

    const unsigned int local_column = threadIdx.x;
    const unsigned int local_row = threadIdx.y;

    const std::size_t row
        = static_cast<std::size_t>(blockIdx.y) * TILE_SIZE + static_cast<std::size_t>(local_row);

    const std::size_t column
        = static_cast<std::size_t>(blockIdx.x) * TILE_SIZE + static_cast<std::size_t>(local_column);

    float accumulator = 0.0F;

    for (std::size_t k_base = 0; k_base < input_features; k_base += TILE_SIZE) {
        const std::size_t input_k = k_base + static_cast<std::size_t>(local_column);

        if (row < row_count && input_k < input_features) {
            input_tile[local_row][local_column] = to_float<T>(in[row * input_features + input_k]);
        } else {
            input_tile[local_row][local_column] = 0.0F;
        }

        const std::size_t weight_column = static_cast<std::size_t>(blockIdx.x) * TILE_SIZE
                                        + static_cast<std::size_t>(local_row);

        const std::size_t weight_k = k_base + static_cast<std::size_t>(local_column);

        if (weight_column < output_features && weight_k < input_features) {
            // Store [K-tile, N-tile] instead of [N-tile, K-tile].
            weight_tile[local_column][local_row]
                = to_float<T>(weight[weight_column * input_features + weight_k]);
        } else {
            weight_tile[local_column][local_row] = 0.0F;
        }

        __syncthreads();

#pragma unroll
        for (unsigned int k = 0; k < TILE_SIZE; ++k) {
            accumulator += input_tile[local_row][k] * weight_tile[k][local_column];
        }

        __syncthreads();
    }

    if (row >= row_count || column >= output_features) { return; }

    if (bias != nullptr) { accumulator += to_float<T>(bias[column]); }

    out[row * output_features + column] = from_float<T>(accumulator);
}

template <typename T, unsigned int TILE_SIZE>
void launch_linear_tiled(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    std::size_t row_count,
    std::size_t output_features,
    std::size_t input_features,
    cudaStream_t stream) {
    constexpr unsigned int block_x = TILE_SIZE;
    constexpr unsigned int block_y = TILE_SIZE;

    const std::size_t grid_x_size = div_ceil(output_features, TILE_SIZE);
    const std::size_t grid_y_size = div_ceil(row_count, TILE_SIZE);

    CHECK_ARGUMENT(
        grid_x_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "Linear: grid x dimension exceeds the supported launch range.");

    CHECK_ARGUMENT(
        grid_y_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "Linear: grid y dimension exceeds the supported launch range.");

    const dim3 block_dimension(block_x, block_y);
    const dim3 grid_dimension(
        static_cast<unsigned int>(grid_x_size), static_cast<unsigned int>(grid_y_size));

    if (config::debug_enabled()) {
        std::fprintf(
            stderr,
            "[Linear][%s] implementation=shared_tiled "
            "M=%zu N=%zu K=%zu tile=%u block=(%u,%u) grid=(%u,%u) "
            "bias=%s accumulation=f32\n",
            GPU_BACKEND_NAME, row_count, output_features, input_features, TILE_SIZE, block_x,
            block_y, grid_dimension.x, grid_dimension.y, bias == nullptr ? "no" : "yes");
    }

    linear_tiled_kernel<T, TILE_SIZE><<<grid_dimension, block_dimension, 0, stream>>>(
        out, in, weight, bias, row_count, output_features, input_features);

    check_kernel("Linear shared tiled kernel");
}

template <typename T>
void launch_linear(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    std::size_t row_count,
    std::size_t output_features,
    std::size_t input_features,
    cudaStream_t stream) {
    const unsigned int tile_size = linear_config::get_tile_size();

    switch (tile_size) {
    case 8:
        return launch_linear_tiled<T, 8>(
            out, in, weight, bias, row_count, output_features, input_features, stream);

    case 16:
        return launch_linear_tiled<T, 16>(
            out, in, weight, bias, row_count, output_features, input_features, stream);

    default:
        throw std::invalid_argument("Linear: unsupported shared tile size.");
    }
}

} // namespace

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    std::size_t row_count,
    std::size_t output_features,
    std::size_t input_features,
    llaisysStream_t stream) {
    const LinearProblem problem
        = validate_linear_arguments(out, in, weight, row_count, output_features, input_features);

    if (problem.output_elements == 0) { return; }

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    return dispatch_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_linear<T>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
            reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(bias), row_count,
            output_features, input_features, cuda_stream);
    });
}

} // namespace llaisys::ops::cuda

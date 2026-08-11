#pragma once

#include "../cuda_compat/rms_norm_cuda_compat.cuh"

#include <cub/block/block_reduce.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace llaisys::ops::nvidia::detail {

// ============================================================
// NVIDIA CUB scalar RMSNorm
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
__global__ void rms_norm_cub_scalar_kernel(
    T *out, const T *in, const T *weight, float eps, std::size_t nrow, std::size_t ncol) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;

    __shared__ typename BlockReduce::TempStorage reduction_storage;

    __shared__ float inverse_rms;

    for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < nrow;
         row += static_cast<std::size_t>(gridDim.x)) {
        const T *row_in = in + row * ncol;

        T *row_out = out + row * ncol;

        float thread_square_sum = 0.0F;

        for (std::size_t col = static_cast<std::size_t>(threadIdx.x); col < ncol;
             col += BLOCK_SIZE) {
            const float value = cuda_compat::to_float<T>(row_in[col]);

            thread_square_sum = fmaf(value, value, thread_square_sum);
        }

        const float row_square_sum = BlockReduce(reduction_storage).Sum(thread_square_sum);

        if (threadIdx.x == 0) {
            const float mean_square = row_square_sum / static_cast<float>(ncol);

            inverse_rms = 1.0F / sqrtf(mean_square + eps);
        }

        __syncthreads();

        for (std::size_t col = static_cast<std::size_t>(threadIdx.x); col < ncol;
             col += BLOCK_SIZE) {
            const float input_value = cuda_compat::to_float<T>(row_in[col]);

            const float weight_value = cuda_compat::to_float<T>(weight[col]);

            row_out[col] = cuda_compat::from_float<T>(input_value * weight_value * inverse_rms);
        }

        __syncthreads();
    }
}

// ============================================================
// NVIDIA CUB packed RMSNorm
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
__global__ void rms_norm_cub_packed_kernel(
    T *out, const T *in, const T *weight, float eps, std::size_t nrow, std::size_t ncol) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;

    constexpr std::size_t elements_per_pack = cuda_compat::PACKED_128_ELEMENTS<T>;

    __shared__ typename BlockReduce::TempStorage reduction_storage;

    __shared__ float inverse_rms;

    const std::size_t pack_count = ncol / elements_per_pack;

    const cuda_compat::Packed128 *packed_weight
        = reinterpret_cast<const cuda_compat::Packed128 *>(weight);

    for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < nrow;
         row += static_cast<std::size_t>(gridDim.x)) {
        const T *row_in = in + row * ncol;

        T *row_out = out + row * ncol;

        const cuda_compat::Packed128 *packed_in
            = reinterpret_cast<const cuda_compat::Packed128 *>(row_in);

        cuda_compat::Packed128 *packed_out = reinterpret_cast<cuda_compat::Packed128 *>(row_out);

        float thread_square_sum = 0.0F;

        for (std::size_t pack_index = static_cast<std::size_t>(threadIdx.x);
             pack_index < pack_count;
             pack_index += BLOCK_SIZE) {
            const cuda_compat::Packed128 input_pack = packed_in[pack_index];

            const T *input_values = reinterpret_cast<const T *>(&input_pack);

#pragma unroll
            for (std::size_t item = 0; item < elements_per_pack; ++item) {
                const float value = cuda_compat::to_float<T>(input_values[item]);

                thread_square_sum = fmaf(value, value, thread_square_sum);
            }
        }

        const float row_square_sum = BlockReduce(reduction_storage).Sum(thread_square_sum);

        if (threadIdx.x == 0) {
            const float mean_square = row_square_sum / static_cast<float>(ncol);

            inverse_rms = 1.0F / sqrtf(mean_square + eps);
        }

        __syncthreads();

        for (std::size_t pack_index = static_cast<std::size_t>(threadIdx.x);
             pack_index < pack_count;
             pack_index += BLOCK_SIZE) {
            const cuda_compat::Packed128 input_pack = packed_in[pack_index];

            const cuda_compat::Packed128 weight_pack = packed_weight[pack_index];

            cuda_compat::Packed128 output_pack{};

            const T *input_values = reinterpret_cast<const T *>(&input_pack);

            const T *weight_values = reinterpret_cast<const T *>(&weight_pack);

            T *output_values = reinterpret_cast<T *>(&output_pack);

#pragma unroll
            for (std::size_t item = 0; item < elements_per_pack; ++item) {
                const float input_value = cuda_compat::to_float<T>(input_values[item]);

                const float weight_value = cuda_compat::to_float<T>(weight_values[item]);

                output_values[item]
                    = cuda_compat::from_float<T>(input_value * weight_value * inverse_rms);
            }

            packed_out[pack_index] = output_pack;
        }

        __syncthreads();
    }
}

// ============================================================
// NVIDIA CUB launcher
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
inline void launch_rms_norm_cub_kernel(
    T *out,
    const T *in,
    const T *weight,
    float eps,
    std::size_t nrow,
    std::size_t ncol,
    std::size_t grid_size,
    bool use_packed_kernel,
    cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned int>(grid_size));

    const dim3 block(BLOCK_SIZE);

    if (use_packed_kernel) {
        rms_norm_cub_packed_kernel<T, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(out, in, weight, eps, nrow, ncol);
    } else {
        rms_norm_cub_scalar_kernel<T, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(out, in, weight, eps, nrow, ncol);
    }
}

} // namespace llaisys::ops::nvidia::detail

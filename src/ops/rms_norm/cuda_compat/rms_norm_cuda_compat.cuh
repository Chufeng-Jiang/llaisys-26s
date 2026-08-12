#pragma once

#include "../../cuda_compat/common.cuh"

#include <cmath>
#include <cstddef>

namespace llaisys::ops::cuda_compat {

// ============================================================
// Portable block reduction
// ============================================================
//
// Deliberately avoids CUB and vendor-specific warp primitives.
//
// Requirements:
//   - BLOCK_SIZE must be a power of two.
//
// This implementation is intended as the CUDA-compatible
// portability baseline.
//
// Individual vendors may provide an optimized reduction path.
// ============================================================

template <unsigned int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float value, float *shared_values) {
    static_assert(
        BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "RMSNorm: BLOCK_SIZE must be a power of two.");

    const unsigned int thread_index = threadIdx.x;

    shared_values[thread_index] = value;

    __syncthreads();

#pragma unroll
    for (unsigned int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (thread_index < offset) {
            shared_values[thread_index] += shared_values[thread_index + offset];
        }

        __syncthreads();
    }

    return shared_values[0];
}

// ============================================================
// Scalar RMSNorm kernel
// ============================================================
//
// One block processes one or more rows.
//
// Accumulation is FP32 for:
//   - FP32
//   - FP16
//   - BF16
//
// Normalization and weight multiplication are fused.
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
__global__ void rms_norm_scalar_kernel(
    T *out, const T *in, const T *weight, float eps, std::size_t nrow, std::size_t ncol) {
    __shared__ float reduction_storage[BLOCK_SIZE];

    __shared__ float inverse_rms;

    for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < nrow;
         row += static_cast<std::size_t>(gridDim.x)) {
        const T *row_in = in + row * ncol;

        T *row_out = out + row * ncol;

        float thread_square_sum = 0.0F;

        for (std::size_t col = static_cast<std::size_t>(threadIdx.x); col < ncol;
             col += BLOCK_SIZE) {
            const float value = to_float<T>(row_in[col]);

            thread_square_sum = fmaf(value, value, thread_square_sum);
        }

        const float row_square_sum
            = block_reduce_sum<BLOCK_SIZE>(thread_square_sum, reduction_storage);

        if (threadIdx.x == 0) {
            const float mean_square = row_square_sum / static_cast<float>(ncol);

            // Preserve the existing numerical behavior:
            // sqrtf + division instead of rsqrtf.
            inverse_rms = 1.0F / sqrtf(mean_square + eps);
        }

        __syncthreads();

        for (std::size_t col = static_cast<std::size_t>(threadIdx.x); col < ncol;
             col += BLOCK_SIZE) {
            const float input_value = to_float<T>(row_in[col]);

            const float weight_value = to_float<T>(weight[col]);

            row_out[col] = from_float<T>(input_value * weight_value * inverse_rms);
        }

        // The shared reduction buffer and inverse RMS
        // are reused by the next row handled by this block.
        __syncthreads();
    }
}

// ============================================================
// Packed-path eligibility
// ============================================================

template <typename T>
inline bool can_use_packed_rms_norm(const T *out, const T *in, const T *weight, std::size_t ncol) {
    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    // Requiring an integral number of packs per row guarantees
    // that every subsequent row also starts on a 16-byte
    // boundary.
    return ncol % elements_per_pack == 0 && are_aligned<PACKED_128_ALIGNMENT>(out, in, weight);
}

// ============================================================
// Packed RMSNorm kernel
// ============================================================
//
// Packed128 contains:
//
//   FP32 -> 4 elements
//   FP16 -> 8 elements
//   BF16 -> 8 elements
//
// Packing optimizes memory traffic while keeping the arithmetic
// and accumulation in FP32.
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
__global__ void rms_norm_packed_kernel(
    T *out, const T *in, const T *weight, float eps, std::size_t nrow, std::size_t ncol) {
    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    __shared__ float reduction_storage[BLOCK_SIZE];

    __shared__ float inverse_rms;

    const std::size_t pack_count = ncol / elements_per_pack;

    const Packed128 *packed_weight = reinterpret_cast<const Packed128 *>(weight);

    for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < nrow;
         row += static_cast<std::size_t>(gridDim.x)) {
        const T *row_in = in + row * ncol;

        T *row_out = out + row * ncol;

        const Packed128 *packed_in = reinterpret_cast<const Packed128 *>(row_in);

        Packed128 *packed_out = reinterpret_cast<Packed128 *>(row_out);

        float thread_square_sum = 0.0F;

        for (std::size_t pack_index = static_cast<std::size_t>(threadIdx.x);
             pack_index < pack_count; pack_index += BLOCK_SIZE) {
            const Packed128 input_pack = packed_in[pack_index];

            const T *input_values = reinterpret_cast<const T *>(&input_pack);

#pragma unroll
            for (std::size_t item = 0; item < elements_per_pack; ++item) {
                const float value = to_float<T>(input_values[item]);

                thread_square_sum = fmaf(value, value, thread_square_sum);
            }
        }

        const float row_square_sum
            = block_reduce_sum<BLOCK_SIZE>(thread_square_sum, reduction_storage);

        if (threadIdx.x == 0) {
            const float mean_square = row_square_sum / static_cast<float>(ncol);

            inverse_rms = 1.0F / sqrtf(mean_square + eps);
        }

        __syncthreads();

        for (std::size_t pack_index = static_cast<std::size_t>(threadIdx.x);
             pack_index < pack_count; pack_index += BLOCK_SIZE) {
            // Load complete input and weight packs before
            // writing the output pack so out == in remains safe.

            const Packed128 input_pack = packed_in[pack_index];

            const Packed128 weight_pack = packed_weight[pack_index];

            Packed128 output_pack{};

            const T *input_values = reinterpret_cast<const T *>(&input_pack);

            const T *weight_values = reinterpret_cast<const T *>(&weight_pack);

            T *output_values = reinterpret_cast<T *>(&output_pack);

#pragma unroll
            for (std::size_t item = 0; item < elements_per_pack; ++item) {
                const float input_value = to_float<T>(input_values[item]);

                const float weight_value = to_float<T>(weight_values[item]);

                output_values[item] = from_float<T>(input_value * weight_value * inverse_rms);
            }

            packed_out[pack_index] = output_pack;
        }

        __syncthreads();
    }
}

// ============================================================
// Shared CUDA-compatible launcher
// ============================================================
//
// The algorithm owns:
//
//   - scalar implementation
//   - packed implementation
//   - FP32 reduction
//   - normalization math
//
// The vendor adapter owns:
//
//   - BLOCK_SIZE selection
//   - grid-size selection
//   - stream conversion
//   - runtime error handling
// ============================================================

template <typename T, unsigned int BLOCK_SIZE, typename StreamT>
inline void launch_rms_norm_kernel(
    T *out,
    const T *in,
    const T *weight,
    float eps,
    std::size_t nrow,
    std::size_t ncol,
    std::size_t grid_size,
    bool use_packed_kernel,
    StreamT stream) {
    if (nrow == 0) { return; }

    const dim3 grid(static_cast<unsigned int>(grid_size));

    const dim3 block(BLOCK_SIZE);

    if (use_packed_kernel) {
        rms_norm_packed_kernel<T, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(out, in, weight, eps, nrow, ncol);
    } else {
        rms_norm_scalar_kernel<T, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(out, in, weight, eps, nrow, ncol);
    }
}

} // namespace llaisys::ops::cuda_compat

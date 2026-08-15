#pragma once

#include "../../cuda/common.cuh"

#include "rms_norm_cuda.hpp"
#include "../../../utils.hpp"
#include "../rms_norm_config.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>

namespace llaisys::ops::cuda {
namespace {

// ============================================================
// Shared tree reduction
// ============================================================
//
// This reduction deliberately avoids:
//
//     CUB
//     warp shuffle intrinsics
//     backend-specific subgroup width
//
// The exact same source and reduction tree are compiled by
// NVCC and MXCC for the controlled cross-GPU baseline.
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
// Shared scalar RMSNorm
// ============================================================
//
// One block processes one or more rows through a row-level
// grid-stride loop.
//
// Numerical policy is intentionally shared:
//
//     FP32 accumulation
//     fmaf(value, value, sum)
//     mean = sum / column_count
//     inverse_rms = 1 / sqrtf(mean + eps)
//     output = input * weight * inverse_rms
//
// The sqrtf + division form preserves the behavior of the
// previous shared CUDA-compatible implementation.
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
__global__ void rms_norm_scalar_kernel(
    T *__restrict__ out,
    const T *__restrict__ in,
    const T *__restrict__ weight,
    float eps,
    std::size_t row_count,
    std::size_t column_count) {
    __shared__ float reduction_storage[BLOCK_SIZE];
    __shared__ float inverse_rms;

    for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < row_count;
         row += static_cast<std::size_t>(gridDim.x)) {
        const T *const row_in = in + row * column_count;
        T *const row_out = out + row * column_count;

        float thread_square_sum = 0.0F;

        for (std::size_t column = static_cast<std::size_t>(threadIdx.x); column < column_count;
             column += BLOCK_SIZE) {
            const float value = to_float<T>(row_in[column]);
            thread_square_sum = fmaf(value, value, thread_square_sum);
        }

        const float row_square_sum
            = block_reduce_sum<BLOCK_SIZE>(thread_square_sum, reduction_storage);

        if (threadIdx.x == 0) {
            const float mean_square = row_square_sum / static_cast<float>(column_count);

            inverse_rms = 1.0F / sqrtf(mean_square + eps);
        }

        __syncthreads();

        for (std::size_t column = static_cast<std::size_t>(threadIdx.x); column < column_count;
             column += BLOCK_SIZE) {
            const float input_value = to_float<T>(row_in[column]);
            const float weight_value = to_float<T>(weight[column]);

            row_out[column] = from_float<T>(input_value * weight_value * inverse_rms);
        }

        // Shared storage is reused by the next row assigned to this block.
        __syncthreads();
    }
}

// ============================================================
// Shared Packed128 eligibility
// ============================================================
//
// Both platforms use exactly the same condition. Requiring a
// complete number of packs per row also guarantees that every
// subsequent row begins at a 16-byte boundary.
// ============================================================

template <typename T>
inline bool
can_use_packed_rms_norm(const T *out, const T *in, const T *weight, std::size_t column_count) {
    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    return column_count % elements_per_pack == 0
        && are_aligned<PACKED_128_ALIGNMENT>(out, in, weight);
}

// ============================================================
// Shared Packed128 RMSNorm
// ============================================================
//
// Packed128 contains:
//
//     FP32 -> 4 elements
//     FP16 -> 8 elements
//     BF16 -> 8 elements
//
// Packing changes memory traversal only. Arithmetic and
// accumulation remain FP32 and match the scalar path.
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
__global__ void rms_norm_packed_kernel(
    T *__restrict__ out,
    const T *__restrict__ in,
    const T *__restrict__ weight,
    float eps,
    std::size_t row_count,
    std::size_t column_count) {
    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    __shared__ float reduction_storage[BLOCK_SIZE];
    __shared__ float inverse_rms;

    const std::size_t pack_count = column_count / elements_per_pack;
    const Packed128 *const packed_weight = reinterpret_cast<const Packed128 *>(weight);

    for (std::size_t row = static_cast<std::size_t>(blockIdx.x); row < row_count;
         row += static_cast<std::size_t>(gridDim.x)) {
        const T *const row_in = in + row * column_count;
        T *const row_out = out + row * column_count;

        const Packed128 *const packed_in = reinterpret_cast<const Packed128 *>(row_in);

        Packed128 *const packed_out = reinterpret_cast<Packed128 *>(row_out);

        float thread_square_sum = 0.0F;

        for (std::size_t pack_index = static_cast<std::size_t>(threadIdx.x);
             pack_index < pack_count; pack_index += BLOCK_SIZE) {
            const Packed128 input_pack = packed_in[pack_index];
            const T *const input_values = reinterpret_cast<const T *>(&input_pack);

#pragma unroll
            for (std::size_t item = 0; item < elements_per_pack; ++item) {
                const float value = to_float<T>(input_values[item]);
                thread_square_sum = fmaf(value, value, thread_square_sum);
            }
        }

        const float row_square_sum
            = block_reduce_sum<BLOCK_SIZE>(thread_square_sum, reduction_storage);

        if (threadIdx.x == 0) {
            const float mean_square = row_square_sum / static_cast<float>(column_count);

            inverse_rms = 1.0F / sqrtf(mean_square + eps);
        }

        __syncthreads();

        for (std::size_t pack_index = static_cast<std::size_t>(threadIdx.x);
             pack_index < pack_count; pack_index += BLOCK_SIZE) {
            // Read complete packs before writing so out == in remains safe.
            const Packed128 input_pack = packed_in[pack_index];
            const Packed128 weight_pack = packed_weight[pack_index];

            Packed128 output_pack{};

            const T *const input_values = reinterpret_cast<const T *>(&input_pack);

            const T *const weight_values = reinterpret_cast<const T *>(&weight_pack);

            T *const output_values = reinterpret_cast<T *>(&output_pack);

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
// Shared launch
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
void launch_rms_norm_fixed_block(
    T *out,
    const T *in,
    const T *weight,
    float eps,
    std::size_t row_count,
    std::size_t column_count,
    std::size_t grid_size,
    bool use_packed_kernel,
    cudaStream_t stream) {
    const dim3 grid_dimension(static_cast<unsigned int>(grid_size));
    const dim3 block_dimension(BLOCK_SIZE);

    if (use_packed_kernel) {
        rms_norm_packed_kernel<T, BLOCK_SIZE><<<grid_dimension, block_dimension, 0, stream>>>(
            out, in, weight, eps, row_count, column_count);
    } else {
        rms_norm_scalar_kernel<T, BLOCK_SIZE><<<grid_dimension, block_dimension, 0, stream>>>(
            out, in, weight, eps, row_count, column_count);
    }

    check_kernel("RMSNorm kernel");
}

template <typename T>
void launch_rms_norm(
    T *out,
    const T *in,
    const T *weight,
    float eps,
    std::size_t row_count,
    std::size_t column_count,
    cudaStream_t stream) {
    CHECK_ARGUMENT(row_count == 0 || out != nullptr, "RMSNorm: output pointer must not be null.");

    CHECK_ARGUMENT(row_count == 0 || in != nullptr, "RMSNorm: input pointer must not be null.");

    CHECK_ARGUMENT(
        row_count == 0 || weight != nullptr, "RMSNorm: weight pointer must not be null.");

    if (row_count == 0) { return; }

    CHECK_ARGUMENT(column_count > 0, "RMSNorm: row width must be greater than zero.");

    CHECK_ARGUMENT(
        std::isfinite(eps) && eps >= 0.0F, "RMSNorm: epsilon must be finite and nonnegative.");

    CHECK_ARGUMENT(
        column_count <= std::numeric_limits<std::size_t>::max() / row_count,
        "RMSNorm: tensor element count overflows size_t.");

    const unsigned int block_size = rms_norm_config::block_size();

    const std::size_t grid_size = cap_grid_size(row_count, rms_norm_config::MAX_BLOCKS);

    CHECK_ARGUMENT(grid_size > 0, "RMSNorm: grid size must be greater than zero.");

    CHECK_ARGUMENT(
        grid_size <= static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()),
        "RMSNorm: grid size exceeds the supported launch range.");

    const bool use_packed_kernel = can_use_packed_rms_norm<T>(out, in, weight, column_count);

    if (config::debug_enabled()) {
        std::fprintf(
            stderr,
            "[RMSNorm][%s] implementation=shared_tree "
            "rows=%zu columns=%zu block=%u grid=%zu kernel=%s "
            "accumulation=f32\n",
            GPU_BACKEND_NAME, row_count, column_count, block_size, grid_size,
            use_packed_kernel ? "packed128" : "scalar");
    }

    switch (block_size) {
    case 64:
        return launch_rms_norm_fixed_block<T, 64>(
            out, in, weight, eps, row_count, column_count, grid_size, use_packed_kernel, stream);

    case 128:
        return launch_rms_norm_fixed_block<T, 128>(
            out, in, weight, eps, row_count, column_count, grid_size, use_packed_kernel, stream);

    case 256:
        return launch_rms_norm_fixed_block<T, 256>(
            out, in, weight, eps, row_count, column_count, grid_size, use_packed_kernel, stream);

    default:
        throw std::invalid_argument("RMSNorm: unsupported shared block size.");
    }
}

} // namespace

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    float eps,
    llaisysDataType_t type,
    std::size_t row_count,
    std::size_t column_count,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    return dispatch_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_rms_norm<T>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
            reinterpret_cast<const T *>(weight), eps, row_count, column_count, cuda_stream);
    });
}

} // namespace llaisys::ops::cuda

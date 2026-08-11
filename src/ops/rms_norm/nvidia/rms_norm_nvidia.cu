#include "rms_norm_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include "../cuda_compat/rms_norm_cuda_compat.cuh"
#include "rms_norm_nvidia_cub.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>

// ============================================================
// RMSNorm implementation selection
// ============================================================
//
// Default:
//     0 -> NVIDIA CUB optimized override
//
// Portability experiment:
//     1 -> shared CUDA-compatible baseline
//
// Later this can become an XMake option for benchmarking.
// ============================================================

#ifndef LLAISYS_NVIDIA_RMS_NORM_USE_CUDA_COMPAT
#define LLAISYS_NVIDIA_RMS_NORM_USE_CUDA_COMPAT 0
#endif

namespace {

namespace cuda_compat = llaisys::ops::cuda_compat;

namespace nvidia_detail = llaisys::ops::nvidia::detail;

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::CUDA_WARP_SIZE;
using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::device::nvidia::to_cuda_stream;

// ============================================================
// NVIDIA-specific RMSNorm tuning
// ============================================================
//
// These remain in the NVIDIA adapter because block-size choices
// are performance tuning decisions rather than algorithmic
// requirements.
// ============================================================

inline constexpr unsigned int SMALL_BLOCK_SIZE = static_cast<unsigned int>(CUDA_WARP_SIZE * 2);

inline constexpr unsigned int MEDIUM_BLOCK_SIZE = static_cast<unsigned int>(CUDA_WARP_SIZE * 4);

inline constexpr unsigned int LARGE_BLOCK_SIZE = static_cast<unsigned int>(CUDA_BLOCK_SIZE);

inline constexpr std::size_t MEDIUM_BLOCK_MAX_COLUMNS = 512;

static_assert(
    SMALL_BLOCK_SIZE <= MEDIUM_BLOCK_SIZE && MEDIUM_BLOCK_SIZE <= LARGE_BLOCK_SIZE,
    "RMSNorm: invalid block-size ordering.");

// ============================================================
// Backend launch
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
void launch_nvidia_rms_norm_kernel(
    T *out,
    const T *in,
    const T *weight,
    float eps,
    std::size_t nrow,
    std::size_t ncol,
    cudaStream_t stream) {
    // Each block initially owns one row.
    //
    // Both shared and CUB kernels use a row-level
    // grid-stride loop, so the grid may be capped.
    const std::size_t grid_size = get_capped_grid_size(nrow, 1);

    const bool use_packed_kernel = cuda_compat::can_use_packed_rms_norm<T>(out, in, weight, ncol);

#if LLAISYS_NVIDIA_RMS_NORM_USE_CUDA_COMPAT

    // ========================================================
    // Shared CUDA-compatible baseline
    // ========================================================

    cuda_compat::launch_rms_norm_kernel<T, BLOCK_SIZE>(
        out, in, weight, eps, nrow, ncol, grid_size, use_packed_kernel, stream);

#else

    // ========================================================
    // NVIDIA CUB optimized override
    // ========================================================

    nvidia_detail::launch_rms_norm_cub_kernel<T, BLOCK_SIZE>(
        out, in, weight, eps, nrow, ncol, grid_size, use_packed_kernel, stream);

#endif

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// NVIDIA launch tuning
// ============================================================

template <typename T>
void launch_nvidia_rms_norm(
    T *out,
    const T *in,
    const T *weight,
    float eps,
    std::size_t nrow,
    std::size_t ncol,
    cudaStream_t stream) {
    CHECK_ARGUMENT(nrow == 0 || out != nullptr, "RMSNorm: output pointer must not be null.");

    CHECK_ARGUMENT(nrow == 0 || in != nullptr, "RMSNorm: input pointer must not be null.");

    CHECK_ARGUMENT(nrow == 0 || weight != nullptr, "RMSNorm: weight pointer must not be null.");

    if (nrow == 0) { return; }

    CHECK_ARGUMENT(ncol > 0, "RMSNorm: row width must be greater than zero.");

    CHECK_ARGUMENT(
        std::isfinite(eps) && eps >= 0.0F, "RMSNorm: epsilon must be finite and nonnegative.");

    CHECK_ARGUMENT(
        ncol <= std::numeric_limits<std::size_t>::max() / nrow,
        "RMSNorm: tensor element count overflows size_t.");

    // NVIDIA scheduling policy:
    //
    // narrow rows:
    //     64 threads
    //
    // medium rows:
    //     128 threads
    //
    // wide rows:
    //     256 threads

    if (ncol <= SMALL_BLOCK_SIZE) {
        return launch_nvidia_rms_norm_kernel<T, SMALL_BLOCK_SIZE>(
            out, in, weight, eps, nrow, ncol, stream);
    }

    if (ncol <= MEDIUM_BLOCK_MAX_COLUMNS) {
        return launch_nvidia_rms_norm_kernel<T, MEDIUM_BLOCK_SIZE>(
            out, in, weight, eps, nrow, ncol, stream);
    }

    return launch_nvidia_rms_norm_kernel<T, LARGE_BLOCK_SIZE>(
        out, in, weight, eps, nrow, ncol, stream);
}

} // namespace

// ============================================================
// Public NVIDIA backend interface
// ============================================================

namespace llaisys::ops::nvidia {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    float eps,
    llaisysDataType_t type,
    std::size_t nrow,
    std::size_t ncol,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = to_cuda_stream(stream);

    return llaisys::device::nvidia::dispatch_cuda_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_nvidia_rms_norm<T>(
            reinterpret_cast<T *>(out),
            reinterpret_cast<const T *>(in),
            reinterpret_cast<const T *>(weight),
            eps,
            nrow,
            ncol,
            cuda_stream);
    });
}

} // namespace llaisys::ops::nvidia
#include "argmax_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../../utils.hpp"

#include "../cuda_compat/argmax_cuda_compat.cuh"
#include "argmax_nvidia_optimized.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

// ============================================================
// NVIDIA Argmax implementation selection
// ============================================================
//
// 0:
//     NVIDIA optimized implementation.
//
// 1:
//     shared CUDA-compatible portability baseline.
//
// Keep 0 as the production default.
// ============================================================

#ifndef LLAISYS_NVIDIA_ARGMAX_USE_CUDA_COMPAT
#define LLAISYS_NVIDIA_ARGMAX_USE_CUDA_COMPAT 0
#endif

namespace {

namespace cuda_compat = llaisys::ops::cuda_compat;

namespace nvidia_detail = llaisys::ops::nvidia::detail;

using llaisys::device::nvidia::get_warp_aligned_block_size;

using llaisys::device::nvidia::to_cuda_stream;

// ============================================================
// NVIDIA Argmax adapter
// ============================================================

template <typename T>
void launch_nvidia_argmax(
    std::int64_t *max_idx,
    T *max_val,
    const T *vals,
    std::size_t numel,
    llaisys::device::DeviceResource *resource,
    cudaStream_t stream) {
#if LLAISYS_NVIDIA_ARGMAX_USE_CUDA_COMPAT

    // ========================================================
    // CUDA-compatible portability baseline
    // ========================================================

    const unsigned int block_size = get_warp_aligned_block_size(numel);

    cuda_compat::launch_argmax_portable<T>(max_idx, max_val, vals, numel, block_size, stream);

    CUDA_CHECK(cudaGetLastError());

#else

    // ========================================================
    // NVIDIA optimized path
    // ========================================================

    unsigned long long *packed_workspace = nullptr;

    // Only the multi-block implementation needs the Runtime
    // workspace.
    if (nvidia_detail::argmax_requires_workspace(numel)) {
        CHECK_ARGUMENT(resource != nullptr, "Argmax: NVIDIA Runtime resource must not be null.");

        CHECK_ARGUMENT(
            resource->getDeviceType() == LLAISYS_DEVICE_NVIDIA,
            "Argmax: Runtime resource is not an NVIDIA resource.");

        auto *nvidia_resource = static_cast<llaisys::device::nvidia::Resource *>(resource);

        packed_workspace = nvidia_resource->argmaxPackedWorkspace();

        CHECK_ARGUMENT(
            packed_workspace != nullptr, "Argmax: NVIDIA packed workspace must not be null.");
    }

    nvidia_detail::launch_argmax_optimized<T>(
        max_idx, max_val, vals, numel, packed_workspace, stream);

#endif
}

} // namespace

// ============================================================
// Public NVIDIA backend interface
// ============================================================

namespace llaisys::ops::nvidia {

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    std::size_t numel,
    llaisys::device::DeviceResource *resource,
    llaisysStream_t stream) {
    CHECK_ARGUMENT(max_idx != nullptr, "Argmax: max_idx pointer must not be null.");

    CHECK_ARGUMENT(max_val != nullptr, "Argmax: max_val pointer must not be null.");

    CHECK_ARGUMENT(vals != nullptr, "Argmax: vals pointer must not be null.");

    CHECK_ARGUMENT(numel > 0, "Argmax: input tensor must not be empty.");

    CHECK_ARGUMENT(
        numel <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),
        "Argmax: implementation supports at most UINT32_MAX elements.");

    const cudaStream_t cuda_stream = to_cuda_stream(stream);

    return llaisys::device::nvidia::dispatch_cuda_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_nvidia_argmax<T>(
            reinterpret_cast<std::int64_t *>(max_idx),
            reinterpret_cast<T *>(max_val),
            reinterpret_cast<const T *>(vals),
            numel,
            resource,
            cuda_stream);
    });
}

} // namespace llaisys::ops::nvidia
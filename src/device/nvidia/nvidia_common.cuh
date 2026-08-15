#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "../../utils.hpp"

namespace llaisys::device::nvidia {



inline void check_cuda(
    cudaError_t error, const char *expression, const char *file, int line, const char *function) {
    if (error == cudaSuccess) { return; }

    throw std::runtime_error(
        std::string("CUDA error: ") + cudaGetErrorString(error) + " while executing " + expression
        + " in " + function + " at " + file + ":" + std::to_string(line));
}

// ============================================================
// No-throw CUDA device cleanup
// ============================================================
//
// Run cleanup on the specified CUDA device and restore the
// previously active device when it can be determined.
//
// This helper is intended for destructor / cleanup paths only.
// It must never throw.

template <typename Cleanup>
inline void run_on_cuda_device_noexcept(int device_id, Cleanup &&cleanup) noexcept {
    static_assert(
        std::is_nothrow_invocable_v<Cleanup &>, "CUDA cleanup callback must be noexcept.");

    int previous_device = -1;

    const cudaError_t get_device_status = cudaGetDevice(&previous_device);

    const cudaError_t set_device_status = cudaSetDevice(device_id);

    // Only run the cleanup after successfully switching to
    // the device that owns the resource.
    if (set_device_status == cudaSuccess) { cleanup(); }

    // Best-effort restoration. Never propagate CUDA failures
    // from a destructor path.
    if (get_device_status == cudaSuccess && previous_device >= 0 && previous_device != device_id) {
        (void)cudaSetDevice(previous_device);
    }
}

// ============================================================
// CUDA stream handle adapter
// ============================================================
//
// llaisysStream_t is an opaque public handle. In the NVIDIA backend,
// its concrete representation is cudaStream_t.
//
// Keep this conversion centralized so backend code does not depend on
// the representation of the public stream handle.

inline cudaStream_t to_cuda_stream(llaisysStream_t stream) noexcept {
    return reinterpret_cast<cudaStream_t>(stream);
}

inline llaisysStream_t from_cuda_stream(cudaStream_t stream) noexcept {
    return reinterpret_cast<llaisysStream_t>(stream);
}

} // namespace llaisys::device::nvidia

#ifndef CUDA_CHECK
#define CUDA_CHECK(CALL)                                                                           \
    ::llaisys::device::nvidia::check_cuda((CALL), #CALL, __FILE__, __LINE__, __func__)
#endif

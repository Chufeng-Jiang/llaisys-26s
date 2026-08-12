#include "../runtime_api.hpp"
#include "nvidia_common.cuh"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::nvidia {

namespace runtime_api {

namespace {

// Convert the LLAISYS memcpy direction into the corresponding
// CUDA Runtime API memcpy direction.
cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;

    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;

    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;

    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;

    default:
        throw std::invalid_argument("Unknown NVIDIA memory copy kind.");
    }
}

} // namespace

// ============================================================
// Device management
// ============================================================

int getDeviceCount() {
    int count = 0;

    CUDA_CHECK(cudaGetDeviceCount(&count));

    return count;
}

void setDevice(int device_id) { CUDA_CHECK(cudaSetDevice(device_id)); }

void deviceSynchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

// ============================================================
// Stream management
// ============================================================

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreate(&stream));

    return from_cuda_stream(stream);
}

void destroyStream(llaisysStream_t stream) {
    // nullptr represents the default CUDA stream.
    // The default stream was not created by cudaStreamCreate(),
    // so it must not be destroyed here.
    if (stream == nullptr) { return; }

    CUDA_CHECK(cudaStreamDestroy(to_cuda_stream(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    // Passing nullptr synchronizes the default CUDA stream.
    CUDA_CHECK(cudaStreamSynchronize(to_cuda_stream(stream)));
}

// ============================================================
// Device memory
// ============================================================

void *mallocDevice(std::size_t size) {
    if (size == 0) { return nullptr; }

    void *ptr = nullptr;

    CUDA_CHECK(cudaMalloc(&ptr, size));

    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) { return; }

    CUDA_CHECK(cudaFree(ptr));
}

// ============================================================
// Pinned host memory
// ============================================================

void *mallocHost(std::size_t size) {
    if (size == 0) { return nullptr; }

    void *ptr = nullptr;

    CUDA_CHECK(cudaMallocHost(&ptr, size));

    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) { return; }

    CUDA_CHECK(cudaFreeHost(ptr));
}

// ============================================================
// Synchronous memory copy
// ============================================================

void memcpySync(void *dst, const void *src, std::size_t size, llaisysMemcpyKind_t kind) {
    if (size == 0) { return; }

    CHECK_ARGUMENT(dst != nullptr, "NVIDIA memcpySync: dst must not be null.");

    CHECK_ARGUMENT(src != nullptr, "NVIDIA memcpySync: src must not be null.");

    CUDA_CHECK(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

// ============================================================
// Asynchronous memory copy
// ============================================================

void memcpyAsync(
    void *dst,
    const void *src,
    std::size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {
    if (size == 0) { return; }

    CHECK_ARGUMENT(dst != nullptr, "NVIDIA memcpyAsync: dst must not be null.");

    CHECK_ARGUMENT(src != nullptr, "NVIDIA memcpyAsync: src must not be null.");

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), to_cuda_stream(stream)));
}

// ============================================================
// NVIDIA Runtime API table
// ============================================================
static const LlaisysRuntimeAPI RUNTIME_API
    = {&getDeviceCount, &setDevice,         &deviceSynchronize, &createStream,
       &destroyStream,  &streamSynchronize, &mallocDevice,      &freeDevice,
       &mallocHost,     &freeHost,          &memcpySync,        &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() { return &runtime_api::RUNTIME_API; }
} // namespace llaisys::device::nvidia

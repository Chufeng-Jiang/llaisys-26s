#include "runtime_api.hpp"

#include "../utils.hpp"

#include <cstddef>

namespace llaisys::device {

namespace {

int unsupportedGetDeviceCount() { return 0; }

void unsupportedSetDevice(int) { EXCEPTION_UNSUPPORTED_DEVICE; }

void unsupportedDeviceSynchronize() { EXCEPTION_UNSUPPORTED_DEVICE; }

llaisysStream_t unsupportedCreateStream() {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void unsupportedDestroyStream(llaisysStream_t) { EXCEPTION_UNSUPPORTED_DEVICE; }

void unsupportedStreamSynchronize(llaisysStream_t) { EXCEPTION_UNSUPPORTED_DEVICE; }

void *unsupportedMallocDevice(std::size_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void unsupportedFreeDevice(void *) { EXCEPTION_UNSUPPORTED_DEVICE; }

void *unsupportedMallocHost(std::size_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
    return nullptr;
}

void unsupportedFreeHost(void *) { EXCEPTION_UNSUPPORTED_DEVICE; }

void unsupportedMemcpySync(void *, const void *, std::size_t, llaisysMemcpyKind_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

void unsupportedMemcpyAsync(
    void *, const void *, std::size_t, llaisysMemcpyKind_t, llaisysStream_t) {
    EXCEPTION_UNSUPPORTED_DEVICE;
}

const LlaisysRuntimeAPI UNSUPPORTED_RUNTIME_API = {
    &unsupportedGetDeviceCount,
    &unsupportedSetDevice,
    &unsupportedDeviceSynchronize,
    &unsupportedCreateStream,
    &unsupportedDestroyStream,
    &unsupportedStreamSynchronize,
    &unsupportedMallocDevice,
    &unsupportedFreeDevice,
    &unsupportedMallocHost,
    &unsupportedFreeHost,
    &unsupportedMemcpySync,
    &unsupportedMemcpyAsync,
};

} // namespace

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI() { return &UNSUPPORTED_RUNTIME_API; }

const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type) {
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return cpu::getRuntimeAPI();

    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return nvidia::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif

    case LLAISYS_DEVICE_METAX:
#ifdef ENABLE_METAX_API
        return metax::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}

} // namespace llaisys::device
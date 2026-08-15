#pragma once

#include "llaisys.h"

namespace llaisys::device {

inline constexpr bool is_cuda_compatible_gpu(llaisysDeviceType_t device_type) noexcept {
#ifdef ENABLE_NVIDIA_API
    if (device_type == LLAISYS_DEVICE_NVIDIA) { return true; }
#endif

#ifdef ENABLE_METAX_API
    if (device_type == LLAISYS_DEVICE_METAX) { return true; }
#endif

    return false;
}

} // namespace llaisys::device
#include "device_resource.hpp"

#include "../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/nvidia_resource_factory.hpp"
#endif

namespace llaisys::device {

std::unique_ptr<DeviceResource>
createDeviceResource(llaisysDeviceType_t device_type, int device_id) {
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return nullptr;

    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return nvidia::createDeviceResource(device_id);
#else
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}

} // namespace llaisys::device
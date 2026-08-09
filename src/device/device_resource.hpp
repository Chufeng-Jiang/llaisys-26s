#pragma once
#include "llaisys.h"

#include "../utils.hpp"

namespace llaisys::device {
class DeviceResource {
private:
    llaisysDeviceType_t _device_type;
    int _device_id;

public:
    DeviceResource(llaisysDeviceType_t device_type, int device_id)
        : _device_type(device_type), _device_id(device_id) {}
    // ~DeviceResource() = default;
    // The destructor must be virtual because Runtime stores
    // backend resources through a DeviceResource base pointer.
    virtual ~DeviceResource() = default;

    // Prevent copying.
    DeviceResource(const DeviceResource &) = delete;
    DeviceResource &operator=(const DeviceResource &) = delete;

    // Prevent moving.
    DeviceResource(DeviceResource &&) = delete;
    DeviceResource &operator=(DeviceResource &&) = delete;

    llaisysDeviceType_t getDeviceType() const { return _device_type; }
    int getDeviceId() const { return _device_id; };
};
} // namespace llaisys::device

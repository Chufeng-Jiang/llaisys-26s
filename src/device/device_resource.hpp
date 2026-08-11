#pragma once

#include "llaisys.h"

#include <memory>

namespace llaisys::device {

class DeviceResource {
private:
    llaisysDeviceType_t _device_type;
    int _device_id;

public:
    DeviceResource(llaisysDeviceType_t device_type, int device_id)
        : _device_type(device_type), _device_id(device_id) {}

    virtual ~DeviceResource() = default;

    DeviceResource(const DeviceResource &) = delete;

    DeviceResource &operator=(const DeviceResource &) = delete;

    DeviceResource(DeviceResource &&) = delete;

    DeviceResource &operator=(DeviceResource &&) = delete;

    llaisysDeviceType_t getDeviceType() const { return _device_type; }

    int getDeviceId() const { return _device_id; }
};

std::unique_ptr<DeviceResource>
createDeviceResource(llaisysDeviceType_t device_type, int device_id);

} // namespace llaisys::device
#pragma once

#include "../device_resource.hpp"

#include <memory>

namespace llaisys::device::nvidia {

std::unique_ptr<llaisys::device::DeviceResource> createDeviceResource(int device_id);

} // namespace llaisys::device::nvidia
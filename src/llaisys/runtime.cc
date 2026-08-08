#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"
#include "error.hpp"
#include "llaisys/runtime.h"

// Llaisys API for setting context runtime.
__C int llaisysSetContextRuntime(llaisysDeviceType_t device_type, int device_id) {
  return llaisys::c_api::guard([&]() { llaisys::core::context().setDevice(device_type, device_id); }) ? 0 : -1;
}

// Llaisys API for getting the runtime APIs.
__C const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t device_type) {
  return llaisys::c_api::guard_result<const LlaisysRuntimeAPI *>([&]() { return llaisys::device::getRuntimeAPI(device_type); }, nullptr);
}
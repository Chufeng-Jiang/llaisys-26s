#include "llaisys/runtime.h"

#include "../core/context/context.hpp"
#include "../device/runtime_api.hpp"

#include "error.hpp"

#include <stdexcept>


// Llaisys API for setting context runtime.
__C int llaisysSetContextRuntime(
    llaisysDeviceType_t device_type,
    int device_id) {

    return llaisys::c_api::guard(
               [&]() {
                   llaisys::core::context().setDevice(
                       device_type,
                       device_id);
               })
        ? 0
        : -1;
}


// Llaisys API for getting the stream owned by the selected Runtime.
__C int llaisysGetContextStream(
    llaisysDeviceType_t device_type,
    int device_id,
    llaisysStream_t *stream) {

    return llaisys::c_api::guard(
               [&]() {
                   if (stream == nullptr) {
                       throw std::invalid_argument(
                           "llaisysGetContextStream: stream must not be null.");
                   }

                   auto &context = llaisys::core::context();

                   context.setDevice(
                       device_type,
                       device_id);

                   *stream = context.runtime().stream();
               })
        ? 0
        : -1;
}


// Llaisys API for getting the runtime APIs.
__C const LlaisysRuntimeAPI *
llaisysGetRuntimeAPI(
    llaisysDeviceType_t device_type) {

    return llaisys::c_api::guard_result<
        const LlaisysRuntimeAPI *>(
        [&]() {
            return llaisys::device::getRuntimeAPI(
                device_type);
        },
        nullptr);
}
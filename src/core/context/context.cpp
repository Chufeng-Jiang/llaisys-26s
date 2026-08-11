#include "context.hpp"

#include "../../utils.hpp"

namespace llaisys::core {

Runtime *Context::getOrCreateRuntime(llaisysDeviceType_t device_type, int device_id) {
    auto runtime_iterator = _runtime_map.find(device_type);

    CHECK_ARGUMENT(runtime_iterator != _runtime_map.end(), "invalid device type");

    auto &runtimes = runtime_iterator->second;

    CHECK_ARGUMENT(
        device_id >= 0 && static_cast<std::size_t>(device_id) < runtimes.size(),
        "invalid device id");

    auto &runtime = runtimes[static_cast<std::size_t>(device_id)];

    if (!runtime) {
        // Runtime's constructor is private and Context is its
        // friend. std::make_unique cannot invoke the private
        // constructor through the library template, so construct
        // the unique_ptr explicitly here.
        runtime = std::unique_ptr<Runtime>(new Runtime(device_type, device_id));
    }

    return runtime.get();
}

Context::Context() {
    std::vector<llaisysDeviceType_t> device_types;

    // Accelerator device types first.
    for (int device_type = 1; device_type < LLAISYS_DEVICE_TYPE_COUNT; ++device_type) {
        device_types.push_back(static_cast<llaisysDeviceType_t>(device_type));
    }

    // CPU fallback last.
    device_types.push_back(LLAISYS_DEVICE_CPU);

    for (const auto device_type : device_types) {
        const LlaisysRuntimeAPI *api = llaisysGetRuntimeAPI(device_type);

        const int device_count = api->get_device_count();

        auto &runtimes = _runtime_map[device_type];

        runtimes.resize(static_cast<std::size_t>(device_count));

        if (_current_runtime == nullptr && device_count > 0) {
            Runtime *runtime = getOrCreateRuntime(device_type, 0);

            runtime->_activate();

            _current_runtime = runtime;
        }
    }
}

void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    if (_current_runtime != nullptr && _current_runtime->deviceType() == device_type
        && _current_runtime->deviceId() == device_id) {
        return;
    }

    Runtime *next_runtime = getOrCreateRuntime(device_type, device_id);

    if (_current_runtime != nullptr) { _current_runtime->_deactivate(); }

    next_runtime->_activate();

    _current_runtime = next_runtime;
}

Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");

    return *_current_runtime;
}

Context &context() {
    thread_local Context thread_context;

    return thread_context;
}

} // namespace llaisys::core
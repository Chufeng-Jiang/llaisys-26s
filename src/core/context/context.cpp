#include <thread>

#include "../../utils.hpp"
#include "context.hpp"

namespace llaisys::core {

// Context::Context() {
//     // All device types, put CPU at the end
//     std::vector<llaisysDeviceType_t> device_typs;
//     for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++) {
//         device_typs.push_back(static_cast<llaisysDeviceType_t>(i));
//     }
//     device_typs.push_back(LLAISYS_DEVICE_CPU);

//     // Create runtimes for each device type.
//     // Activate the first available device. If no other device is available,
//     activate CPU runtime. for (auto device_type : device_typs) {
//         const LlaisysRuntimeAPI *api_ = llaisysGetRuntimeAPI(device_type);
//         int device_count = api_->get_device_count();
//         std::vector<Runtime *> runtimes_(device_count);
//         for (int device_id = 0; device_id < device_count; device_id++) {

//             if (_current_runtime == nullptr) {
//                 auto runtime = new Runtime(device_type, device_id);
//                 runtime->_activate();
//                 runtimes_[device_id] = runtime;
//                 _current_runtime = runtime;
//             }
//         }
//         _runtime_map[device_type] = runtimes_;
//     }
// }
Runtime *Context::getOrCreateRuntime(llaisysDeviceType_t device_type, int device_id) {
    auto runtime_it = _runtime_map.find(device_type);

    CHECK_ARGUMENT(runtime_it != _runtime_map.end(), "invalid device type");

    auto &runtimes = runtime_it->second;

    CHECK_ARGUMENT(
        device_id >= 0 && static_cast<std::size_t>(device_id) < runtimes.size(),
        "invalid device id");

    auto &runtime = runtimes[static_cast<std::size_t>(device_id)];

    if (!runtime) { runtime = std::unique_ptr<Runtime>(new Runtime(device_type, device_id)); }

    return runtime.get();
}

Context::Context() {
    std::vector<llaisysDeviceType_t> device_types;

    for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; ++i) {
        device_types.push_back(static_cast<llaisysDeviceType_t>(i));
    }

    // CPU fallback last.
    device_types.push_back(LLAISYS_DEVICE_CPU);

    for (const auto device_type : device_types) {
        const LlaisysRuntimeAPI *api = llaisysGetRuntimeAPI(device_type);

        const int device_count = api->get_device_count();

        auto &runtimes = _runtime_map[device_type];

        runtimes.resize(static_cast<std::size_t>(device_count));

        // Lazily construct only the first available runtime.
        if (_current_runtime == nullptr && device_count > 0) {
            Runtime *runtime = getOrCreateRuntime(device_type, 0);

            runtime->_activate();
            _current_runtime = runtime;
        }
    }
}

// Context::~Context() {
// 	// Destroy current runtime first.
// 	delete _current_runtime;

// 	for (auto &runtime_entry : _runtime_map) {
// 		std::vector<Runtime *> runtimes = runtime_entry.second;
// 		for (auto runtime : runtimes) {
// 			if (runtime != nullptr && runtime != _current_runtime) {
// 				runtime->_activate();
// 				delete runtime;
// 			}
// 		}
// 		runtimes.clear();
// 	}
// 	_current_runtime = nullptr;
// 	_runtime_map.clear();
// }

Context::~Context() {
    _current_runtime = nullptr;

    for (auto &[device_type, runtimes] : _runtime_map) {
        (void)device_type;

        for (auto &runtime : runtimes) {
            if (!runtime) { continue; }

            runtime->_activate();

            // Destroy while its device is active.
            runtime.reset();
        }
    }

    _runtime_map.clear();
}

// void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
// 	// If doest not match the current runtime.
// 	if (_current_runtime == nullptr || _current_runtime->deviceType() !=
// device_type || _current_runtime->deviceId() != device_id) { 		auto
// runtimes = _runtime_map[device_type];
// CHECK_ARGUMENT((size_t)device_id < runtimes.size() && device_id >= 0,
// "invalid device id"); 		if
// (_current_runtime != nullptr) {
// _current_runtime->_deactivate();
// 		}
// 		if (runtimes[device_id] == nullptr) {
// 			runtimes[device_id] = new Runtime(device_type,
// device_id);
// 		}
// 		runtimes[device_id]->_activate();
// 		_current_runtime = runtimes[device_id];
// 	}
// }

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

// Global API to get thread-local context.
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}

} // namespace llaisys::core

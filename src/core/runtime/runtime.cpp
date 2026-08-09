#include "runtime.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/naive_allocator.hpp"

#ifdef ENABLE_NVIDIA_API
#include "../../device/nvidia/nvidia_resource.cuh"
#endif

#include <memory>

namespace llaisys::core {

Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type), _device_id(device_id), _api(nullptr), _allocator(nullptr),
      _is_active(false), _stream(nullptr), _resource(nullptr) {
    _api = llaisys::device::getRuntimeAPI(_device_type);

    _stream = _api->create_stream();

    _allocator = new allocators::NaiveAllocator(_api);

    // Create backend-specific reusable resources.
    switch (_device_type) {
    case LLAISYS_DEVICE_CPU:
        // The CPU backend currently has no extra Runtime resource.
        _resource = nullptr;
        break;

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        _resource = std::make_unique<llaisys::device::nvidia::Resource>(_device_id);
        break;
#endif

    default:
        // Other backends can add their own Resource classes later.
        _resource = nullptr;
        break;
    }
}

Runtime::~Runtime() {
    if (!_is_active) { std::cerr << "Mallicious destruction of inactive runtime." << std::endl; }

    // Select this Runtime's device before synchronizing or
    // destroying its device-specific resources.
    if (_api != nullptr) { _api->set_device(_device_id); }

    // The Argmax workspace may still be used by kernels submitted
    // to this Runtime's stream.
    //
    // Complete those operations before freeing the workspace.
    if (_api != nullptr && _stream != nullptr) { _api->stream_synchronize(_stream); }

    // Destroy the backend-specific Resource before destroying
    // the CUDA stream.
    //
    // For NVIDIA, this invokes nvidia::Resource::~Resource(),
    // which frees the reusable Argmax workspace.
    _resource.reset();

    delete _allocator;
    _allocator = nullptr;

    if (_api != nullptr && _stream != nullptr) {
        _api->destroy_stream(_stream);

        _stream = nullptr;
    }

    _api = nullptr;
}

void Runtime::_activate() {
    _api->set_device(_device_id);

    _is_active = true;
}

void Runtime::_deactivate() { _is_active = false; }

bool Runtime::isActive() const { return _is_active; }

llaisysDeviceType_t Runtime::deviceType() const { return _device_type; }

int Runtime::deviceId() const { return _device_id; }

const LlaisysRuntimeAPI *Runtime::api() const { return _api; }

storage_t Runtime::allocateDeviceStorage(std::size_t size) {
    return std::shared_ptr<Storage>(new Storage(_allocator->allocate(size), size, *this, false));
}

storage_t Runtime::allocateHostStorage(std::size_t size) {
    return std::shared_ptr<Storage>(
        new Storage(reinterpret_cast<std::byte *>(_api->malloc_host(size)), size, *this, true));
}

void Runtime::freeStorage(Storage *storage) {
    if (storage->isHost()) {
        _api->free_host(storage->memory());
    } else {
        _allocator->release(storage->memory());
    }
}

llaisysStream_t Runtime::stream() const { return _stream; }

llaisys::device::DeviceResource *Runtime::resource() { return _resource.get(); }

const llaisys::device::DeviceResource *Runtime::resource() const { return _resource.get(); }

void Runtime::synchronize() const { _api->stream_synchronize(_stream); }

} // namespace llaisys::core
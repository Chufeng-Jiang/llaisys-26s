#include "runtime.hpp"
#include "../allocator/naive_allocator.hpp"
#include "../storage/storage.hpp"

#include <memory>
#include <utility>

namespace llaisys::core {

Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type), _device_id(device_id), _api(nullptr), _allocator(nullptr),
      _is_active(false), _stream(nullptr), _resource(nullptr) {
    // ========================================================
    // Runtime API
    // ========================================================

    _api = llaisys::device::getRuntimeAPI(_device_type);

    // ========================================================
    // Select the device BEFORE creating device-owned resources
    // ========================================================
    //
    // Streams and backend resources are associated with the
    // currently selected accelerator device.
    // ========================================================

    _api->set_device(_device_id);

    // ========================================================
    // Stream
    // ========================================================

    _stream = _api->create_stream();

    try {
        // ====================================================
        // Device-memory allocator
        // ====================================================

        _allocator = std::make_unique<allocators::NaiveAllocator>(_api);

        // ====================================================
        // Optional backend-specific reusable resource
        // ====================================================

        _resource = llaisys::device::createDeviceResource(_device_type, _device_id);
    } catch (...) {
        // Runtime construction did not complete, therefore the
        // destructor will not run. Release resources created
        // before the exception here.

        _resource.reset();
        _allocator.reset();

        if (_api != nullptr && _stream != nullptr) {
            try {
                _api->destroy_stream(_stream);
            } catch (...) {
                // Best-effort cleanup while propagating the
                // original construction failure.
            }

            _stream = nullptr;
        }

        throw;
    }
}

Runtime::~Runtime() noexcept {
    // Runtime destruction must be self-contained.
    //
    // Context does not need to activate this Runtime before
    // destroying it.

    if (_api == nullptr) { return; }

    // ========================================================
    // Select this Runtime's own device
    // ========================================================

    try {
        _api->set_device(_device_id);
    } catch (...) {
        // Destructors must not throw.
    }

    // ========================================================
    // Complete work submitted to this Runtime's stream
    // ========================================================

    if (_stream != nullptr) {
        try {
            _api->stream_synchronize(_stream);
        } catch (...) {
            // Best-effort cleanup.
        }
    }

    // ========================================================
    // Release backend resources
    // ========================================================
    //
    // Resource destruction occurs before stream destruction
    // because resources may have been used by work submitted
    // to this stream.
    // ========================================================

    _resource.reset();

    // ========================================================
    // Release allocator
    // ========================================================

    _allocator.reset();

    // ========================================================
    // Destroy Runtime-owned stream
    // ========================================================

    if (_stream != nullptr) {
        try {
            _api->destroy_stream(_stream);
        } catch (...) {
            // Destructors must not throw.
        }

        _stream = nullptr;
    }

    _is_active = false;

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

        return;
    }

    _allocator->release(storage->memory());
}

llaisysStream_t Runtime::stream() const { return _stream; }

llaisys::device::DeviceResource *Runtime::resource() { return _resource.get(); }

const llaisys::device::DeviceResource *Runtime::resource() const { return _resource.get(); }

void Runtime::synchronize() const { _api->stream_synchronize(_stream); }

} // namespace llaisys::core
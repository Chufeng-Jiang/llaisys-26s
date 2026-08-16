#include "runtime.hpp"

#include "../allocator/naive_allocator.hpp"
#include "../storage/storage.hpp"

#include <memory>

namespace llaisys::core {

Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type),
      _device_id(device_id),
      _api(nullptr),
      _allocator(nullptr),
      _is_active(false),
      _stream(nullptr) {
    _api = llaisys::device::getRuntimeAPI(_device_type);

    _api->set_device(_device_id);

    _stream = _api->create_stream();

    try {
        _allocator = std::make_unique<allocators::NaiveAllocator>(_api);
    } catch (...) {
        if (_stream != nullptr) {
            try {
                _api->destroy_stream(_stream);
            } catch (...) {
            }

            _stream = nullptr;
        }

        throw;
    }
}

Runtime::~Runtime() noexcept {
    if (_api == nullptr) { return; }

    try {
        _api->set_device(_device_id);
    } catch (...) {
    }

    if (_stream != nullptr) {
        try {
            _api->stream_synchronize(_stream);
        } catch (...) {
        }
    }

    _allocator.reset();

    if (_stream != nullptr) {
        try {
            _api->destroy_stream(_stream);
        } catch (...) {
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

llaisysDeviceType_t Runtime::deviceType() const { return _device_type; }

int Runtime::deviceId() const { return _device_id; }

bool Runtime::isActive() const { return _is_active; }

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

void Runtime::synchronize() const { _api->stream_synchronize(_stream); }

} // namespace llaisys::core
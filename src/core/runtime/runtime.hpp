#pragma once

#include "llaisys.h"

#include "../../device/device_resource.hpp"
#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"
#include "../core.hpp"

#include <memory>

namespace llaisys::core {

class Runtime {
private:
	llaisysDeviceType_t _device_type;
	int _device_id;

	const LlaisysRuntimeAPI *_api;
	MemoryAllocator *_allocator;

	bool _is_active;

	llaisysStream_t _stream;

	// Store backend-specific reusable resources.
	//
	// CPU:
	//     nullptr
	//
	// NVIDIA:
	//     llaisys::device::nvidia::Resource
	std::unique_ptr<llaisys::device::DeviceResource> _resource;

	void _activate();
	void _deactivate();

	Runtime(
		llaisysDeviceType_t device_type,
		int device_id
	);

public:
	friend class Context;

	~Runtime();

	// Prevent copying.
	Runtime(const Runtime &) = delete;
	Runtime &operator=(const Runtime &) = delete;

	// Prevent moving.
	Runtime(Runtime &&) = delete;
	Runtime &operator=(Runtime &&) = delete;

	llaisysDeviceType_t deviceType() const;
	int deviceId() const;
	bool isActive() const;

	const LlaisysRuntimeAPI *api() const;

	storage_t allocateDeviceStorage(std::size_t size);
	storage_t allocateHostStorage(std::size_t size);

	void freeStorage(Storage *storage);

	llaisysStream_t stream() const;

	llaisys::device::DeviceResource *resource();
	const llaisys::device::DeviceResource *resource() const;

	void synchronize() const;
};

} // namespace llaisys::core
#pragma once

#include "../device_resource.hpp"

namespace llaisys::device::nvidia {

class Resource final
	: public llaisys::device::DeviceResource {
public:
	explicit Resource(int device_id);

	~Resource() override;

	// Prevent copying.
	Resource(const Resource &) = delete;
	Resource &operator=(const Resource &) = delete;

	// Prevent moving.
	Resource(Resource &&) = delete;
	Resource &operator=(Resource &&) = delete;

	// Lazily allocate and return the reusable Argmax workspace.
	//
	// The workspace stores:
	// - one FP32 value;
	// - one uint32 index;
	//
	// packed into one unsigned long long.
	unsigned long long *argmaxPackedWorkspace();

private:
	unsigned long long *_argmax_packed_workspace{nullptr};
};

} // namespace llaisys::device::nvidia
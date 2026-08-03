#pragma once

#include "../device_resource.hpp"

#include "llaisys.h"

#include <cublas_v2.h>

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

// ============================================================
// cuBLAS resource and error helpers
// ============================================================

const char *cublas_status_name(
	cublasStatus_t status
) noexcept;

void check_cublas(
	cublasStatus_t status,
	const char *expression,
	const char *file,
	int line,
	const char *function
);

// Return a reusable cuBLAS handle for the current host thread and current
// CUDA device. The handle is rebound to the supplied LLAISYS Runtime stream
// before it is returned.
//
// The provided Runtime API exposes streams but does not expose its private
// DeviceResource object to operators. Keeping this handle pool in the NVIDIA
// resource translation unit avoids modifying the generic Runtime interface.
cublasHandle_t get_cublas_handle(
	llaisysStream_t stream
);

} // namespace llaisys::device::nvidia

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(CALL) \
	::llaisys::device::nvidia::check_cublas( \
		(CALL), \
		#CALL, \
		__FILE__, \
		__LINE__, \
		__func__ \
	)
#endif
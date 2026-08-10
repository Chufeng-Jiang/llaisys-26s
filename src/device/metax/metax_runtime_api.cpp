#include "../runtime_api.hpp"

#include "../../utils.hpp"
#include "metax_common.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace llaisys::device::metax {

namespace runtime_api {

namespace {

mcMemcpyKind toMcMemcpyKind(
	llaisysMemcpyKind_t kind
) {
	switch (kind) {
	case LLAISYS_MEMCPY_H2D:
		return mcMemcpyHostToDevice;

	case LLAISYS_MEMCPY_D2H:
		return mcMemcpyDeviceToHost;

	case LLAISYS_MEMCPY_D2D:
		return mcMemcpyDeviceToDevice;

	default:
		throw std::invalid_argument(
			"Unknown MetaX memory copy kind."
		);
	}
}

} // namespace

int getDeviceCount() {
	int count = 0;

	MC_CHECK(
		mcGetDeviceCount(
			&count
		)
	);

	return count;
}

void setDevice(
	int device_id
) {
	MC_CHECK(
		mcSetDevice(
			device_id
		)
	);
}

void deviceSynchronize() {
	MC_CHECK(
		mcDeviceSynchronize()
	);
}

llaisysStream_t createStream() {
	mcStream_t stream = nullptr;

	MC_CHECK(
		mcStreamCreate(
			&stream
		)
	);

	return reinterpret_cast<llaisysStream_t>(
		stream
	);
}

void destroyStream(
	llaisysStream_t stream
) {
	if (stream == nullptr) {
		return;
	}

	MC_CHECK(
		mcStreamDestroy(
			reinterpret_cast<mcStream_t>(
				stream
			)
		)
	);
}

void streamSynchronize(
	llaisysStream_t stream
) {
	MC_CHECK(
		mcStreamSynchronize(
			reinterpret_cast<mcStream_t>(
				stream
			)
		)
	);
}

void *mallocDevice(
	std::size_t size
) {
	if (size == 0) {
		return nullptr;
	}

	void *pointer = nullptr;

	MC_CHECK(
		mcMalloc(
			&pointer,
			size
		)
	);

	return pointer;
}

void freeDevice(
	void *pointer
) {
	if (pointer == nullptr) {
		return;
	}

	MC_CHECK(
		mcFree(
			pointer
		)
	);
}

void *mallocHost(
	std::size_t size
) {
	if (size == 0) {
		return nullptr;
	}

	void *pointer = nullptr;

	MC_CHECK(
		mcMallocHost(
			&pointer,
			size
		)
	);

	return pointer;
}

void freeHost(
	void *pointer
) {
	if (pointer == nullptr) {
		return;
	}

	MC_CHECK(
		mcFreeHost(
			pointer
		)
	);
}

void memcpySync(
	void *destination,
	const void *source,
	std::size_t size,
	llaisysMemcpyKind_t kind
) {
	if (size == 0) {
		return;
	}

	CHECK_ARGUMENT(
		destination != nullptr,
		"MetaX memcpySync: destination must not be null."
	);

	CHECK_ARGUMENT(
		source != nullptr,
		"MetaX memcpySync: source must not be null."
	);

	// H2H does not require the accelerator runtime.
	if (kind == LLAISYS_MEMCPY_H2H) {
		std::memcpy(
			destination,
			source,
			size
		);

		return;
	}

	MC_CHECK(
		mcMemcpy(
			destination,
			source,
			size,
			toMcMemcpyKind(kind)
		)
	);
}

void memcpyAsync(
	void *destination,
	const void *source,
	std::size_t size,
	llaisysMemcpyKind_t kind,
	llaisysStream_t stream
) {
	if (size == 0) {
		return;
	}

	CHECK_ARGUMENT(
		destination != nullptr,
		"MetaX memcpyAsync: destination must not be null."
	);

	CHECK_ARGUMENT(
		source != nullptr,
		"MetaX memcpyAsync: source must not be null."
	);

	// A synchronous host copy is still semantically safe for H2H.
	// Accelerator transfers continue to use the supplied MACA stream.
	if (kind == LLAISYS_MEMCPY_H2H) {
		std::memcpy(
			destination,
			source,
			size
		);

		return;
	}

	MC_CHECK(
		mcMemcpyAsync(
			destination,
			source,
			size,
			toMcMemcpyKind(kind),
			reinterpret_cast<mcStream_t>(
				stream
			)
		)
	);
}

const LlaisysRuntimeAPI RUNTIME_API = {
	&getDeviceCount,
	&setDevice,
	&deviceSynchronize,
	&createStream,
	&destroyStream,
	&streamSynchronize,
	&mallocDevice,
	&freeDevice,
	&mallocHost,
	&freeHost,
	&memcpySync,
	&memcpyAsync,
};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
	return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::metax
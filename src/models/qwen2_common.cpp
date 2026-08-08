#include "qwen2_common.hpp"

#include "../core/context/context.hpp"

#include <cstring>
#include <stdexcept>

llaisys::tensor_t qwen2_create_tensor(
	const LlaisysQwen2Model &model,
	const std::vector<std::size_t> &shape,
	llaisysDataType_t dtype
) {
	if (model.device_ids.empty()) {
		throw std::runtime_error(
			"Qwen2 has no configured device ID."
		);
	}

	return llaisys::Tensor::create(
		shape,
		dtype,
		model.device,
		model.device_ids.front()
	);
}

void qwen2_copy_to_host(
	void *destination,
	const llaisys::tensor_t &source,
	std::size_t nbytes
) {
	if (destination == nullptr) {
		throw std::invalid_argument(
			"Qwen2 host destination must not be null."
		);
	}

	if (source == nullptr) {
		throw std::invalid_argument(
			"Qwen2 source tensor must not be null."
		);
	}

	if (nbytes == 0) {
		return;
	}

	if (source->deviceType() == LLAISYS_DEVICE_CPU) {
		std::memcpy(
			destination,
			source->data(),
			nbytes
		);
		return;
	}

	llaisys::core::context().setDevice(
		source->deviceType(),
		source->deviceId()
	);

	llaisys::core::context()
		.runtime()
		.api()
		->memcpy_sync(
			destination,
			source->data(),
			nbytes,
			LLAISYS_MEMCPY_D2H
		);
}

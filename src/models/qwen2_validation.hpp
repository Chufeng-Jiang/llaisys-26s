#pragma once

#include "qwen2.hpp"

#include <cstddef>
#include <cstdint>

void qwen2_validate_model_configuration(
	const LlaisysQwen2Meta &meta,
	const int *device_ids,
	int ndevice
);

void qwen2_validate_inference_request(
	const LlaisysQwen2Model &model,
	const std::int64_t *token_ids,
	std::size_t ntoken
);

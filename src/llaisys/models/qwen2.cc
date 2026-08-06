#include "llaisys/models/qwen2.h"

#include "../../models/qwen2.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>

__C {

LlaisysQwen2Model *llaisysQwen2ModelCreate(
	const LlaisysQwen2Meta *meta,
	llaisysDeviceType_t device,
	int *device_ids,
	int ndevice
) {
	if (meta == nullptr) {
		std::cerr
			<< "Qwen2: meta must not be null."
			<< std::endl;

		return nullptr;
	}

	try {
		return new LlaisysQwen2Model(
			*meta,
			device,
			device_ids,
			ndevice
		);
	} catch (const std::exception &error) {
		std::cerr
			<< "Qwen2 model creation failed: "
			<< error.what()
			<< std::endl;

		return nullptr;
	}
}

void llaisysQwen2ModelDestroy(
	LlaisysQwen2Model *model
) {
	delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(
	LlaisysQwen2Model *model
) {
	if (model == nullptr) {
		return nullptr;
	}

	return &model->weights;
}

void llaisysQwen2ModelReset(
	LlaisysQwen2Model *model
) {
	if (model == nullptr) {
		return;
	}

	model->reset_cache();
}

std::int64_t llaisysQwen2ModelInfer(
	LlaisysQwen2Model *model,
	std::int64_t *token_ids,
	std::size_t ntoken
) {
	if (model == nullptr) {
		std::cerr
			<< "Qwen2 inference failed: model is null."
			<< std::endl;

		return -1;
	}

	try {
		return model->infer(
			token_ids,
			ntoken
		);
	} catch (const std::exception &error) {
		std::cerr
			<< "Qwen2 inference failed: "
			<< error.what()
			<< std::endl;

		return -1;
	}
}

} // extern "C"
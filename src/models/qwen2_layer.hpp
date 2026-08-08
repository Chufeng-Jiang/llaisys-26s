#pragma once

#include "qwen2.hpp"
#include "qwen2_weights.hpp"

#include <cstddef>

void qwen2_layer_forward(
	LlaisysQwen2Model &model,
	std::size_t layer,
	const Qwen2LayerWeights &weights,
	const llaisys::tensor_t &hidden_states,
	const llaisys::tensor_t &position_ids,
	std::size_t sequence_length,
	std::size_t previous_cache_length,
	std::size_t total_length,
	const llaisys::tensor_t &output_hidden
);

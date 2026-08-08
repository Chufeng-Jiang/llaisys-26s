#pragma once

#include "qwen2.hpp"
#include "qwen2_weights.hpp"

#include <cstddef>

llaisys::tensor_t qwen2_attention_forward(
	LlaisysQwen2Model &model,
	std::size_t layer,
	const Qwen2LayerWeights &weights,
	const llaisys::tensor_t &hidden_states,
	const llaisys::tensor_t &position_ids,
	std::size_t sequence_length,
	std::size_t previous_cache_length,
	std::size_t total_length
);

#pragma once

#include "qwen2.hpp"
#include "qwen2_weights.hpp"

#include <cstddef>
#include <cstdint>

struct Qwen2PreparedInput {
	llaisys::tensor_t hidden_states;
	llaisys::tensor_t position_ids;
};

Qwen2PreparedInput qwen2_prepare_input(
	LlaisysQwen2Model &model,
	const std::int64_t *token_ids,
	std::size_t sequence_length,
	std::size_t previous_cache_length,
	const Qwen2GlobalWeights &weights
);

std::int64_t qwen2_predict_next_token(
	LlaisysQwen2Model &model,
	const llaisys::tensor_t &hidden_states,
	std::size_t sequence_length,
	const Qwen2GlobalWeights &weights
);

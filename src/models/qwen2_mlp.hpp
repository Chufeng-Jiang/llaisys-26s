#pragma once

#include "qwen2.hpp"
#include "qwen2_weights.hpp"

#include <cstddef>

void qwen2_mlp_forward(
    LlaisysQwen2Model &model,
    const Qwen2LayerWeights &weights,
    const llaisys::tensor_t &attention_residual,
    std::size_t sequence_length,
    const llaisys::tensor_t &output_hidden);

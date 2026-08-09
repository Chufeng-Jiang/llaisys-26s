#pragma once

#include "qwen2.hpp"

#include <cstddef>

struct Qwen2GlobalWeights {
    llaisys::tensor_t input_embedding;
    llaisys::tensor_t output_embedding;
    llaisys::tensor_t output_norm;
};

struct Qwen2LayerWeights {
    llaisys::tensor_t attention_norm;

    llaisys::tensor_t query_weight;
    llaisys::tensor_t query_bias;

    llaisys::tensor_t key_weight;
    llaisys::tensor_t key_bias;

    llaisys::tensor_t value_weight;
    llaisys::tensor_t value_bias;

    llaisys::tensor_t output_weight;

    llaisys::tensor_t mlp_norm;
    llaisys::tensor_t gate_weight;
    llaisys::tensor_t up_weight;
    llaisys::tensor_t down_weight;
};

void qwen2_allocate_weight_arrays(LlaisysQwen2Weights &weights, std::size_t nlayer);

void qwen2_release_weight_arrays(LlaisysQwen2Weights &weights) noexcept;

void qwen2_validate_layer_weight_arrays(const LlaisysQwen2Model &model);

Qwen2GlobalWeights qwen2_global_weights(const LlaisysQwen2Model &model);

Qwen2LayerWeights qwen2_layer_weights(const LlaisysQwen2Model &model, std::size_t layer);

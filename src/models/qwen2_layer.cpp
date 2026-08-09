#include "qwen2_layer.hpp"

#include "qwen2_attention.hpp"
#include "qwen2_mlp.hpp"

void qwen2_layer_forward(
    LlaisysQwen2Model &model,
    std::size_t layer,
    const Qwen2LayerWeights &weights,
    const llaisys::tensor_t &hidden_states,
    const llaisys::tensor_t &position_ids,
    std::size_t sequence_length,
    std::size_t previous_cache_length,
    std::size_t total_length,
    const llaisys::tensor_t &output_hidden) {
    auto attention_residual = qwen2_attention_forward(
        model,
        layer,
        weights,
        hidden_states,
        position_ids,
        sequence_length,
        previous_cache_length,
        total_length);

    qwen2_mlp_forward(model, weights, attention_residual, sequence_length, output_hidden);
}

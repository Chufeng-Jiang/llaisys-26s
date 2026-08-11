#include "qwen2_attention.hpp"

#include "qwen2_workspace.hpp"

#include "../ops/add/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"

#include <cmath>
#include <stdexcept>

namespace {

Qwen2Workspace &workspace_of(LlaisysQwen2Model &model) {
    if (model.workspace == nullptr) {
        throw std::runtime_error("Qwen2 workspace is not initialized.");
    }

    return *model.workspace;
}

} // namespace

llaisys::tensor_t qwen2_attention_forward(
    LlaisysQwen2Model &model,
    std::size_t layer,
    const Qwen2LayerWeights &weights,
    const llaisys::tensor_t &hidden_states,
    const llaisys::tensor_t &position_ids,
    std::size_t sequence_length,
    std::size_t previous_cache_length,
    std::size_t total_length) {
    auto &workspace = workspace_of(model);

    const std::size_t hidden_size = model.meta.hs;

    const std::size_t attention_heads = model.meta.nh;

    const std::size_t kv_heads = model.meta.nkvh;

    const std::size_t head_dimension = model.meta.dh;

    const auto dtype = model.meta.dtype;

    auto normalized_hidden
        = workspace.get(Qwen2WorkspaceSlot::AttentionNorm, {sequence_length, hidden_size}, dtype);

    llaisys::ops::rms_norm(
        normalized_hidden, hidden_states, weights.attention_norm, model.meta.epsilon);

    auto query_2d = workspace.get(
        Qwen2WorkspaceSlot::Query, {sequence_length, attention_heads * head_dimension}, dtype);

    auto key_2d = workspace.get(
        Qwen2WorkspaceSlot::Key, {sequence_length, kv_heads * head_dimension}, dtype);

    auto value_2d = workspace.get(
        Qwen2WorkspaceSlot::Value, {sequence_length, kv_heads * head_dimension}, dtype);

    llaisys::ops::linear(query_2d, normalized_hidden, weights.query_weight, weights.query_bias);

    llaisys::ops::linear(key_2d, normalized_hidden, weights.key_weight, weights.key_bias);

    llaisys::ops::linear(value_2d, normalized_hidden, weights.value_weight, weights.value_bias);

    auto query_3d = query_2d->view({sequence_length, attention_heads, head_dimension});

    auto key_3d = key_2d->view({sequence_length, kv_heads, head_dimension});

    auto value_3d = value_2d->view({sequence_length, kv_heads, head_dimension});

    auto rotated_query = workspace.get(
        Qwen2WorkspaceSlot::RotatedQuery,
        {sequence_length, attention_heads, head_dimension},
        dtype);

    auto rotated_key = workspace.get(
        Qwen2WorkspaceSlot::RotatedKey, {sequence_length, kv_heads, head_dimension}, dtype);

    llaisys::ops::rope(rotated_query, query_3d, position_ids, model.meta.theta);

    llaisys::ops::rope(rotated_key, key_3d, position_ids, model.meta.theta);

    auto key_cache_destination
        = model.key_cache[layer]->slice(0, previous_cache_length, total_length);

    auto value_cache_destination
        = model.value_cache[layer]->slice(0, previous_cache_length, total_length);

    llaisys::ops::rearrange(key_cache_destination, rotated_key);

    llaisys::ops::rearrange(value_cache_destination, value_3d);

    auto key_history = model.key_cache[layer]->slice(0, 0, total_length);

    auto value_history = model.value_cache[layer]->slice(0, 0, total_length);

    auto attention_output_3d = workspace.get(
        Qwen2WorkspaceSlot::AttentionOutput,
        {sequence_length, attention_heads, head_dimension},
        dtype);

    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));

    llaisys::ops::self_attention(
        attention_output_3d, rotated_query, key_history, value_history, attention_scale);

    auto attention_output_2d = attention_output_3d->view({sequence_length, hidden_size});

    auto projected_attention = workspace.get(
        Qwen2WorkspaceSlot::AttentionProjection, {sequence_length, hidden_size}, dtype);

    llaisys::ops::linear(projected_attention, attention_output_2d, weights.output_weight, nullptr);

    auto attention_residual = workspace.get(
        Qwen2WorkspaceSlot::AttentionResidual, {sequence_length, hidden_size}, dtype);

    llaisys::ops::add(attention_residual, hidden_states, projected_attention);

    return attention_residual;
}

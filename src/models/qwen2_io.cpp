#include "qwen2_io.hpp"

#include "qwen2_common.hpp"
#include "qwen2_workspace.hpp"

#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"

#include <cstdint>
#include <stdexcept>

namespace {

Qwen2Workspace &workspace_of(LlaisysQwen2Model &model) {
    if (model.workspace == nullptr) {
        throw std::runtime_error("Qwen2 workspace is not initialized.");
    }

    return *model.workspace;
}

} // namespace

Qwen2PreparedInput qwen2_prepare_input(
    LlaisysQwen2Model &model,
    const std::int64_t *token_ids,
    std::size_t sequence_length,
    std::size_t previous_cache_length,
    const Qwen2GlobalWeights &weights) {
    auto &workspace = workspace_of(model);

    auto input_ids
        = workspace.get(Qwen2WorkspaceSlot::InputIds, {sequence_length}, LLAISYS_DTYPE_I64);

    input_ids->load(token_ids);

    auto &position_values = workspace.position_values(sequence_length);

    for (std::size_t index = 0; index < sequence_length; ++index) {
        position_values[index] = static_cast<std::int64_t>(previous_cache_length + index);
    }

    auto position_ids
        = workspace.get(Qwen2WorkspaceSlot::PositionIds, {sequence_length}, LLAISYS_DTYPE_I64);

    position_ids->load(position_values.data());

    auto hidden_states = workspace.get(
        Qwen2WorkspaceSlot::Hidden0, {sequence_length, model.meta.hs}, model.meta.dtype);

    llaisys::ops::embedding(hidden_states, input_ids, weights.input_embedding);

    return Qwen2PreparedInput{hidden_states, position_ids};
}

std::int64_t qwen2_predict_next_token(
    LlaisysQwen2Model &model,
    const llaisys::tensor_t &hidden_states,
    std::size_t sequence_length,
    const Qwen2GlobalWeights &weights) {
    if (hidden_states == nullptr) {
        throw std::invalid_argument("Qwen2 output hidden state must not be null.");
    }

    auto &workspace = workspace_of(model);

    auto last_hidden = hidden_states->slice(0, sequence_length - 1, sequence_length);

    auto normalized_last_hidden
        = workspace.get(Qwen2WorkspaceSlot::AttentionNorm, {1, model.meta.hs}, model.meta.dtype);

    llaisys::ops::rms_norm(
        normalized_last_hidden, last_hidden, weights.output_norm, model.meta.epsilon);

    auto logits_2d
        = workspace.get(Qwen2WorkspaceSlot::Logits, {1, model.meta.voc}, model.meta.dtype);

    llaisys::ops::linear(logits_2d, normalized_last_hidden, weights.output_embedding, nullptr);

    auto logits = logits_2d->view({model.meta.voc});

    auto max_index = workspace.get(Qwen2WorkspaceSlot::MaxIndex, {1}, LLAISYS_DTYPE_I64);

    auto max_value = workspace.get(Qwen2WorkspaceSlot::MaxValue, {1}, model.meta.dtype);

    llaisys::ops::argmax(max_index, max_value, logits);

    std::int64_t next_token = -1;

    // Copy the single argmax index directly into host memory instead of
    // allocating a temporary CPU Tensor on every generated token.
    qwen2_copy_to_host(&next_token, max_index, sizeof(next_token));

    return next_token;
}

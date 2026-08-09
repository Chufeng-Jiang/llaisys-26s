#include "qwen2.hpp"

#include "qwen2_io.hpp"
#include "qwen2_layer.hpp"
#include "qwen2_validation.hpp"
#include "qwen2_weights.hpp"
#include "qwen2_workspace.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

LlaisysQwen2Model::LlaisysQwen2Model(
    const LlaisysQwen2Meta &model_meta,
    llaisysDeviceType_t model_device,
    const int *model_device_ids,
    int ndevice)
    : meta(model_meta), device(model_device) {
    qwen2_validate_model_configuration(meta, model_device_ids, ndevice);

    device_ids.assign(model_device_ids, model_device_ids + ndevice);

    // Perform potentially-throwing RAII allocations before publishing the
    // raw C-compatible weight-array pointers. If one of these operations
    // throws, their own destructors clean themselves up automatically.
    key_cache.resize(meta.nlayer);

    value_cache.resize(meta.nlayer);

    workspace = std::make_unique<Qwen2Workspace>(device, device_ids.front());

    // This function is internally exception-safe: all arrays are held by
    // temporary unique_ptr objects until every allocation has succeeded.
    qwen2_allocate_weight_arrays(weights, meta.nlayer);
}

LlaisysQwen2Model::~LlaisysQwen2Model() {
    // Scratch storage can be dropped independently from the persistent cache.
    workspace.reset();

    key_cache.clear();
    value_cache.clear();

    cache_length = 0;
    cache_capacity = 0;

    qwen2_release_weight_arrays(weights);
}

std::int64_t LlaisysQwen2Model::infer(const std::int64_t *token_ids, std::size_t ntoken) {
    qwen2_validate_inference_request(*this, token_ids, ntoken);

    // Resolve global handles once per infer() call.
    const Qwen2GlobalWeights global_weights = qwen2_global_weights(*this);

    qwen2_validate_layer_weight_arrays(*this);

    const std::size_t sequence_length = ntoken;

    const std::size_t previous_cache_length = cache_length;

    const std::size_t total_length = previous_cache_length + sequence_length;

    ensure_cache_capacity(total_length);

    Qwen2PreparedInput prepared = qwen2_prepare_input(
        *this, token_ids, sequence_length, previous_cache_length, global_weights);

    llaisys::tensor_t hidden_states = prepared.hidden_states;

    for (std::size_t layer = 0; layer < meta.nlayer; ++layer) {
        // Collapse all weights.xxx[layer] indexing into one validated
        // per-layer view.
        const Qwen2LayerWeights layer_weights = qwen2_layer_weights(*this, layer);

        // Hidden-state storage is ping-ponged between two persistent
        // workspace slots. No new device allocation is needed per layer.
        const Qwen2WorkspaceSlot output_slot
            = (layer % 2 == 0) ? Qwen2WorkspaceSlot::Hidden1 : Qwen2WorkspaceSlot::Hidden0;

        auto next_hidden_states
            = workspace->get(output_slot, {sequence_length, meta.hs}, meta.dtype);

        qwen2_layer_forward(
            *this,
            layer,
            layer_weights,
            hidden_states,
            prepared.position_ids,
            sequence_length,
            previous_cache_length,
            total_length,
            next_hidden_states);

        hidden_states = std::move(next_hidden_states);
    }

    // Commit only after every transformer layer has completed.
    // If a layer throws, cache_length remains unchanged and the same
    // cache interval can be overwritten safely by a retry.
    cache_length = total_length;

    return qwen2_predict_next_token(*this, hidden_states, sequence_length, global_weights);
}

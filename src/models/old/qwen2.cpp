#include "qwen2.hpp"

#include "../llaisys/llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

llaisysTensor_t *allocate_layer_weights(std::size_t nlayer) {
    return new llaisysTensor_t[nlayer]{};
}

llaisys::tensor_t require_tensor(llaisysTensor_t handle, const std::string &name) {
    if (handle == nullptr) { throw std::runtime_error("Qwen2 weight handle is null: " + name); }

    if (handle->tensor == nullptr) {
        throw std::runtime_error("Qwen2 internal tensor is null: " + name);
    }

    return handle->tensor;
}

llaisys::tensor_t create_tensor(
    const LlaisysQwen2Model &model,
    const std::vector<std::size_t> &shape,
    llaisysDataType_t dtype) {
    if (model.device_ids.empty()) {
        throw std::runtime_error("Qwen2 has no configured device ID.");
    }

    return llaisys::Tensor::create(shape, dtype, model.device, model.device_ids.front());
}

void validate_global_weights(const LlaisysQwen2Model &model) {
    require_tensor(model.weights.in_embed, "model.embed_tokens.weight");

    require_tensor(model.weights.out_embed, "lm_head.weight");

    require_tensor(model.weights.out_norm_w, "model.norm.weight");
}

void validate_layer_weight_arrays(const LlaisysQwen2Model &model) {
    if (model.weights.attn_norm_w == nullptr || model.weights.attn_q_w == nullptr
        || model.weights.attn_q_b == nullptr || model.weights.attn_k_w == nullptr
        || model.weights.attn_k_b == nullptr || model.weights.attn_v_w == nullptr
        || model.weights.attn_v_b == nullptr || model.weights.attn_o_w == nullptr
        || model.weights.mlp_norm_w == nullptr || model.weights.mlp_gate_w == nullptr
        || model.weights.mlp_up_w == nullptr || model.weights.mlp_down_w == nullptr) {
        throw std::runtime_error("Qwen2 layer-weight arrays are not initialized.");
    }
}

} // namespace

LlaisysQwen2Model::LlaisysQwen2Model(
    const LlaisysQwen2Meta &model_meta,
    llaisysDeviceType_t model_device,
    const int *model_device_ids,
    int ndevice)
    : meta(model_meta), device(model_device) {
    if (ndevice <= 0) { throw std::invalid_argument("Qwen2: ndevice must be greater than zero."); }

    if (model_device_ids == nullptr) {
        throw std::invalid_argument("Qwen2: device_ids must not be null.");
    }

    // 当前版本只实现单设备推理。
    if (ndevice != 1) {
        throw std::invalid_argument("Qwen2: only one inference device is currently supported.");
    }

    if (meta.nlayer == 0) {
        throw std::invalid_argument("Qwen2: number of layers must be greater than zero.");
    }

    if (meta.hs == 0) {
        throw std::invalid_argument("Qwen2: hidden size must be greater than zero.");
    }

    if (meta.nh == 0) {
        throw std::invalid_argument("Qwen2: attention head count must be greater than zero.");
    }

    if (meta.nkvh == 0) {
        throw std::invalid_argument("Qwen2: KV head count must be greater than zero.");
    }

    if (meta.dh == 0) {
        throw std::invalid_argument("Qwen2: head dimension must be greater than zero.");
    }

    if (meta.hs != meta.nh * meta.dh) {
        throw std::invalid_argument(
            "Qwen2: hidden size must equal "
            "attention head count times head dimension.");
    }

    if (meta.nh % meta.nkvh != 0) {
        throw std::invalid_argument(
            "Qwen2: attention head count must be divisible "
            "by KV head count.");
    }

    if (meta.di == 0) {
        throw std::invalid_argument("Qwen2: intermediate size must be greater than zero.");
    }

    if (meta.maxseq == 0) {
        throw std::invalid_argument("Qwen2: maximum sequence length must be greater than zero.");
    }

    if (meta.voc == 0) {
        throw std::invalid_argument("Qwen2: vocabulary size must be greater than zero.");
    }

    device_ids.assign(model_device_ids, model_device_ids + ndevice);

    const std::size_t nlayer = meta.nlayer;

    weights.attn_norm_w = allocate_layer_weights(nlayer);

    weights.attn_q_w = allocate_layer_weights(nlayer);

    weights.attn_q_b = allocate_layer_weights(nlayer);

    weights.attn_k_w = allocate_layer_weights(nlayer);

    weights.attn_k_b = allocate_layer_weights(nlayer);

    weights.attn_v_w = allocate_layer_weights(nlayer);

    weights.attn_v_b = allocate_layer_weights(nlayer);

    weights.attn_o_w = allocate_layer_weights(nlayer);

    weights.mlp_norm_w = allocate_layer_weights(nlayer);

    weights.mlp_gate_w = allocate_layer_weights(nlayer);

    weights.mlp_up_w = allocate_layer_weights(nlayer);

    weights.mlp_down_w = allocate_layer_weights(nlayer);

    key_cache.resize(nlayer);
    value_cache.resize(nlayer);
}

LlaisysQwen2Model::~LlaisysQwen2Model() {
    key_cache.clear();
    value_cache.clear();

    cache_length = 0;
    cache_capacity = 0;

    delete[] weights.attn_norm_w;

    delete[] weights.attn_q_w;
    delete[] weights.attn_q_b;

    delete[] weights.attn_k_w;
    delete[] weights.attn_k_b;

    delete[] weights.attn_v_w;
    delete[] weights.attn_v_b;

    delete[] weights.attn_o_w;

    delete[] weights.mlp_norm_w;
    delete[] weights.mlp_gate_w;
    delete[] weights.mlp_up_w;
    delete[] weights.mlp_down_w;

    weights.attn_norm_w = nullptr;

    weights.attn_q_w = nullptr;
    weights.attn_q_b = nullptr;

    weights.attn_k_w = nullptr;
    weights.attn_k_b = nullptr;

    weights.attn_v_w = nullptr;
    weights.attn_v_b = nullptr;

    weights.attn_o_w = nullptr;

    weights.mlp_norm_w = nullptr;
    weights.mlp_gate_w = nullptr;
    weights.mlp_up_w = nullptr;
    weights.mlp_down_w = nullptr;

    // 权重 Tensor 由 Python 侧拥有。
    weights.in_embed = nullptr;
    weights.out_embed = nullptr;
    weights.out_norm_w = nullptr;
}

void LlaisysQwen2Model::reset_cache() {
    // 保留已经分配的内存，下次 generate 可以重复使用。
    // 只把有效缓存长度归零。
    cache_length = 0;
}

void LlaisysQwen2Model::ensure_cache_capacity(std::size_t required_capacity) {
    if (required_capacity > meta.maxseq) {
        throw std::invalid_argument("Qwen2 KV cache exceeds maximum sequence length.");
    }

    if (required_capacity <= cache_capacity) { return; }

    /*
     * 按 2 倍扩容，初始最少分配 64 个 token。
     *
     * 不能直接按照 maxseq=131072 分配，否则该模型的
     * 28 层完整 KV Cache 会占用约 3.5 GiB。
     */
    std::size_t new_capacity = cache_capacity;

    if (new_capacity == 0) { new_capacity = std::min<std::size_t>(meta.maxseq, 64); }

    while (new_capacity < required_capacity) {
        if (new_capacity >= meta.maxseq) { break; }

        if (new_capacity > meta.maxseq / 2) {
            new_capacity = meta.maxseq;
        } else {
            new_capacity *= 2;
        }
    }

    if (new_capacity < required_capacity) {
        throw std::runtime_error("Qwen2 failed to grow KV cache.");
    }

    for (std::size_t layer = 0; layer < meta.nlayer; ++layer) {
        auto new_key_cache = create_tensor(*this, {new_capacity, meta.nkvh, meta.dh}, meta.dtype);

        auto new_value_cache = create_tensor(*this, {new_capacity, meta.nkvh, meta.dh}, meta.dtype);

        /*
         * 扩容时复制原来的有效缓存。
         *
         * slice() 只创建视图，不复制数据；
         * rearrange() 才真正执行 CPU/GPU 数据复制。
         */
        if (cache_length > 0 && key_cache[layer] != nullptr && value_cache[layer] != nullptr) {
            auto old_key_prefix = key_cache[layer]->slice(0, 0, cache_length);

            auto old_value_prefix = value_cache[layer]->slice(0, 0, cache_length);

            auto new_key_prefix = new_key_cache->slice(0, 0, cache_length);

            auto new_value_prefix = new_value_cache->slice(0, 0, cache_length);

            llaisys::ops::rearrange(new_key_prefix, old_key_prefix);

            llaisys::ops::rearrange(new_value_prefix, old_value_prefix);
        }

        key_cache[layer] = std::move(new_key_cache);

        value_cache[layer] = std::move(new_value_cache);
    }

    std::cout << "Qwen2 KV cache capacity: " << cache_capacity << " -> " << new_capacity
              << std::endl;

    cache_capacity = new_capacity;
}

std::int64_t LlaisysQwen2Model::infer(const std::int64_t *token_ids, std::size_t ntoken) {
    if (token_ids == nullptr) {
        throw std::invalid_argument("Qwen2 inference token_ids must not be null.");
    }

    if (ntoken == 0) {
        throw std::invalid_argument("Qwen2 inference requires at least one token.");
    }

    if (cache_length > meta.maxseq) {
        throw std::runtime_error("Qwen2 KV cache length is invalid.");
    }

    if (ntoken > meta.maxseq - cache_length) {
        throw std::invalid_argument(
            "Qwen2 input and KV cache exceed "
            "maximum sequence length.");
    }

    validate_global_weights(*this);
    validate_layer_weight_arrays(*this);

    const std::size_t sequence_length = ntoken;

    const std::size_t previous_cache_length = cache_length;

    const std::size_t total_length = previous_cache_length + sequence_length;

    ensure_cache_capacity(total_length);

    const std::size_t hidden_size = meta.hs;

    const std::size_t attention_heads = meta.nh;

    const std::size_t kv_heads = meta.nkvh;

    const std::size_t head_dimension = meta.dh;

    const std::size_t intermediate_size = meta.di;

    const std::size_t vocabulary_size = meta.voc;

    const llaisysDataType_t dtype = meta.dtype;

    /*
     * Input token IDs
     * Shape: [sequence_length]
     */
    auto input_ids = create_tensor(*this, {sequence_length}, LLAISYS_DTYPE_I64);

    input_ids->load(token_ids);

    /*
     * Position IDs
     * Shape: [sequence_length]
     */
    std::vector<std::int64_t> position_values(sequence_length);

    for (std::size_t index = 0; index < sequence_length; ++index) {
        position_values[index] = static_cast<std::int64_t>(previous_cache_length + index);
    }

    auto position_ids = create_tensor(*this, {sequence_length}, LLAISYS_DTYPE_I64);

    position_ids->load(position_values.data());

    /*
     * Embedding
     * [S] × [V, H] -> [S, H]
     */
    auto hidden_states = create_tensor(*this, {sequence_length, hidden_size}, dtype);

    llaisys::ops::embedding(
        hidden_states, input_ids, require_tensor(weights.in_embed, "model.embed_tokens.weight"));

    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));

    for (std::size_t layer = 0; layer < meta.nlayer; ++layer) {
        const std::string layer_prefix = "model.layers." + std::to_string(layer) + ".";

        auto attention_norm_weight
            = require_tensor(weights.attn_norm_w[layer], layer_prefix + "input_layernorm.weight");

        auto query_weight
            = require_tensor(weights.attn_q_w[layer], layer_prefix + "self_attn.q_proj.weight");

        auto query_bias
            = require_tensor(weights.attn_q_b[layer], layer_prefix + "self_attn.q_proj.bias");

        auto key_weight
            = require_tensor(weights.attn_k_w[layer], layer_prefix + "self_attn.k_proj.weight");

        auto key_bias
            = require_tensor(weights.attn_k_b[layer], layer_prefix + "self_attn.k_proj.bias");

        auto value_weight
            = require_tensor(weights.attn_v_w[layer], layer_prefix + "self_attn.v_proj.weight");

        auto value_bias
            = require_tensor(weights.attn_v_b[layer], layer_prefix + "self_attn.v_proj.bias");

        auto output_weight
            = require_tensor(weights.attn_o_w[layer], layer_prefix + "self_attn.o_proj.weight");

        auto mlp_norm_weight = require_tensor(
            weights.mlp_norm_w[layer], layer_prefix + "post_attention_layernorm.weight");

        auto gate_weight
            = require_tensor(weights.mlp_gate_w[layer], layer_prefix + "mlp.gate_proj.weight");

        auto up_weight
            = require_tensor(weights.mlp_up_w[layer], layer_prefix + "mlp.up_proj.weight");

        auto down_weight
            = require_tensor(weights.mlp_down_w[layer], layer_prefix + "mlp.down_proj.weight");

        /*
         * Attention RMSNorm
         */
        auto normalized_hidden = create_tensor(*this, {sequence_length, hidden_size}, dtype);

        llaisys::ops::rms_norm(
            normalized_hidden, hidden_states, attention_norm_weight, meta.epsilon);

        /*
         * Q projection
         * [S, H] -> [S, nh * dh]
         */
        auto query_2d
            = create_tensor(*this, {sequence_length, attention_heads * head_dimension}, dtype);

        llaisys::ops::linear(query_2d, normalized_hidden, query_weight, query_bias);

        /*
         * K projection
         * [S, H] -> [S, nkvh * dh]
         */
        auto key_2d = create_tensor(*this, {sequence_length, kv_heads * head_dimension}, dtype);

        llaisys::ops::linear(key_2d, normalized_hidden, key_weight, key_bias);

        /*
         * V projection
         * [S, H] -> [S, nkvh * dh]
         */
        auto value_2d = create_tensor(*this, {sequence_length, kv_heads * head_dimension}, dtype);

        llaisys::ops::linear(value_2d, normalized_hidden, value_weight, value_bias);

        auto query_3d = query_2d->view({sequence_length, attention_heads, head_dimension});

        auto key_3d = key_2d->view({sequence_length, kv_heads, head_dimension});

        auto value_3d = value_2d->view({sequence_length, kv_heads, head_dimension});

        /*
         * RoPE
         */
        auto rotated_query
            = create_tensor(*this, {sequence_length, attention_heads, head_dimension}, dtype);

        auto rotated_key = create_tensor(*this, {sequence_length, kv_heads, head_dimension}, dtype);

        llaisys::ops::rope(rotated_query, query_3d, position_ids, meta.theta);

        llaisys::ops::rope(rotated_key, key_3d, position_ids, meta.theta);

        /*
         * 把当前调用产生的新 K/V 写入缓存：
         *
         * [previous_cache_length, total_length)
         */
        auto key_cache_destination
            = key_cache[layer]->slice(0, previous_cache_length, total_length);

        auto value_cache_destination
            = value_cache[layer]->slice(0, previous_cache_length, total_length);

        llaisys::ops::rearrange(key_cache_destination, rotated_key);

        llaisys::ops::rearrange(value_cache_destination, value_3d);

        /*
         * Attention 读取从第 0 个 token 到当前 token 的
         * 全部有效缓存。
         */
        auto key_history = key_cache[layer]->slice(0, 0, total_length);

        auto value_history = value_cache[layer]->slice(0, 0, total_length);

        /*
         * Causal grouped-query attention.
         */
        auto attention_output_3d
            = create_tensor(*this, {sequence_length, attention_heads, head_dimension}, dtype);

        llaisys::ops::self_attention(
            attention_output_3d, rotated_query, key_history, value_history, attention_scale);

        auto attention_output_2d = attention_output_3d->view({sequence_length, hidden_size});

        /*
         * Attention output projection.
         * No bias in Qwen2 o_proj.
         */
        auto projected_attention = create_tensor(*this, {sequence_length, hidden_size}, dtype);

        llaisys::ops::linear(projected_attention, attention_output_2d, output_weight, nullptr);

        /*
         * First residual connection.
         */
        auto attention_residual = create_tensor(*this, {sequence_length, hidden_size}, dtype);

        llaisys::ops::add(attention_residual, hidden_states, projected_attention);

        /*
         * MLP RMSNorm.
         */
        auto normalized_mlp = create_tensor(*this, {sequence_length, hidden_size}, dtype);

        llaisys::ops::rms_norm(normalized_mlp, attention_residual, mlp_norm_weight, meta.epsilon);

        /*
         * MLP gate/up projections.
         */
        auto gate = create_tensor(*this, {sequence_length, intermediate_size}, dtype);

        auto up = create_tensor(*this, {sequence_length, intermediate_size}, dtype);

        llaisys::ops::linear(gate, normalized_mlp, gate_weight, nullptr);

        llaisys::ops::linear(up, normalized_mlp, up_weight, nullptr);

        auto activated = create_tensor(*this, {sequence_length, intermediate_size}, dtype);

        llaisys::ops::swiglu(activated, gate, up);

        /*
         * MLP down projection.
         */
        auto mlp_output = create_tensor(*this, {sequence_length, hidden_size}, dtype);

        llaisys::ops::linear(mlp_output, activated, down_weight, nullptr);

        /*
         * Second residual connection.
         */
        auto next_hidden_states = create_tensor(*this, {sequence_length, hidden_size}, dtype);

        llaisys::ops::add(next_hidden_states, attention_residual, mlp_output);

        hidden_states = std::move(next_hidden_states);
    }

    /*
     * 所有层都成功写入缓存后，才正式提交新的缓存长度。
     *
     * 如果中间某一层抛出异常，cache_length 仍保持旧值，
     * 下一次可以重新覆盖同一段缓存。
     */
    cache_length = total_length;

    /*
     * Only the last position is needed for next-token prediction.
     * Shape: [1, H]
     */
    auto last_hidden = hidden_states->slice(0, sequence_length - 1, sequence_length);

    auto normalized_last_hidden = create_tensor(*this, {1, hidden_size}, dtype);

    llaisys::ops::rms_norm(
        normalized_last_hidden, last_hidden,
        require_tensor(weights.out_norm_w, "model.norm.weight"), meta.epsilon);

    /*
     * LM head
     * [1, H] × [V, H]^T -> [1, V]
     */
    auto logits_2d = create_tensor(*this, {1, vocabulary_size}, dtype);

    llaisys::ops::linear(
        logits_2d, normalized_last_hidden, require_tensor(weights.out_embed, "lm_head.weight"),
        nullptr);

    auto logits = logits_2d->view({vocabulary_size});

    auto max_index = create_tensor(*this, {1}, LLAISYS_DTYPE_I64);

    auto max_value = create_tensor(*this, {1}, dtype);

    llaisys::ops::argmax(max_index, max_value, logits);

    /*
     * Copy argmax index to CPU before returning.
     */
    auto host_index = max_index->to(LLAISYS_DEVICE_CPU, 0);

    std::int64_t next_token = -1;

    std::memcpy(&next_token, host_index->data(), sizeof(next_token));

    return next_token;
}
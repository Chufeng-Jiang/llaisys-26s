#include "qwen2_weights.hpp"

#include "../llaisys/llaisys_tensor.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace {

llaisys::tensor_t require_tensor(llaisysTensor_t handle, const std::string &name) {
    if (handle == nullptr) { throw std::runtime_error("Qwen2 weight handle is null: " + name); }

    if (handle->tensor == nullptr) {
        throw std::runtime_error("Qwen2 internal tensor is null: " + name);
    }

    return handle->tensor;
}

using WeightArray = std::unique_ptr<llaisysTensor_t[]>;

WeightArray make_weight_array(std::size_t nlayer) {
    return WeightArray(new llaisysTensor_t[nlayer]{});
}

} // namespace

void qwen2_allocate_weight_arrays(LlaisysQwen2Weights &weights, std::size_t nlayer) {
    // Build every array under temporary RAII ownership first.
    // If any allocation throws, all earlier arrays are released automatically.
    auto attn_norm_w = make_weight_array(nlayer);

    auto attn_q_w = make_weight_array(nlayer);
    auto attn_q_b = make_weight_array(nlayer);

    auto attn_k_w = make_weight_array(nlayer);
    auto attn_k_b = make_weight_array(nlayer);

    auto attn_v_w = make_weight_array(nlayer);
    auto attn_v_b = make_weight_array(nlayer);

    auto attn_o_w = make_weight_array(nlayer);

    auto mlp_norm_w = make_weight_array(nlayer);
    auto mlp_gate_w = make_weight_array(nlayer);
    auto mlp_up_w = make_weight_array(nlayer);
    auto mlp_down_w = make_weight_array(nlayer);

    weights.attn_norm_w = attn_norm_w.release();

    weights.attn_q_w = attn_q_w.release();
    weights.attn_q_b = attn_q_b.release();

    weights.attn_k_w = attn_k_w.release();
    weights.attn_k_b = attn_k_b.release();

    weights.attn_v_w = attn_v_w.release();
    weights.attn_v_b = attn_v_b.release();

    weights.attn_o_w = attn_o_w.release();

    weights.mlp_norm_w = mlp_norm_w.release();
    weights.mlp_gate_w = mlp_gate_w.release();
    weights.mlp_up_w = mlp_up_w.release();
    weights.mlp_down_w = mlp_down_w.release();
}

void qwen2_release_weight_arrays(LlaisysQwen2Weights &weights) noexcept {
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

    // The underlying weight Tensor handles are populated and owned by
    // the existing Python/model loading layer. The model only stores
    // non-owning references to those handles.
    weights.in_embed = nullptr;
    weights.out_embed = nullptr;
    weights.out_norm_w = nullptr;
}

void qwen2_validate_layer_weight_arrays(const LlaisysQwen2Model &model) {
    const auto &weights = model.weights;

    if (weights.attn_norm_w == nullptr || weights.attn_q_w == nullptr || weights.attn_q_b == nullptr
        || weights.attn_k_w == nullptr || weights.attn_k_b == nullptr || weights.attn_v_w == nullptr
        || weights.attn_v_b == nullptr || weights.attn_o_w == nullptr
        || weights.mlp_norm_w == nullptr || weights.mlp_gate_w == nullptr
        || weights.mlp_up_w == nullptr || weights.mlp_down_w == nullptr) {
        throw std::runtime_error("Qwen2 layer-weight arrays are not initialized.");
    }
}

Qwen2GlobalWeights qwen2_global_weights(const LlaisysQwen2Model &model) {
    return Qwen2GlobalWeights{
        require_tensor(model.weights.in_embed, "model.embed_tokens.weight"),
        require_tensor(model.weights.out_embed, "lm_head.weight"),
        require_tensor(model.weights.out_norm_w, "model.norm.weight"),
    };
}

Qwen2LayerWeights qwen2_layer_weights(const LlaisysQwen2Model &model, std::size_t layer) {
    if (layer >= model.meta.nlayer) {
        throw std::out_of_range("Qwen2 layer index is out of range.");
    }

    qwen2_validate_layer_weight_arrays(model);

    const std::string prefix = "model.layers." + std::to_string(layer) + ".";

    const auto &weights = model.weights;

    return Qwen2LayerWeights{
        require_tensor(weights.attn_norm_w[layer], prefix + "input_layernorm.weight"),

        require_tensor(weights.attn_q_w[layer], prefix + "self_attn.q_proj.weight"),
        require_tensor(weights.attn_q_b[layer], prefix + "self_attn.q_proj.bias"),

        require_tensor(weights.attn_k_w[layer], prefix + "self_attn.k_proj.weight"),
        require_tensor(weights.attn_k_b[layer], prefix + "self_attn.k_proj.bias"),

        require_tensor(weights.attn_v_w[layer], prefix + "self_attn.v_proj.weight"),
        require_tensor(weights.attn_v_b[layer], prefix + "self_attn.v_proj.bias"),

        require_tensor(weights.attn_o_w[layer], prefix + "self_attn.o_proj.weight"),

        require_tensor(weights.mlp_norm_w[layer], prefix + "post_attention_layernorm.weight"),
        require_tensor(weights.mlp_gate_w[layer], prefix + "mlp.gate_proj.weight"),
        require_tensor(weights.mlp_up_w[layer], prefix + "mlp.up_proj.weight"),
        require_tensor(weights.mlp_down_w[layer], prefix + "mlp.down_proj.weight"),
    };
}

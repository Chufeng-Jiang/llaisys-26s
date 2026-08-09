#include "qwen2_validation.hpp"

#include <stdexcept>

void qwen2_validate_model_configuration(
    const LlaisysQwen2Meta &meta, const int *device_ids, int ndevice) {
    if (ndevice <= 0) { throw std::invalid_argument("Qwen2: ndevice must be greater than zero."); }

    if (device_ids == nullptr) {
        throw std::invalid_argument("Qwen2: device_ids must not be null.");
    }

    // The current inference implementation is intentionally single-device.
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
}

void qwen2_validate_inference_request(
    const LlaisysQwen2Model &model, const std::int64_t *token_ids, std::size_t ntoken) {
    if (token_ids == nullptr) {
        throw std::invalid_argument("Qwen2 inference token_ids must not be null.");
    }

    if (ntoken == 0) {
        throw std::invalid_argument("Qwen2 inference requires at least one token.");
    }

    if (model.cache_length > model.meta.maxseq) {
        throw std::runtime_error("Qwen2 KV cache length is invalid.");
    }

    if (ntoken > model.meta.maxseq - model.cache_length) {
        throw std::invalid_argument(
            "Qwen2 input and KV cache exceed "
            "maximum sequence length.");
    }
}

#pragma once

#include "llaisys/models/qwen2.h"

#include "../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class Qwen2Workspace;

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};

    llaisysDeviceType_t device{};
    std::vector<int> device_ids;

    // Persistent KV cache. These tensors intentionally survive across infer()
    // calls until the model is destroyed.
    std::vector<llaisys::tensor_t> key_cache;
    std::vector<llaisys::tensor_t> value_cache;

    std::size_t cache_length{0};
    std::size_t cache_capacity{0};

    // Reusable scratch buffers for non-persistent intermediate tensors.
    // The workspace owns device storage; individual infer() calls only create
    // lightweight Tensor views on top of that storage.
    std::unique_ptr<Qwen2Workspace> workspace;

    LlaisysQwen2Model(
        const LlaisysQwen2Meta &model_meta,
        llaisysDeviceType_t model_device,
        const int *model_device_ids,
        int ndevice);

    ~LlaisysQwen2Model();

    LlaisysQwen2Model(const LlaisysQwen2Model &) = delete;
    LlaisysQwen2Model &operator=(const LlaisysQwen2Model &) = delete;

    void reset_cache();

    void ensure_cache_capacity(std::size_t required_capacity);

    std::int64_t infer(const std::int64_t *token_ids, std::size_t ntoken);
};

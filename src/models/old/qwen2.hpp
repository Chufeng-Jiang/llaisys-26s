#pragma once

#include "llaisys/models/qwen2.h"

#include "../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};

    llaisysDeviceType_t device{LLAISYS_DEVICE_CPU};

    std::vector<int> device_ids;

    LlaisysQwen2Weights weights{};

    // 每层分别保存：
    // [cache_capacity, nkvh, dh]
    std::vector<llaisys::tensor_t> key_cache;
    std::vector<llaisys::tensor_t> value_cache;

    std::size_t cache_length{0};
    std::size_t cache_capacity{0};

    LlaisysQwen2Model(
        const LlaisysQwen2Meta &model_meta,
        llaisysDeviceType_t model_device,
        const int *model_device_ids,
        int ndevice);

    ~LlaisysQwen2Model();

    void reset_cache();

    void ensure_cache_capacity(std::size_t required_capacity);

    std::int64_t infer(const std::int64_t *token_ids, std::size_t ntoken);

    LlaisysQwen2Model(const LlaisysQwen2Model &) = delete;

    LlaisysQwen2Model &operator=(const LlaisysQwen2Model &) = delete;
};
#include "qwen2.hpp"

#include "qwen2_common.hpp"

#include "../ops/rearrange/op.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

void LlaisysQwen2Model::reset_cache() {
    // Keep allocated KV storage so the next generation can reuse it.
    cache_length = 0;
}

void LlaisysQwen2Model::ensure_cache_capacity(std::size_t required_capacity) {
    if (required_capacity > meta.maxseq) {
        throw std::invalid_argument("Qwen2 KV cache exceeds maximum sequence length.");
    }

    if (required_capacity <= cache_capacity) { return; }

    // Grow geometrically, but do not eagerly allocate meta.maxseq.
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
        auto new_key_cache
            = qwen2_create_tensor(*this, {new_capacity, meta.nkvh, meta.dh}, meta.dtype);

        auto new_value_cache
            = qwen2_create_tensor(*this, {new_capacity, meta.nkvh, meta.dh}, meta.dtype);

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

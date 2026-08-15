#pragma once

#include "../config.hpp"

#include <cstddef>

namespace llaisys::ops::embedding_config {

inline constexpr unsigned int DEFAULT_MAX_BLOCKS = 256;
inline constexpr bool DEFAULT_VECTORIZED = true;

struct LaunchPolicy {
    std::size_t block_size;
    std::size_t max_blocks;
    bool allow_vectorized;
};

inline const LaunchPolicy &get_launch_policy() {
    static const LaunchPolicy policy{
        static_cast<std::size_t>(config::block_size()),

        static_cast<std::size_t>(
            config::get_unsigned("LLAISYS_EMBEDDING_MAX_BLOCKS", DEFAULT_MAX_BLOCKS, 1, 4096)),

        config::get_bool("LLAISYS_EMBEDDING_VECTORIZED", DEFAULT_VECTORIZED),
    };

    return policy;
}

} // namespace llaisys::ops::embedding_config
#pragma once

#include "../config.hpp"

#include <cstddef>

namespace llaisys::ops::add_config {

inline constexpr unsigned int DEFAULT_MAX_BLOCKS = 256;
inline constexpr unsigned int DEFAULT_ENABLE_VECTORIZED = 1;

struct LaunchPolicy {
    std::size_t block_size;
    std::size_t max_blocks;
    bool enable_vectorized;
};

inline const LaunchPolicy &get_launch_policy() {
    static const LaunchPolicy policy{
        static_cast<std::size_t>(config::block_size()),
        static_cast<std::size_t>(
            config::get_unsigned(
                "LLAISYS_ADD_MAX_BLOCKS",
                DEFAULT_MAX_BLOCKS,
                1,
                4096)),
        config::get_unsigned(
            "LLAISYS_ADD_ENABLE_VECTORIZED",
            DEFAULT_ENABLE_VECTORIZED,
            0,
            1)
            != 0,
    };

    return policy;
}

} // namespace llaisys::ops::add_config
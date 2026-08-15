#pragma once

#include "../config.hpp"

#include <cstddef>

namespace llaisys::ops::add_config {

inline constexpr unsigned int DEFAULT_MAX_BLOCKS = 256;

struct LaunchPolicy {
    std::size_t block_size;
    std::size_t max_blocks;
};

inline const LaunchPolicy &get_launch_policy() {
    static const LaunchPolicy policy{
        static_cast<std::size_t>(config::block_size()),
        static_cast<std::size_t>(
            config::get_unsigned("LLAISYS_ADD_MAX_BLOCKS", DEFAULT_MAX_BLOCKS, 1, 4096)),
    };

    return policy;
}

} // namespace llaisys::ops::add_config
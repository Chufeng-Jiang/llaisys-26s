#pragma once

#include "../config.hpp"
#include "../cuda/common.cuh"

#include <cstddef>

namespace llaisys::ops::argmax_config {

inline constexpr unsigned int DEFAULT_MAX_BLOCKS = 256;

struct LaunchConfig {
    unsigned int block_size;
    unsigned int grid_size;
};

inline unsigned int get_max_blocks() {
    return config::get_unsigned("LLAISYS_ARGMAX_MAX_BLOCKS", DEFAULT_MAX_BLOCKS, 1, 4096);
}

inline LaunchConfig get_launch_config(std::size_t numel) {
    const unsigned int block_size = config::block_size();
    const unsigned int max_blocks = get_max_blocks();

    const std::size_t grid_size = cuda::get_capped_grid_size(
        numel, static_cast<std::size_t>(block_size), static_cast<std::size_t>(max_blocks));

    return LaunchConfig{
        block_size,
        static_cast<unsigned int>(grid_size),
    };
}

} // namespace llaisys::ops::argmax_config
#pragma once

#include "../config.hpp"

#include <stdexcept>

namespace llaisys::ops::linear_config {

inline unsigned int get_tile_size() {
    const unsigned int block_size = config::block_size();

    switch (block_size) {
    case 64:
        return 8;

    case 256:
        return 16;

    default:
        throw std::invalid_argument(
            "Linear: LLAISYS_BLOCK_SIZE must be 64 or 256 "
            "for the square tiled Linear kernel.");
    }
}

} // namespace llaisys::ops::linear_config
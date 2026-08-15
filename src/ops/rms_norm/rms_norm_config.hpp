#pragma once

#include "../config.hpp"

#include <cstddef>
#include <stdexcept>

namespace llaisys::ops::rms_norm_config {

// ============================================================
// Cross-GPU shared RMSNorm launch policy
// ============================================================
//
// RMSNorm uses the global controlled-experiment variable:
//
//     LLAISYS_BLOCK_SIZE
//
// NVIDIA and MetaX therefore use exactly the same thread count
// for the shared-source baseline.
//
// The shared tree reduction currently supports the block sizes
// that were already exercised by both original backends.
// ============================================================

inline unsigned int block_size() {
    const unsigned int value = config::block_size();

    switch (value) {
    case 64:
    case 128:
    case 256:
        return value;

    default:
        throw std::invalid_argument(
            "RMSNorm: LLAISYS_BLOCK_SIZE must be 64, 128, or 256 "
            "for the shared tree-reduction kernel.");
    }
}

inline constexpr std::size_t MAX_BLOCKS = 65535;

} // namespace llaisys::ops::rms_norm_config

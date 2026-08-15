#pragma once

#include "../config.hpp"

#include <cstddef>

namespace llaisys::ops::rearrange_config {

// ============================================================
// Cross-GPU shared Rearrange launch policy
// ============================================================
//
// Thread-block size is a global controlled-experiment variable:
//
//     LLAISYS_BLOCK_SIZE
//
// NVIDIA and MetaX must use the same value for the shared-source
// cross-GPU baseline.
//
// The grid cap is intentionally the same constant on both backends.
// It is not exposed as a vendor- or operator-specific environment
// variable in the controlled baseline.
// ============================================================

inline constexpr std::size_t MAX_BLOCKS = 65535;

inline std::size_t block_size() { return static_cast<std::size_t>(config::block_size()); }

} // namespace llaisys::ops::rearrange_config

#pragma once

#include "../config.hpp"

#include <cstddef>

namespace llaisys::ops::self_attention_config {

// ============================================================
// Cross-GPU shared Self-Attention launch policy
// ============================================================
//
// The controlled baseline uses the global launch variable:
//
//     LLAISYS_BLOCK_SIZE
//
// NVIDIA and MetaX therefore use the same number of threads per
// block for the exact same shared-source attention kernel.
//
// The grid cap is also identical on both backends.
// ============================================================

inline std::size_t block_size() { return static_cast<std::size_t>(config::block_size()); }

inline constexpr std::size_t MAX_BLOCKS = 65535;

} // namespace llaisys::ops::self_attention_config

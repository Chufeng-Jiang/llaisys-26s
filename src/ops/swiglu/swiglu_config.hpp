#pragma once

#include "../config.hpp"

#include <cstddef>

namespace llaisys::ops::swiglu_config {

// ============================================================
// Cross-GPU shared SwiGLU launch policy
// ============================================================
//
// The controlled baseline uses the global shared block size:
//
//     LLAISYS_BLOCK_SIZE
//
// NVIDIA and MetaX therefore launch the exact same SwiGLU
// implementation with the same number of threads per block.
//
// Grid capping is also identical on both backends.
// ============================================================

inline std::size_t block_size() { return static_cast<std::size_t>(config::block_size()); }

inline constexpr std::size_t MAX_BLOCKS = 65535;

} // namespace llaisys::ops::swiglu_config

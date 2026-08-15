#pragma once

#include "../config.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace llaisys::ops::rope_config {

// ============================================================
// Cross-GPU RoPE policy
// ============================================================
//
// Thread-block size is shared globally:
//
//     LLAISYS_BLOCK_SIZE
//
// RoPE implementation selection is an operator-specific mechanism:
//
//     LLAISYS_ROPE_IMPL=direct
//     LLAISYS_ROPE_IMPL=cached
//
// There is deliberately no "auto" mode in the controlled baseline.
// An automatic cached/direct threshold would reintroduce a
// platform/shape-dependent policy into the primary comparison.
//
// Default:
//     direct
// ============================================================

enum class Implementation {
    DIRECT,
    CACHED,
};

inline std::size_t block_size() { return static_cast<std::size_t>(config::block_size()); }

inline Implementation implementation() {
    const char *value = std::getenv("LLAISYS_ROPE_IMPL");

    if (value == nullptr || std::strcmp(value, "direct") == 0) { return Implementation::DIRECT; }

    if (std::strcmp(value, "cached") == 0) { return Implementation::CACHED; }

    throw std::invalid_argument("LLAISYS_ROPE_IMPL must be direct or cached.");
}

inline constexpr std::size_t MAX_BLOCKS = 65535;

} // namespace llaisys::ops::rope_config

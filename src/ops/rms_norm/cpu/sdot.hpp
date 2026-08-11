#pragma once

#include <cstddef>

namespace llaisys::ops::cpu {

// Compute the FP32 dot product x[0:N] · y[0:N].
//
// The implementation performs runtime SIMD dispatch on supported x86
// compilers and falls back to a portable scalar implementation elsewhere.
float sdot(const float *x, const float *y, std::size_t count);

} // namespace llaisys::ops::cpu

#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

// Apply non-interleaved rotary position embedding to a contiguous tensor:
//
//     input/output shape: [seqlen, nhead, d]
//     position ids shape: [seqlen]
//
// The first and second halves of the final dimension are rotated together.
// F32, F16, and BF16 are supported; all arithmetic is performed in FP32.
// The CUDA work is submitted to the supplied LLAISYS Runtime stream.
void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    float theta,
    llaisysDataType_t type,
    std::size_t seqlen,
    std::size_t nhead,
    std::size_t d,
    llaisysStream_t stream);

} // namespace llaisys::ops::nvidia

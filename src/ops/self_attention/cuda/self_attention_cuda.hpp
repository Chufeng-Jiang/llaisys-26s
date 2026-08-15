#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {

void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    float scale,
    llaisysDataType_t type,
    std::size_t seqlen,
    std::size_t nhead,
    std::size_t dv,
    std::size_t total_len,
    std::size_t nkvhead,
    std::size_t d,
    llaisysStream_t stream);

} // namespace llaisys::ops::cuda

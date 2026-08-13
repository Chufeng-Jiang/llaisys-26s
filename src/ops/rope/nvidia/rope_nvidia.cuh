#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

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
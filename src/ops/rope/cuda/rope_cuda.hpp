#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    float theta,
    llaisysDataType_t type,
    std::size_t sequence_length,
    std::size_t head_count,
    std::size_t head_dimension,
    llaisysStream_t stream);

} // namespace llaisys::ops::cuda

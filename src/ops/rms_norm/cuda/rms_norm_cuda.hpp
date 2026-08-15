#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    float eps,
    llaisysDataType_t type,
    std::size_t row_count,
    std::size_t column_count,
    llaisysStream_t stream);

} // namespace llaisys::ops::cuda

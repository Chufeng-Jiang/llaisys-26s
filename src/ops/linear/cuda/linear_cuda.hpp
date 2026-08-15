#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    std::size_t row_count,
    std::size_t output_features,
    std::size_t input_features,
    llaisysStream_t stream);

} // namespace llaisys::ops::cuda

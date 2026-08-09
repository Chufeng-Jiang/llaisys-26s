#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    std::size_t nrow,
    std::size_t ncol_out,
    std::size_t ncol_in,
    llaisysStream_t stream);

} // namespace llaisys::ops::nvidia

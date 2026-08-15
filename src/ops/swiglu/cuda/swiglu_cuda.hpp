#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    std::size_t numel,
    llaisysStream_t stream);

} // namespace llaisys::ops::cuda

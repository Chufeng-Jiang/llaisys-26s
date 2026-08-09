#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    float eps,
    llaisysDataType_t type,
    std::size_t nrow,
    std::size_t ncol);

} // namespace llaisys::ops::cpu

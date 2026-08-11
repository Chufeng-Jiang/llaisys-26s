#pragma once

#include <cstddef>

#include "llaisys.h"

namespace llaisys::ops::cpu {

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    std::size_t numel);

} // namespace llaisys::ops::cpu
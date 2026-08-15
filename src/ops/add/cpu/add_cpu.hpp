#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    std::size_t numel);

} // namespace llaisys::ops::cpu
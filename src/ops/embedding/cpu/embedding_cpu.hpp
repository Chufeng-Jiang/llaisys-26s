#pragma once

#include <cstddef>

#include "llaisys.h"

namespace llaisys::ops::cpu {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    std::size_t numel,
    std::size_t len,
    std::size_t vocabulary_size);

} // namespace llaisys::ops::cpu
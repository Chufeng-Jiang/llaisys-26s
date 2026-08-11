#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    std::size_t numel,
    std::size_t len,
    std::size_t vocabulary_size,
    llaisysStream_t stream);

} // namespace llaisys::ops::nvidia
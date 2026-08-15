#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cuda {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    std::size_t numel,
    std::size_t embedding_length,
    std::size_t vocabulary_size,
    llaisysStream_t stream);

} // namespace llaisys::ops::cuda

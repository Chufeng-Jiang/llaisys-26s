#pragma once

#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::metax {

void rearrange(
    std::byte *out,
    const std::byte *in,
    llaisysDataType_t type,
    std::size_t numel,
    const std::vector<std::size_t> &out_shape,
    const std::vector<std::ptrdiff_t> &out_strides,
    const std::vector<std::size_t> &in_shape,
    const std::vector<std::ptrdiff_t> &in_strides,
    llaisysStream_t stream);

} // namespace llaisys::ops::metax
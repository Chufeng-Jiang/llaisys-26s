#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

enum class RopeImplementation {
    AUTO,
    DIRECT,
    CACHED,
};

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
} // namespace llaisys::ops

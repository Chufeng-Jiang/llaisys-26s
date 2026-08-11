#pragma once

#include "qwen2.hpp"

#include <cstddef>
#include <vector>

llaisys::tensor_t qwen2_create_tensor(
    const LlaisysQwen2Model &model, const std::vector<std::size_t> &shape, llaisysDataType_t dtype);

void qwen2_copy_to_host(void *destination, const llaisys::tensor_t &source, std::size_t nbytes);

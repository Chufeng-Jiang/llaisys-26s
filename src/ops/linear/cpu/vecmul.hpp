#pragma once

#include <cstddef>

// c = a * B^T + c, all tensors are row-major.
//
// c: [N]
// a: [K]
// B: [N, K]
void vecmul(const float *a, const float *B, float *c, std::size_t N, std::size_t K);

#pragma once

#include <cstddef>

// C = A * B^T + C, all tensors are row-major.
//
// C: [M, N]
// A: [M, K]
// B: [N, K]
void matmul(const float *A, const float *B, float *C, std::size_t M, std::size_t N, std::size_t K);

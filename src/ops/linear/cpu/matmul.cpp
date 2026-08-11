#include "matmul.hpp"

#include <algorithm>
#include <cstddef>

namespace {

inline float dot_product(const float *a, const float *b, std::size_t count) {
    float sum = 0.0F;

#pragma omp simd reduction(+ : sum)
    for (std::size_t k = 0; k < count; ++k) { sum += a[k] * b[k]; }

    return sum;
}

} // namespace

void matmul(const float *A, const float *B, float *C, std::size_t M, std::size_t N, std::size_t K) {
    // Tiling improves locality while keeping all temporary state on the stack.
    // Unlike the reference implementation, this version has no shared global
    // packing buffers, so simultaneous Linear calls are thread-safe.
    constexpr std::size_t M_BLOCK = 4;
    constexpr std::size_t N_BLOCK = 32;

    const bool use_openmp = M >= 2 && N >= 32 && K >= 64;

#pragma omp parallel for collapse(2) if (use_openmp) schedule(static)
    for (std::size_t row_block = 0; row_block < M; row_block += M_BLOCK) {
        for (std::size_t column_block = 0; column_block < N; column_block += N_BLOCK) {
            const std::size_t row_end = std::min(row_block + M_BLOCK, M);

            const std::size_t column_end = std::min(column_block + N_BLOCK, N);

            for (std::size_t row = row_block; row < row_end; ++row) {
                const float *const input_row = A + row * K;

                float *const output_row = C + row * N;

                for (std::size_t column = column_block; column < column_end; ++column) {
                    const float *const weight_row = B + column * K;

                    output_row[column] += dot_product(input_row, weight_row, K);
                }
            }
        }
    }
}

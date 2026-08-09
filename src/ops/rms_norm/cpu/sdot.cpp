#include "sdot.hpp"

#include <cstddef>

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define LLAISYS_X86_SIMD_DISPATCH 1
#else
#define LLAISYS_X86_SIMD_DISPATCH 0
#endif

namespace {

using SdotFunction = float (*)(const float *, const float *, std::size_t);

float sdot_scalar(const float *x, const float *y, std::size_t count) {
    // Multiple independent accumulators reduce the dependency chain and also
    // give the compiler a simple loop to auto-vectorize on non-x86 platforms.
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    float sum2 = 0.0F;
    float sum3 = 0.0F;

    std::size_t i = 0;

    for (; i + 4 <= count; i += 4) {
        sum0 += x[i] * y[i];
        sum1 += x[i + 1] * y[i + 1];
        sum2 += x[i + 2] * y[i + 2];
        sum3 += x[i + 3] * y[i + 3];
    }

    float sum = (sum0 + sum1) + (sum2 + sum3);

    for (; i < count; ++i) { sum += x[i] * y[i]; }

    return sum;
}

#if LLAISYS_X86_SIMD_DISPATCH

__attribute__((target("avx2,fma"))) float
sdot_avx2(const float *x, const float *y, std::size_t count) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    std::size_t i = 0;

    for (; i + 32 <= count; i += 32) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i), sum0);

        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8), sum1);

        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16), sum2);

        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24), sum3);
    }

    const __m256 sum01 = _mm256_add_ps(sum0, sum1);
    const __m256 sum23 = _mm256_add_ps(sum2, sum3);
    const __m256 vector_sum = _mm256_add_ps(sum01, sum23);

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, vector_sum);

    float sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]) + (lanes[4] + lanes[5])
              + (lanes[6] + lanes[7]);

    for (; i < count; ++i) { sum += x[i] * y[i]; }

    return sum;
}

__attribute__((target("avx512f,fma"))) float
sdot_avx512(const float *x, const float *y, std::size_t count) {
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();

    std::size_t i = 0;

    for (; i + 64 <= count; i += 64) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i), sum0);

        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(y + i + 16), sum1);

        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 32), _mm512_loadu_ps(y + i + 32), sum2);

        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 48), _mm512_loadu_ps(y + i + 48), sum3);
    }

    const __m512 sum01 = _mm512_add_ps(sum0, sum1);
    const __m512 sum23 = _mm512_add_ps(sum2, sum3);
    const __m512 vector_sum = _mm512_add_ps(sum01, sum23);

    float sum = _mm512_reduce_add_ps(vector_sum);

    for (; i < count; ++i) { sum += x[i] * y[i]; }

    return sum;
}

SdotFunction select_sdot_implementation() {
    __builtin_cpu_init();

    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) { return &sdot_avx512; }

    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) { return &sdot_avx2; }

    return &sdot_scalar;
}

#endif

} // namespace

namespace llaisys::ops::cpu {

float sdot(const float *x, const float *y, std::size_t count) {
    if (count == 0) { return 0.0F; }

#if LLAISYS_X86_SIMD_DISPATCH
    // Resolve the best implementation only once per process. The selected
    // function is then reused without repeating CPUID checks on every row.
    static const SdotFunction implementation = select_sdot_implementation();

    return implementation(x, y, count);
#else
    return sdot_scalar(x, y, count);
#endif
}

} // namespace llaisys::ops::cpu

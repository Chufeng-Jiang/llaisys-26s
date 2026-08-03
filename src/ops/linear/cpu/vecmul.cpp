#include "vecmul.hpp"

#include <cstddef>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

inline float dot_product(
	const float *a,
	const float *b,
	std::size_t count
) {
	std::size_t k = 0;
	float sum = 0.0F;

#if defined(__AVX512F__)
	__m512 sum0 = _mm512_setzero_ps();
	__m512 sum1 = _mm512_setzero_ps();
	__m512 sum2 = _mm512_setzero_ps();
	__m512 sum3 = _mm512_setzero_ps();

	for (; k + 63 < count; k += 64) {
		sum0 = _mm512_fmadd_ps(
			_mm512_loadu_ps(a + k),
			_mm512_loadu_ps(b + k),
			sum0
		);
		sum1 = _mm512_fmadd_ps(
			_mm512_loadu_ps(a + k + 16),
			_mm512_loadu_ps(b + k + 16),
			sum1
		);
		sum2 = _mm512_fmadd_ps(
			_mm512_loadu_ps(a + k + 32),
			_mm512_loadu_ps(b + k + 32),
			sum2
		);
		sum3 = _mm512_fmadd_ps(
			_mm512_loadu_ps(a + k + 48),
			_mm512_loadu_ps(b + k + 48),
			sum3
		);
	}

	sum0 = _mm512_add_ps(sum0, sum1);
	sum2 = _mm512_add_ps(sum2, sum3);
	sum = _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum2));

#elif defined(__AVX2__)
	__m256 sum0 = _mm256_setzero_ps();
	__m256 sum1 = _mm256_setzero_ps();
	__m256 sum2 = _mm256_setzero_ps();
	__m256 sum3 = _mm256_setzero_ps();

	for (; k + 31 < count; k += 32) {
#if defined(__FMA__)
		sum0 = _mm256_fmadd_ps(
			_mm256_loadu_ps(a + k),
			_mm256_loadu_ps(b + k),
			sum0
		);
		sum1 = _mm256_fmadd_ps(
			_mm256_loadu_ps(a + k + 8),
			_mm256_loadu_ps(b + k + 8),
			sum1
		);
		sum2 = _mm256_fmadd_ps(
			_mm256_loadu_ps(a + k + 16),
			_mm256_loadu_ps(b + k + 16),
			sum2
		);
		sum3 = _mm256_fmadd_ps(
			_mm256_loadu_ps(a + k + 24),
			_mm256_loadu_ps(b + k + 24),
			sum3
		);
#else
		sum0 = _mm256_add_ps(
			sum0,
			_mm256_mul_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k))
		);
		sum1 = _mm256_add_ps(
			sum1,
			_mm256_mul_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8))
		);
		sum2 = _mm256_add_ps(
			sum2,
			_mm256_mul_ps(_mm256_loadu_ps(a + k + 16), _mm256_loadu_ps(b + k + 16))
		);
		sum3 = _mm256_add_ps(
			sum3,
			_mm256_mul_ps(_mm256_loadu_ps(a + k + 24), _mm256_loadu_ps(b + k + 24))
		);
#endif
	}

	sum0 = _mm256_add_ps(sum0, sum1);
	sum2 = _mm256_add_ps(sum2, sum3);
	sum0 = _mm256_add_ps(sum0, sum2);

	alignas(32) float lanes[8];
	_mm256_store_ps(lanes, sum0);

	for (float value : lanes) {
		sum += value;
	}
#endif

	// The compiler can vectorize the remaining loop when no explicit SIMD
	// implementation is enabled. With AVX enabled, it handles only the tail.
#pragma omp simd reduction(+ : sum)
	for (std::size_t tail = k; tail < count; ++tail) {
		sum += a[tail] * b[tail];
	}

	return sum;
}

} // namespace

void vecmul(
	const float *a,
	const float *B,
	float *c,
	std::size_t N,
	std::size_t K
) {
	// Creating an OpenMP team for a tiny matrix is usually slower than running
	// the loop serially. Parallelize only when both dimensions contain enough
	// work to amortize that overhead.
	const bool use_openmp =
		N >= 8
		&& K >= 256;

#pragma omp parallel for if(use_openmp) schedule(static)
	for (std::size_t row = 0; row < N; ++row) {
		const float *const weight_row =
			B + row * K;

		c[row] += dot_product(
			a,
			weight_row,
			K
		);
	}
}

#include "rope_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

using llaisys::device::nvidia::CUDA_DEFAULT_MAX_GRID_SIZE;
using llaisys::device::nvidia::from_float;
using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::device::nvidia::get_warp_aligned_block_size;
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::to_cuda_stream;

using llaisys::utils::checked_product;

// Cache cosine and sine values in dynamic shared memory when the
// half-dimension is reasonably small.
//
// Shared-memory usage:
//
//     2 * half_dimension * sizeof(float)
//
// At the limit below, this is:
//
//     2 * 2048 * 4 = 16384 bytes
//
// This is conservative for common NVIDIA GPUs.
inline constexpr std::size_t MAX_CACHED_HALF_DIMENSION = 2048;

// ============================================================
// RoPE angle
// ============================================================
//
// Keep the same Float32 evaluation order as the PyTorch reference:
//
//     exponent    = 2 * pair / dimension
//     denominator = theta ^ exponent
//     angle       = position / denominator
//
// Do not rewrite this as:
//
//     position * reciprocal
//     powf(theta, -exponent)
//     exp2f(...)
//
// Although mathematically equivalent, those forms can round differently
// for large position IDs and fail strict Float32 tests.
__device__ __forceinline__ float rope_angle(
	float position,
	std::size_t pair_index,
	std::size_t dimension,
	float theta
) {
	const float exponent =
		2.0F
		* static_cast<float>(pair_index)
		/ static_cast<float>(dimension);

	const float denominator =
		powf(theta, exponent);

	return position / denominator;
}

// ============================================================
// Shared-memory cached kernel
// ============================================================
//
// One block processes one token at a time.
//
// Sine and cosine are calculated once per dimension pair and then reused by
// every attention head for that token. This avoids repeating powf/sincosf
// for every head.
//
// out and in intentionally do not use __restrict__. Exact in-place execution
// (out == in) is supported, and __restrict__ would make that aliasing invalid.
template <typename T>
__global__ void rope_cached_kernel(
	T *out,
	const T *in,
	const std::int64_t *__restrict__ position_ids,
	float theta,
	std::size_t sequence_length,
	std::size_t head_count,
	std::size_t dimension
) {
	extern __shared__ float trigonometric_cache[];

	const std::size_t half_dimension =
		dimension / 2;

	float *cosine_cache =
		trigonometric_cache;

	float *sine_cache =
		trigonometric_cache + half_dimension;

	for (
		std::size_t token =
			static_cast<std::size_t>(blockIdx.x);
		token < sequence_length;
		token += static_cast<std::size_t>(gridDim.x)
	) {
		const float position =
			static_cast<float>(
				position_ids[token]
			);

		// Calculate one sine/cosine pair per RoPE dimension pair.
		for (
			std::size_t pair =
				static_cast<std::size_t>(threadIdx.x);
			pair < half_dimension;
			pair += static_cast<std::size_t>(blockDim.x)
		) {
			const float angle =
				rope_angle(
					position,
					pair,
					dimension,
					theta
				);

			float sine;
			float cosine;

			sincosf(
				angle,
				&sine,
				&cosine
			);

			cosine_cache[pair] =
				cosine;

			sine_cache[pair] =
				sine;
		}

		// All trigonometric values must be ready before any head uses them.
		__syncthreads();

		const std::size_t pair_count =
			head_count * half_dimension;

		const std::size_t token_offset =
			token * head_count * dimension;

		// Flatten [head, pair] to expose enough parallel work.
		for (
			std::size_t index =
				static_cast<std::size_t>(threadIdx.x);
			index < pair_count;
			index += static_cast<std::size_t>(blockDim.x)
		) {
			const std::size_t head =
				index / half_dimension;

			const std::size_t pair =
				index - head * half_dimension;

			const std::size_t vector_offset =
				token_offset + head * dimension;

			const std::size_t low_index =
				vector_offset + pair;

			const std::size_t high_index =
				low_index + half_dimension;

			// Load both elements before writing either result.
			// This preserves exact in-place execution.
			const float low =
				to_float<T>(
					in[low_index]
				);

			const float high =
				to_float<T>(
					in[high_index]
				);

			const float cosine =
				cosine_cache[pair];

			const float sine =
				sine_cache[pair];

			const float rotated_low =
				low * cosine
				- high * sine;

			const float rotated_high =
				high * cosine
				+ low * sine;

			out[low_index] =
				from_float<T>(
					rotated_low
				);

			out[high_index] =
				from_float<T>(
					rotated_high
				);
		}

		// A capped grid allows one block to process multiple tokens.
		// Do not overwrite shared memory until every thread has finished.
		__syncthreads();
	}
}

// ============================================================
// Direct kernel
// ============================================================
//
// Used for:
// - one-head inputs, where shared-memory reuse has little benefit;
// - unusually large head dimensions that exceed the cache limit.
//
// Each thread owns one or more dimension pairs and reuses the computed
// sine/cosine values while iterating over all heads.
template <typename T>
__global__ void rope_direct_kernel(
	T *out,
	const T *in,
	const std::int64_t *__restrict__ position_ids,
	float theta,
	std::size_t sequence_length,
	std::size_t head_count,
	std::size_t dimension
) {
	const std::size_t half_dimension =
		dimension / 2;

	for (
		std::size_t token =
			static_cast<std::size_t>(blockIdx.x);
		token < sequence_length;
		token += static_cast<std::size_t>(gridDim.x)
	) {
		const float position =
			static_cast<float>(
				position_ids[token]
			);

		const std::size_t token_offset =
			token * head_count * dimension;

		for (
			std::size_t pair =
				static_cast<std::size_t>(threadIdx.x);
			pair < half_dimension;
			pair += static_cast<std::size_t>(blockDim.x)
		) {
			const float angle =
				rope_angle(
					position,
					pair,
					dimension,
					theta
				);

			float sine;
			float cosine;

			sincosf(
				angle,
				&sine,
				&cosine
			);

			for (
				std::size_t head = 0;
				head < head_count;
				++head
			) {
				const std::size_t vector_offset =
					token_offset + head * dimension;

				const std::size_t low_index =
					vector_offset + pair;

				const std::size_t high_index =
					low_index + half_dimension;

				const float low =
					to_float<T>(
						in[low_index]
					);

				const float high =
					to_float<T>(
						in[high_index]
					);

				const float rotated_low =
					low * cosine
					- high * sine;

				const float rotated_high =
					high * cosine
					+ low * sine;

				out[low_index] =
					from_float<T>(
						rotated_low
					);

				out[high_index] =
					from_float<T>(
						rotated_high
					);
			}
		}
	}
}

// ============================================================
// Launcher
// ============================================================

template <typename T>
void launch_rope(
	T *out,
	const T *in,
	const std::int64_t *position_ids,
	float theta,
	std::size_t sequence_length,
	std::size_t head_count,
	std::size_t dimension,
	cudaStream_t stream
) {
	const std::size_t vector_count =
		checked_product(
			sequence_length,
			head_count,
			"RoPE: sequence/head count overflows size_t."
		);

	const std::size_t element_count =
		checked_product(
			vector_count,
			dimension,
			"RoPE: tensor element count overflows size_t."
		);

	CHECK_ARGUMENT(
		element_count == 0 || out != nullptr,
		"RoPE: output pointer must not be null."
	);

	CHECK_ARGUMENT(
		element_count == 0 || in != nullptr,
		"RoPE: input pointer must not be null."
	);

	CHECK_ARGUMENT(
		sequence_length == 0 || position_ids != nullptr,
		"RoPE: position-id pointer must not be null."
	);

	CHECK_ARGUMENT(
		sequence_length == 0 || head_count > 0,
		"RoPE: head count must be greater than zero for a nonempty sequence."
	);

	CHECK_ARGUMENT(
		dimension > 0,
		"RoPE: head dimension must be greater than zero."
	);

	CHECK_ARGUMENT(
		dimension % 2 == 0,
		"RoPE: head dimension must be even."
	);

	CHECK_ARGUMENT(
		std::isfinite(theta) && theta > 0.0F,
		"RoPE: theta must be finite and greater than zero."
	);

	// CUDA does not allow a zero-block launch.
	if (element_count == 0) {
		return;
	}

	const std::size_t half_dimension =
		dimension / 2;

	const std::size_t pair_count =
		checked_product(
			head_count,
			half_dimension,
			"RoPE: head/pair count overflows size_t."
		);

	const unsigned int block_size =
		get_warp_aligned_block_size(
			std::max(
				half_dimension,
				pair_count
			)
		);

	const std::size_t grid_size =
		get_capped_grid_size(
			sequence_length,
			1,
			CUDA_DEFAULT_MAX_GRID_SIZE
		);

	const bool use_cached_kernel =
		head_count > 1
		&& half_dimension
			<= MAX_CACHED_HALF_DIMENSION;

	if (use_cached_kernel) {
		const std::size_t shared_memory_bytes =
			2
			* half_dimension
			* sizeof(float);

		rope_cached_kernel<T><<<
			static_cast<unsigned int>(grid_size),
			block_size,
			shared_memory_bytes,
			stream
		>>>(
			out,
			in,
			position_ids,
			theta,
			sequence_length,
			head_count,
			dimension
		);
	} else {
		rope_direct_kernel<T><<<
			static_cast<unsigned int>(grid_size),
			block_size,
			0,
			stream
		>>>(
			out,
			in,
			position_ids,
			theta,
			sequence_length,
			head_count,
			dimension
		);
	}

	// Check launch/configuration errors without synchronizing the device.
	CUDA_CHECK(
		cudaGetLastError()
	);
}

} // namespace

namespace llaisys::ops::nvidia {

// ============================================================
// Public NVIDIA backend
// ============================================================

void rope(
	std::byte *out,
	const std::byte *in,
	const std::byte *pos_ids,
	float theta,
	llaisysDataType_t type,
	std::size_t seqlen,
	std::size_t nhead,
	std::size_t d,
	llaisysStream_t stream
) {

	const cudaStream_t cuda_stream = llaisys::device::nvidia::to_cuda_stream(stream);
	
	return llaisys::device::nvidia::dispatch_cuda_dtype(
		type,
		[&](auto tag) {
			using T = typename decltype(tag)::type;

			return launch_rope<T>(
				reinterpret_cast<T *>(out),
				reinterpret_cast<const T *>(in),
				reinterpret_cast<const std::int64_t *>(pos_ids),
				theta,
				seqlen,
				nhead,
				d,
				cuda_stream
			);
		}
	);
}

} // namespace llaisys::ops::nvidia

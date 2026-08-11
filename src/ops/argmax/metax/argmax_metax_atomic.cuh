#pragma once

#include "../../../device/metax/metax_common.hpp"
#include "../cuda_compat/argmax_cuda_compat.cuh"
#include "argmax_metax_shuffle.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>


namespace llaisys::ops::metax::detail {

namespace cuda_compat =
	llaisys::ops::cuda_compat;

using cuda_compat::ARGMAX_INVALID_INDEX;
using cuda_compat::ArgmaxResult;
using cuda_compat::from_float;
using cuda_compat::invalid_argmax_result;
using cuda_compat::is_better_argmax;
using cuda_compat::is_valid_argmax_result;
using cuda_compat::to_float;
using cuda_compat::update_argmax_result;


// ============================================================
// Packed global result
// ============================================================
//
// High 32 bits:
//     raw FP32 value bits
//
// Low 32 bits:
//     uint32 Argmax index
//
// This mirrors the NVIDIA optimized Argmax representation and
// allows value + index to be replaced together by one 64-bit
// atomicCAS operation.
// ============================================================

static_assert(
	sizeof(unsigned long long) == 8,
	"MetaX atomic Argmax requires 64-bit unsigned long long."
);


__device__ __forceinline__
unsigned long long pack_argmax_result(
	float value,
	std::uint32_t index
) {
	const unsigned int value_bits =
		__float_as_uint(
			value
		);

	return
		(
			static_cast<unsigned long long>(
				value_bits
			)
			<< 32
		)
		| static_cast<unsigned long long>(
			index
		);
}


__device__ __forceinline__
float unpack_argmax_value(
	unsigned long long packed
) {
	const unsigned int value_bits =
		static_cast<unsigned int>(
			packed >> 32
		);

	return
		__uint_as_float(
			value_bits
		);
}


__device__ __forceinline__
std::uint32_t unpack_argmax_index(
	unsigned long long packed
) {
	return
		static_cast<std::uint32_t>(
			packed & 0xFFFFFFFFULL
		);
}


// ============================================================
// Global atomic merge
// ============================================================
//
// Exactly one thread per block submits one block-level result.
// The CAS loop preserves the shared Argmax comparison contract:
//
//     NaN > non-NaN
//     larger numeric value wins
//     ties -> smaller index
// ============================================================

__device__ __forceinline__
void atomic_merge_argmax_result(
	unsigned long long *packed_workspace,
	const ArgmaxResult &candidate
) {
	if (
		!is_valid_argmax_result(
			candidate
		)
	) {
		return;
	}

	// Atomic read without changing a non-zero workspace value.
	// The initialized invalid value is pack(0, UINT32_MAX), so it
	// is not zero and this CAS acts as an atomic load.
	unsigned long long observed =
		atomicCAS(
			packed_workspace,
			0ULL,
			0ULL
		);

	while (true) {
		const float current_value =
			unpack_argmax_value(
				observed
			);

		const std::uint32_t current_index =
			unpack_argmax_index(
				observed
			);

		const bool current_valid =
			current_index
			!= ARGMAX_INVALID_INDEX;

		if (
			current_valid
			&& !is_better_argmax(
				candidate.value,
				candidate.index,
				current_value,
				current_index
			)
		) {
			return;
		}

		const unsigned long long desired =
			pack_argmax_result(
				candidate.value,
				candidate.index
			);

		const unsigned long long previous =
			atomicCAS(
				packed_workspace,
				observed,
				desired
			);

		if (
			previous == observed
		) {
			return;
		}

		observed =
			previous;
	}
}


// ============================================================
// Workspace initialization
// ============================================================

__global__
void initialize_argmax_atomic_workspace_kernel(
	unsigned long long *packed_workspace
) {
	if (
		blockIdx.x == 0
		&& threadIdx.x == 0
	) {
		*packed_workspace =
			pack_argmax_result(
				0.0F,
				ARGMAX_INVALID_INDEX
			);
	}
}


// ============================================================
// Multi-block Shuffle64 + atomicCAS kernel
// ============================================================


template <
	typename T,
	unsigned int BLOCK_SIZE
>
__global__
void argmax_multiblock_shuffle_atomic_kernel(
	unsigned long long *__restrict__ packed_workspace,
	const T *__restrict__ vals,
	std::size_t numel
) {
	constexpr unsigned int NUM_WARPS =
		BLOCK_SIZE
		/ METAX_ARGMAX_WARP_SIZE;

	__shared__
	ArgmaxResult warp_results[
		NUM_WARPS
	];

	const std::size_t start =
		static_cast<std::size_t>(
			blockIdx.x
		)
		* static_cast<std::size_t>(
			BLOCK_SIZE
		)
		+ static_cast<std::size_t>(
			threadIdx.x
		);

	const std::size_t stride =
		static_cast<std::size_t>(
			BLOCK_SIZE
		)
		* static_cast<std::size_t>(
			gridDim.x
		);

	ArgmaxResult thread_result =
		invalid_argmax_result();

	for (
		std::size_t index = start;
		index < numel;
		index += stride
	) {
		update_argmax_result(
			thread_result,
			to_float<T>(
				vals[index]
			),
			static_cast<std::uint32_t>(
				index
			)
		);
	}

	const ArgmaxResult block_result =
		block_reduce_argmax_shuffle64<
			BLOCK_SIZE
		>(
			thread_result,
			warp_results
		);

	if (
		threadIdx.x == 0
	) {
		atomic_merge_argmax_result(
			packed_workspace,
			block_result
		);
	}
}


// ============================================================
// Final output kernel
// ============================================================


template <typename T>
__global__
void finalize_argmax_atomic_result_kernel(
	const unsigned long long *__restrict__ packed_workspace,
	std::int64_t *__restrict__ max_idx,
	T *__restrict__ max_val
) {
	if (
		blockIdx.x != 0
		|| threadIdx.x != 0
	) {
		return;
	}

	const unsigned long long packed =
		*packed_workspace;

	const float value =
		unpack_argmax_value(
			packed
		);

	const std::uint32_t index =
		unpack_argmax_index(
			packed
		);

	if (
		index == ARGMAX_INVALID_INDEX
	) {
		return;
	}

	*max_idx =
		static_cast<std::int64_t>(
			index
		);

	*max_val =
		from_float<T>(
			value
		);
}


// ============================================================
// Fixed block-size launcher
// ============================================================


template <
	typename T,
	unsigned int BLOCK_SIZE
>
inline
void launch_argmax_multiblock_shuffle_atomic_fixed(
	std::int64_t *max_idx,
	T *max_val,
	const T *vals,
	std::size_t numel,
	unsigned long long *packed_workspace,
	unsigned int grid_size,
	mcStream_t stream
) {
	initialize_argmax_atomic_workspace_kernel
		<<<1, 1, 0, stream>>>(
			packed_workspace
		);

	MC_CHECK(
		mcGetLastError()
	);

	argmax_multiblock_shuffle_atomic_kernel<
		T,
		BLOCK_SIZE
	>
		<<<
			grid_size,
			BLOCK_SIZE,
			0,
			stream
		>>>(
			packed_workspace,
			vals,
			numel
		);

	MC_CHECK(
		mcGetLastError()
	);

	finalize_argmax_atomic_result_kernel<T>
		<<<1, 1, 0, stream>>>(
			packed_workspace,
			max_idx,
			max_val
		);

	MC_CHECK(
		mcGetLastError()
	);
}


// ============================================================
// Runtime block-size dispatch
// ============================================================


template <typename T>
inline
void launch_argmax_multiblock_shuffle_atomic(
	std::int64_t *max_idx,
	T *max_val,
	const T *vals,
	std::size_t numel,
	unsigned long long *packed_workspace,
	unsigned int block_size,
	unsigned int grid_size,
	mcStream_t stream
) {
	switch (block_size) {

	case 64:
		return
			launch_argmax_multiblock_shuffle_atomic_fixed<
				T,
				64
			>(
				max_idx,
				max_val,
				vals,
				numel,
				packed_workspace,
				grid_size,
				stream
			);

	case 128:
		return
			launch_argmax_multiblock_shuffle_atomic_fixed<
				T,
				128
			>(
				max_idx,
				max_val,
				vals,
				numel,
				packed_workspace,
				grid_size,
				stream
			);

	case 256:
		return
			launch_argmax_multiblock_shuffle_atomic_fixed<
				T,
				256
			>(
				max_idx,
				max_val,
				vals,
				numel,
				packed_workspace,
				grid_size,
				stream
			);

	default:
		throw std::invalid_argument(
			"MetaX atomic Argmax requires "
			"block size 64, 128, or 256."
		);
	}
}

} // namespace llaisys::ops::metax::detail

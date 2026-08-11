#pragma once

#include "../../../device/metax/metax_common.hpp"

#include "../cuda_compat/argmax_cuda_compat.cuh"

#include <cub/block/block_reduce.cuh>

#include <cstddef>
#include <cstdint>
#include <stdexcept>


namespace llaisys::ops::metax::detail {

namespace cuda_compat =
	llaisys::ops::cuda_compat;

using cuda_compat::ArgmaxResult;
using cuda_compat::block_reduce_argmax_portable;
using cuda_compat::from_float;
using cuda_compat::invalid_argmax_result;
using cuda_compat::is_valid_argmax_result;
using cuda_compat::to_float;
using cuda_compat::update_argmax_result;


static_assert(
	sizeof(ArgmaxResult) == 8,
	"ArgmaxResult is expected to contain "
	"one float and one uint32 index."
);


// ============================================================
// Shared CUB reduction operator
// ============================================================
//
// IMPORTANT:
//
// Do NOT duplicate Argmax ordering semantics here.
//
// The existing shared:
//
//     update_argmax_result()
//
// remains the source of truth for:
//
//     numeric comparison
//     NaN ordering
//     equal-value first-index tie breaking
//
// Therefore:
//
//     Tree
//     CUB
//
// differ only in the block-wide reduction primitive.
// ============================================================

struct ArgmaxReduceOp {

	__device__
	ArgmaxResult operator()(
		const ArgmaxResult &left,
		const ArgmaxResult &right
	) const {
		ArgmaxResult result =
			left;

		if (
			is_valid_argmax_result(
				right
			)
		) {
			update_argmax_result(
				result,
				right.value,
				right.index
			);
		}

		return
			result;
	}
};


// ============================================================
// TREE Stage 1
// ============================================================
//
// Multiple blocks scan the original input.
//
// Each block:
//     grid-stride scan
//         ↓
//     one ArgmaxResult per thread
//         ↓
//     portable shared-memory tree reduction
//         ↓
//     one partial result
//
// ============================================================

template <typename T>
__global__ void argmax_multiblock_tree_stage1_kernel(
	ArgmaxResult *__restrict__ partial_results,
	const T *__restrict__ vals,
	std::size_t numel
) {
	extern __shared__
	ArgmaxResult shared_results[];

	const std::size_t start =
		static_cast<std::size_t>(
			blockIdx.x
		)
		* static_cast<std::size_t>(
			blockDim.x
		)
		+ static_cast<std::size_t>(
			threadIdx.x
		);

	const std::size_t stride =
		static_cast<std::size_t>(
			blockDim.x
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
		const float value =
			to_float<T>(
				vals[index]
			);

		update_argmax_result(
			thread_result,
			value,
			static_cast<std::uint32_t>(
				index
			)
		);
	}

	const ArgmaxResult block_result =
		block_reduce_argmax_portable(
			thread_result,
			shared_results
		);

	if (
		threadIdx.x == 0
		&& is_valid_argmax_result(
			block_result
		)
	) {
		partial_results[
			blockIdx.x
		] = block_result;
	}
}


// ============================================================
// TREE Stage 2
// ============================================================

template <typename T>
__global__ void argmax_multiblock_tree_stage2_kernel(
	std::int64_t *__restrict__ max_idx,
	T *__restrict__ max_val,
	const ArgmaxResult *__restrict__ partial_results,
	std::size_t partial_count
) {
	extern __shared__
	ArgmaxResult shared_results[];

	ArgmaxResult thread_result =
		invalid_argmax_result();

	for (
		std::size_t index =
			static_cast<std::size_t>(
				threadIdx.x
			);
		index < partial_count;
		index += static_cast<std::size_t>(
			blockDim.x
		)
	) {
		const ArgmaxResult candidate =
			partial_results[index];

		if (
			is_valid_argmax_result(
				candidate
			)
		) {
			update_argmax_result(
				thread_result,
				candidate.value,
				candidate.index
			);
		}
	}

	const ArgmaxResult final_result =
		block_reduce_argmax_portable(
			thread_result,
			shared_results
		);

	if (
		threadIdx.x == 0
		&& is_valid_argmax_result(
			final_result
		)
	) {
		*max_idx =
			static_cast<std::int64_t>(
				final_result.index
			);

		*max_val =
			from_float<T>(
				final_result.value
			);
	}
}


// ============================================================
// TREE launcher
// ============================================================

template <typename T>
inline void launch_argmax_multiblock_tree(
	std::int64_t *max_idx,
	T *max_val,
	const T *vals,
	std::size_t numel,
	ArgmaxResult *partial_results,
	unsigned int block_size,
	unsigned int grid_size,
	mcStream_t stream
) {
	if (
		numel == 0
		|| grid_size == 0
	) {
		return;
	}

	const std::size_t shared_memory_bytes =
		static_cast<std::size_t>(
			block_size
		)
		* sizeof(
			ArgmaxResult
		);

	// --------------------------------------------------------
	// Stage 1
	// --------------------------------------------------------

	argmax_multiblock_tree_stage1_kernel<T>
		<<<
			grid_size,
			block_size,
			shared_memory_bytes,
			stream
		>>>(
			partial_results,
			vals,
			numel
		);

	MC_CHECK(
		mcGetLastError()
	);

	// --------------------------------------------------------
	// Stage 2
	// --------------------------------------------------------

	argmax_multiblock_tree_stage2_kernel<T>
		<<<
			1,
			block_size,
			shared_memory_bytes,
			stream
		>>>(
			max_idx,
			max_val,
			partial_results,
			static_cast<std::size_t>(
				grid_size
			)
		);

	MC_CHECK(
		mcGetLastError()
	);
}


// ============================================================
// CUB Stage 1
// ============================================================
//
// BLOCK_SIZE must be compile-time constant for CUB.
//
// The input scan is intentionally identical to TREE.
// Only the block-wide reduction primitive changes.
//
// ============================================================

template <
	typename T,
	unsigned int BLOCK_SIZE
>
__global__ void argmax_multiblock_cub_stage1_kernel(
	ArgmaxResult *__restrict__ partial_results,
	const T *__restrict__ vals,
	std::size_t numel
) {
	using BlockReduce =
		cub::BlockReduce<
			ArgmaxResult,
			BLOCK_SIZE
		>;

	__shared__
	typename BlockReduce::TempStorage
		reduction_storage;

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
		const float value =
			to_float<T>(
				vals[index]
			);

		update_argmax_result(
			thread_result,
			value,
			static_cast<std::uint32_t>(
				index
			)
		);
	}

	const ArgmaxResult block_result =
		BlockReduce(
			reduction_storage
		).Reduce(
			thread_result,
			ArgmaxReduceOp{}
		);

	if (
		threadIdx.x == 0
		&& is_valid_argmax_result(
			block_result
		)
	) {
		partial_results[
			blockIdx.x
		] = block_result;
	}
}


// ============================================================
// CUB Stage 2
// ============================================================

template <
	typename T,
	unsigned int BLOCK_SIZE
>
__global__ void argmax_multiblock_cub_stage2_kernel(
	std::int64_t *__restrict__ max_idx,
	T *__restrict__ max_val,
	const ArgmaxResult *__restrict__ partial_results,
	std::size_t partial_count
) {
	using BlockReduce =
		cub::BlockReduce<
			ArgmaxResult,
			BLOCK_SIZE
		>;

	__shared__
	typename BlockReduce::TempStorage
		reduction_storage;

	ArgmaxResult thread_result =
		invalid_argmax_result();

	for (
		std::size_t index =
			static_cast<std::size_t>(
				threadIdx.x
			);
		index < partial_count;
		index += static_cast<std::size_t>(
			BLOCK_SIZE
		)
	) {
		const ArgmaxResult candidate =
			partial_results[index];

		if (
			is_valid_argmax_result(
				candidate
			)
		) {
			update_argmax_result(
				thread_result,
				candidate.value,
				candidate.index
			);
		}
	}

	const ArgmaxResult final_result =
		BlockReduce(
			reduction_storage
		).Reduce(
			thread_result,
			ArgmaxReduceOp{}
		);

	if (
		threadIdx.x == 0
		&& is_valid_argmax_result(
			final_result
		)
	) {
		*max_idx =
			static_cast<std::int64_t>(
				final_result.index
			);

		*max_val =
			from_float<T>(
				final_result.value
			);
	}
}


// ============================================================
// Fixed-size CUB launcher
// ============================================================

template <
	typename T,
	unsigned int BLOCK_SIZE
>
inline void launch_argmax_multiblock_cub_fixed(
	std::int64_t *max_idx,
	T *max_val,
	const T *vals,
	std::size_t numel,
	ArgmaxResult *partial_results,
	unsigned int grid_size,
	mcStream_t stream
) {
	if (
		numel == 0
		|| grid_size == 0
	) {
		return;
	}

	// --------------------------------------------------------
	// Stage 1
	// --------------------------------------------------------

	argmax_multiblock_cub_stage1_kernel<
		T,
		BLOCK_SIZE
	>
		<<<
			grid_size,
			BLOCK_SIZE,
			0,
			stream
		>>>(
			partial_results,
			vals,
			numel
		);

	MC_CHECK(
		mcGetLastError()
	);

	// --------------------------------------------------------
	// Stage 2
	// --------------------------------------------------------

	argmax_multiblock_cub_stage2_kernel<
		T,
		BLOCK_SIZE
	>
		<<<
			1,
			BLOCK_SIZE,
			0,
			stream
		>>>(
			max_idx,
			max_val,
			partial_results,
			static_cast<std::size_t>(
				grid_size
			)
		);

	MC_CHECK(
		mcGetLastError()
	);
}


// ============================================================
// Runtime block-size → compile-time CUB specialization
// ============================================================
//
// We retain 64/128/256 because they already exist as Argmax
// experimental block-size controls.
//
// Production currently uses 256.
//
// ============================================================

template <typename T>
inline void launch_argmax_multiblock_cub(
	std::int64_t *max_idx,
	T *max_val,
	const T *vals,
	std::size_t numel,
	ArgmaxResult *partial_results,
	unsigned int block_size,
	unsigned int grid_size,
	mcStream_t stream
) {
	switch (block_size) {

	case 64:
		return
			launch_argmax_multiblock_cub_fixed<
				T,
				64
			>(
				max_idx,
				max_val,
				vals,
				numel,
				partial_results,
				grid_size,
				stream
			);

	case 128:
		return
			launch_argmax_multiblock_cub_fixed<
				T,
				128
			>(
				max_idx,
				max_val,
				vals,
				numel,
				partial_results,
				grid_size,
				stream
			);

	case 256:
		return
			launch_argmax_multiblock_cub_fixed<
				T,
				256
			>(
				max_idx,
				max_val,
				vals,
				numel,
				partial_results,
				grid_size,
				stream
			);

	default:
		throw std::invalid_argument(
			"Argmax CUB reduction requires "
			"block size 64, 128, or 256."
		);
	}
}

} // namespace llaisys::ops::metax::detail
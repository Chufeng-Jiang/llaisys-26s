#include "swiglu_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include "../cuda_compat/swiglu_cuda_compat.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

namespace cuda_compat =
	llaisys::ops::cuda_compat;

using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::device::nvidia::get_warp_aligned_block_size;
using llaisys::device::nvidia::to_cuda_stream;

// ============================================================
// NVIDIA SwiGLU adapter
//
// CUDA-compatible shared layer owns:
//
//   - SwiGLU arithmetic
//   - FP32 evaluation order
//   - scalar kernel
//   - 128-bit packed kernel
//   - tail handling
//   - packed-path eligibility
//
// NVIDIA adapter owns:
//
//   - CUDA block-size tuning
//   - CUDA grid-size tuning
//   - CUDA stream type
//   - CUDA launch error handling
// ============================================================

template <typename T>
void launch_nvidia_swiglu(
	T *out,
	const T *gate,
	const T *up,
	std::size_t numel,
	cudaStream_t stream
) {
	if (numel == 0) {
		return;
	}

	// ========================================================
	// Shared algorithm-path selection
	// ========================================================

	const bool use_packed_kernel =
		cuda_compat::
			can_use_packed_swiglu<T>(
				out,
				gate,
				up,
				numel
			);

	const std::size_t work_items =
		cuda_compat::
			get_swiglu_work_items<T>(
				numel,
				use_packed_kernel
			);

	// ========================================================
	// NVIDIA-specific launch tuning
	// ========================================================

	const unsigned int block_size =
		get_warp_aligned_block_size(
			work_items
		);

	const std::size_t grid_size =
		get_capped_grid_size(
			work_items,
			static_cast<std::size_t>(
				block_size
			)
		);

	// ========================================================
	// Shared CUDA-compatible implementation
	// ========================================================

	cuda_compat::
		launch_swiglu_kernel<T>(
			out,
			gate,
			up,
			numel,
			block_size,
			grid_size,
			use_packed_kernel,
			stream
		);

	// ========================================================
	// NVIDIA-specific error handling
	// ========================================================

	CUDA_CHECK(
		cudaGetLastError()
	);
}

} // namespace

// ============================================================
// Public NVIDIA backend
// ============================================================

namespace llaisys::ops::nvidia {

void swiglu(
	std::byte *out,
	const std::byte *gate,
	const std::byte *up,
	llaisysDataType_t type,
	std::size_t numel,
	llaisysStream_t stream
) {
	CHECK_ARGUMENT(
		numel == 0 || out != nullptr,
		"SwiGLU: output pointer must not be null."
	);

	CHECK_ARGUMENT(
		numel == 0 || gate != nullptr,
		"SwiGLU: gate pointer must not be null."
	);

	CHECK_ARGUMENT(
		numel == 0 || up != nullptr,
		"SwiGLU: up pointer must not be null."
	);

	if (numel == 0) {
		return;
	}

	const cudaStream_t cuda_stream =
		to_cuda_stream(
			stream
		);

	return llaisys::device::nvidia::
		dispatch_cuda_dtype(
			type,
			[&](auto tag) {
				using T =
					typename decltype(tag)::type;

				return launch_nvidia_swiglu<T>(
					reinterpret_cast<T *>(
						out
					),
					reinterpret_cast<
						const T *
					>(
						gate
					),
					reinterpret_cast<
						const T *
					>(
						up
					),
					numel,
					cuda_stream
				);
			}
		);
}

} // namespace llaisys::ops::nvidia
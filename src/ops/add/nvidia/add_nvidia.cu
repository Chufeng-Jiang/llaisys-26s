#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include "../cuda_compat/add_cuda_compat.cuh"
#include "add_nvidia.cuh"

#include <cstddef>

namespace {

namespace cuda_compat =
	llaisys::ops::cuda_compat;

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::device::nvidia::to_cuda_stream;

// ============================================================
// NVIDIA Add adapter
//
// Shared:
//   - scalar addition
//   - scalar kernel
//   - vector traits
//   - vectorized kernel
//   - tail handling
//
// NVIDIA-specific:
//   - CUDA block size
//   - CUDA grid cap
//   - cudaStream_t
//   - CUDA launch error checking
// ============================================================

template <typename T>
void launch_nvidia_add(
	T *c,
	const T *a,
	const T *b,
	std::size_t numel,
	cudaStream_t stream
) {
	if (numel == 0) {
		return;
	}

	// ========================================================
	// Select shared scalar/vectorized implementation
	// ========================================================

	const bool use_vectorized_kernel =
		cuda_compat::can_use_vectorized_add<T>(
			c,
			a,
			b,
			numel
		);

	// ========================================================
	// NVIDIA-specific launch configuration
	// ========================================================

	constexpr std::size_t block_size =
		CUDA_BLOCK_SIZE;

	const std::size_t work_items =
		cuda_compat::get_add_work_items<T>(
			numel,
			use_vectorized_kernel
		);

	const std::size_t grid_size =
		get_capped_grid_size(
			work_items,
			block_size
		);

	// ========================================================
	// Shared CUDA-compatible kernel
	// ========================================================

	cuda_compat::launch_add_kernel<T>(
		c,
		a,
		b,
		numel,
		block_size,
		grid_size,
		use_vectorized_kernel,
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

namespace llaisys::ops::nvidia {

void add(
	std::byte *c,
	const std::byte *a,
	const std::byte *b,
	llaisysDataType_t type,
	std::size_t numel,
	llaisysStream_t stream
) {
	CHECK_ARGUMENT(
		numel == 0 || c != nullptr,
		"Add: output pointer c must not be null."
	);

	CHECK_ARGUMENT(
		numel == 0 || a != nullptr,
		"Add: input pointer a must not be null."
	);

	CHECK_ARGUMENT(
		numel == 0 || b != nullptr,
		"Add: input pointer b must not be null."
	);

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

				return launch_nvidia_add<T>(
					reinterpret_cast<T *>(
						c
					),
					reinterpret_cast<
						const T *
					>(
						a
					),
					reinterpret_cast<
						const T *
					>(
						b
					),
					numel,
					cuda_stream
				);
			}
		);
}

} // namespace llaisys::ops::nvidia
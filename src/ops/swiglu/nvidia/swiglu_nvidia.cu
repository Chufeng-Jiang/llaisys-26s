#include "swiglu_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

namespace llaisys::ops::nvidia {
namespace {

using llaisys::device::nvidia::are_aligned;
using llaisys::device::nvidia::div_ceil;
using llaisys::device::nvidia::from_float;
using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::device::nvidia::get_warp_aligned_block_size;
using llaisys::device::nvidia::Packed128;
using llaisys::device::nvidia::PACKED_128_ALIGNMENT;
using llaisys::device::nvidia::PACKED_128_ELEMENTS;
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::to_cuda_stream;


// Keep the same FP32 evaluation order as the reference:
//
//     up * gate / (1 + exp(-gate))
//
// Do not rewrite this as:
//
//     up * (gate / denominator)
//
// Also do not use __expf, because it is a lower-precision
// approximation and may fail strict F32 tests.
__device__ __forceinline__ float swiglu_value(
	float gate_value,
	float up_value
) {
	const float denominator =
		1.0F + expf(-gate_value);

	return up_value * gate_value / denominator;
}

template <typename T>
__device__ __forceinline__ T swiglu_element(
	T gate_value,
	T up_value
) {
	const float gate_float =
		to_float<T>(gate_value);

	const float up_float =
		to_float<T>(up_value);

	return from_float<T>(
		swiglu_value(
			gate_float,
			up_float
		)
	);
}

// ============================================================
// Scalar fallback
// ============================================================
//
// Used when:
// - one of the pointers is not 16-byte aligned;
// - numel is smaller than one complete 128-bit pack.
//
// A grid-stride loop allows the grid size to remain capped.
template <typename T>
__global__ void swiglu_scalar_kernel(
	T *out,
	const T *gate,
	const T *up,
	std::size_t numel
) {
	const std::size_t thread_index =
		static_cast<std::size_t>(blockIdx.x)
			* static_cast<std::size_t>(blockDim.x)
		+ static_cast<std::size_t>(threadIdx.x);

	const std::size_t thread_stride =
		static_cast<std::size_t>(gridDim.x)
			* static_cast<std::size_t>(blockDim.x);

	for (
		std::size_t index = thread_index;
		index < numel;
		index += thread_stride
	) {
		// Load both inputs before writing output.
		//
		// This keeps exact in-place execution safe:
		//
		//     out == gate
		//     out == up
		const T gate_value =
			gate[index];

		const T up_value =
			up[index];

		out[index] =
			swiglu_element<T>(
				gate_value,
				up_value
			);
	}
}

// ============================================================
// 128-bit packed kernel
// ============================================================
//
// Elements handled by each Packed128:
//
//     float:          4
//     half:           8
//     __nv_bfloat16:  8
//
// Arithmetic is still performed in FP32 because expf does not
// have a useful packed F16/BF16 equivalent. The packed path
// reduces global-memory load/store instructions.
template <typename T>
__global__ void swiglu_packed_kernel(
	T *out,
	const T *gate,
	const T *up,
	std::size_t numel
) {
	constexpr std::size_t elements_per_pack =
		PACKED_128_ELEMENTS<T>;

	static_assert(
		elements_per_pack > 0,
		"Packed128 must contain at least one element."
	);

	const std::size_t pack_count =
		numel / elements_per_pack;

	const std::size_t thread_index =
		static_cast<std::size_t>(blockIdx.x)
			* static_cast<std::size_t>(blockDim.x)
		+ static_cast<std::size_t>(threadIdx.x);

	const std::size_t thread_stride =
		static_cast<std::size_t>(gridDim.x)
			* static_cast<std::size_t>(blockDim.x);

	const Packed128 *gate_packs =
		reinterpret_cast<const Packed128 *>(gate);

	const Packed128 *up_packs =
		reinterpret_cast<const Packed128 *>(up);

	Packed128 *out_packs =
		reinterpret_cast<Packed128 *>(out);

	for (
		std::size_t pack_index = thread_index;
		pack_index < pack_count;
		pack_index += thread_stride
	) {
		// Load complete gate/up packs before storing output.
		// This preserves exact in-place execution.
		const Packed128 gate_pack =
			gate_packs[pack_index];

		const Packed128 up_pack =
			up_packs[pack_index];

		Packed128 out_pack{};

		const T *gate_values =
			reinterpret_cast<const T *>(
				&gate_pack
			);

		const T *up_values =
			reinterpret_cast<const T *>(
				&up_pack
			);

		T *out_values =
			reinterpret_cast<T *>(
				&out_pack
			);

#pragma unroll
		for (
			std::size_t lane = 0;
			lane < elements_per_pack;
			++lane
		) {
			out_values[lane] =
				swiglu_element<T>(
					gate_values[lane],
					up_values[lane]
				);
		}

		out_packs[pack_index] =
			out_pack;
	}

	// Handle elements that do not fill one complete
	// 128-bit pack.
	const std::size_t tail_start =
		pack_count * elements_per_pack;

	for (
		std::size_t index =
			tail_start + thread_index;
		index < numel;
		index += thread_stride
	) {
		const T gate_value =
			gate[index];

		const T up_value =
			up[index];

		out[index] =
			swiglu_element<T>(
				gate_value,
				up_value
			);
	}
}

// ============================================================
// Kernel launcher
// ============================================================

template <typename T>
void launch_swiglu(
	T *out,
	const T *gate,
	const T *up,
	std::size_t numel,
	cudaStream_t stream
) {
	constexpr std::size_t elements_per_pack =
		PACKED_128_ELEMENTS<T>;

	const bool use_packed_kernel =
		numel >= elements_per_pack
		&& are_aligned<PACKED_128_ALIGNMENT>(
			out,
			gate,
			up
		);

	// The packed kernel handles the remaining tail internally.
	const std::size_t work_items =
		use_packed_kernel
			? div_ceil(
				numel,
				elements_per_pack
			)
			: numel;

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

	if (use_packed_kernel) {
		swiglu_packed_kernel<T><<<
			static_cast<unsigned int>(
				grid_size
			),
			block_size,
			0,
			stream
		>>>(
			out,
			gate,
			up,
			numel
		);
	} else {
		swiglu_scalar_kernel<T><<<
			static_cast<unsigned int>(
				grid_size
			),
			block_size,
			0,
			stream
		>>>(
			out,
			gate,
			up,
			numel
		);
	}

	// Detect launch and configuration errors without
	// synchronizing the entire CUDA device.
	CUDA_CHECK(cudaGetLastError());
}

} // namespace

// ============================================================
// Public NVIDIA backend
// ============================================================

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

	// CUDA does not permit launching a kernel with
	// zero blocks.
	if (numel == 0) {
		return;
	}

	const cudaStream_t cuda_stream = llaisys::device::nvidia::to_cuda_stream(stream);

	return llaisys::device::nvidia::dispatch_cuda_dtype(
		type,
		[&](auto tag) {
			using T = typename decltype(tag)::type;

			return launch_swiglu<T>(
				reinterpret_cast<T *>(out),
				reinterpret_cast<const T *>(gate),
				reinterpret_cast<const T *>(up),
				numel,
				cuda_stream
			);
		}
	);
}

} // namespace llaisys::ops::nvidia
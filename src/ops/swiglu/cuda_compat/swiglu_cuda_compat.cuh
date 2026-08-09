#pragma once

#include "../../cuda_compat/common.cuh"

#include <cmath>
#include <cstddef>

namespace llaisys::ops::cuda_compat {

// ============================================================
// SwiGLU scalar computation
// ============================================================
//
// Preserve the FP32 evaluation order:
//
//     up * gate / (1 + exp(-gate))
//
// Do not rewrite this as:
//
//     up * (gate / denominator)
//
// Also do not use __expf here. The current NVIDIA
// implementation intentionally uses expf to preserve strict
// F32 numerical behavior.
// ============================================================

__device__ __forceinline__ float swiglu_value(
	float gate_value,
	float up_value
) {
	const float denominator =
		1.0F + expf(-gate_value);

	return up_value
		* gate_value
		/ denominator;
}

template <typename T>
__device__ __forceinline__ T swiglu_element(
	T gate_value,
	T up_value
) {
	const float gate_float =
		to_float<T>(
			gate_value
		);

	const float up_float =
		to_float<T>(
			up_value
		);

	return from_float<T>(
		swiglu_value(
			gate_float,
			up_float
		)
	);
}

// ============================================================
// Scalar kernel
// ============================================================
//
// Used when:
//
//   - one or more pointers are not 16-byte aligned;
//   - numel is smaller than one complete 128-bit pack.
//
// The grid-stride loop allows a backend to cap its grid size.
// ============================================================

template <typename T>
__global__ void swiglu_scalar_kernel(
	T *out,
	const T *gate,
	const T *up,
	std::size_t numel
) {
	const std::size_t thread_index =
		static_cast<std::size_t>(
			blockIdx.x
		)
		* static_cast<std::size_t>(
			blockDim.x
		)
		+ static_cast<std::size_t>(
			threadIdx.x
		);

	const std::size_t thread_stride =
		static_cast<std::size_t>(
			gridDim.x
		)
		* static_cast<std::size_t>(
			blockDim.x
		);

	for (
		std::size_t index = thread_index;
		index < numel;
		index += thread_stride
	) {
		// Load both operands before writing the result.
		//
		// This preserves exact in-place execution:
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
// Number of elements represented by a Packed128:
//
//     float:          4
//     half:           8
//     __nv_bfloat16:  8
//
// Arithmetic remains FP32 because the expensive nonlinear
// operation is expf. Packed access primarily reduces the number
// of global-memory load/store instructions.
// ============================================================

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
		static_cast<std::size_t>(
			blockIdx.x
		)
		* static_cast<std::size_t>(
			blockDim.x
		)
		+ static_cast<std::size_t>(
			threadIdx.x
		);

	const std::size_t thread_stride =
		static_cast<std::size_t>(
			gridDim.x
		)
		* static_cast<std::size_t>(
			blockDim.x
		);

	const Packed128 *gate_packs =
		reinterpret_cast<
			const Packed128 *
		>(
			gate
		);

	const Packed128 *up_packs =
		reinterpret_cast<
			const Packed128 *
		>(
			up
		);

	Packed128 *out_packs =
		reinterpret_cast<
			Packed128 *
		>(
			out
		);

	for (
		std::size_t pack_index =
			thread_index;
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
			reinterpret_cast<
				const T *
			>(
				&gate_pack
			);

		const T *up_values =
			reinterpret_cast<
				const T *
			>(
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

	// ========================================================
	// Scalar tail
	// ========================================================

	const std::size_t tail_start =
		pack_count
		* elements_per_pack;

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
// Packed-path eligibility
// ============================================================

template <typename T>
inline bool can_use_packed_swiglu(
	const T *out,
	const T *gate,
	const T *up,
	std::size_t numel
) {
	constexpr std::size_t elements_per_pack =
		PACKED_128_ELEMENTS<T>;

	return
		numel >= elements_per_pack
		&& are_aligned<
			PACKED_128_ALIGNMENT
		>(
			out,
			gate,
			up
		);
}

// ============================================================
// Logical work-item count
//
// The CUDA-compatible algorithm determines how many logical
// pieces of work exist.
//
// The vendor adapter determines how those work items are mapped
// to its physical grid/block configuration.
// ============================================================

template <typename T>
inline std::size_t get_swiglu_work_items(
	std::size_t numel,
	bool use_packed_kernel
) {
	if (!use_packed_kernel) {
		return numel;
	}

	constexpr std::size_t elements_per_pack =
		PACKED_128_ELEMENTS<T>;

	// The packed kernel handles a partial final pack through
	// its scalar tail path, so include that tail when sizing
	// the logical work.
	return div_ceil(
		numel,
		elements_per_pack
	);
}

// ============================================================
// Shared CUDA-compatible launcher
//
// Deliberately NOT responsible for:
//
//   - block-size selection
//   - grid-size capping
//   - llaisysStream_t conversion
//   - CUDA error handling
//
// Those belong to each vendor adapter.
// ============================================================

template <
	typename T,
	typename StreamT
>
inline void launch_swiglu_kernel(
	T *out,
	const T *gate,
	const T *up,
	std::size_t numel,
	unsigned int block_size,
	std::size_t grid_size,
	bool use_packed_kernel,
	StreamT stream
) {
	if (numel == 0) {
		return;
	}

	if (use_packed_kernel) {
		swiglu_packed_kernel<T>
			<<<
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
		swiglu_scalar_kernel<T>
			<<<
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
}

} // namespace llaisys::ops::cuda_compat
#include "swiglu_cpu.hpp"

#include "../../../device/cpu/cpu_common.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace {

using llaisys::device::cpu::OPENMP_THRESHOLD;

inline float swiglu_value(
	float gate_value,
	float up_value
) {
	// Preserve the same Float32 evaluation order as the reference:
	//
	//     up * gate / (1 + exp(-gate))
	//
	// Avoid replacing the division with multiplication by a reciprocal,
	// because that can produce slightly different Float32 rounding.
	const float denominator =
		1.0F + std::exp(-gate_value);

	return up_value * gate_value / denominator;
}

template <typename T>
void swiglu_impl(
	T *out,
	const T *gate,
	const T *up,
	std::size_t numel
) {
	static_assert(
		std::is_same_v<T, float>
			|| std::is_same_v<T, llaisys::fp16_t>
			|| std::is_same_v<T, llaisys::bf16_t>,
		"SwiGLU: unsupported CPU element type."
	);

	if (numel == 0) {
		return;
	}

	CHECK_ARGUMENT(
		out != nullptr,
		"SwiGLU: output pointer must not be null."
	);

	CHECK_ARGUMENT(
		gate != nullptr,
		"SwiGLU: gate pointer must not be null."
	);

	CHECK_ARGUMENT(
		up != nullptr,
		"SwiGLU: up pointer must not be null."
	);

	const bool use_openmp =
		numel >= OPENMP_THRESHOLD;

	// Exact in-place execution remains safe for:
	//
	//     out == gate
	//     out == up
	//
	// because both input values are loaded before out[i] is written.
#pragma omp parallel for simd if(use_openmp) schedule(static)
	for (std::size_t index = 0; index < numel; ++index) {
		if constexpr (std::is_same_v<T, float>) {
			const float gate_value =
				gate[index];

			const float up_value =
				up[index];

			out[index] =
				swiglu_value(
					gate_value,
					up_value
				);
		} else {
			const float gate_value =
				llaisys::utils::cast<float>(
					gate[index]
				);

			const float up_value =
				llaisys::utils::cast<float>(
					up[index]
				);

			const float result =
				swiglu_value(
					gate_value,
					up_value
				);

			out[index] =
				llaisys::utils::cast<T>(
					result
				);
		}
	}
}

} // namespace

namespace llaisys::ops::cpu {

void swiglu(
	std::byte *out,
	const std::byte *gate,
	const std::byte *up,
	llaisysDataType_t type,
	std::size_t numel
) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return swiglu_impl<float>(
			reinterpret_cast<float *>(out),
			reinterpret_cast<const float *>(gate),
			reinterpret_cast<const float *>(up),
			numel
		);

	case LLAISYS_DTYPE_F16:
		return swiglu_impl<llaisys::fp16_t>(
			reinterpret_cast<llaisys::fp16_t *>(out),
			reinterpret_cast<const llaisys::fp16_t *>(gate),
			reinterpret_cast<const llaisys::fp16_t *>(up),
			numel
		);

	case LLAISYS_DTYPE_BF16:
		return swiglu_impl<llaisys::bf16_t>(
			reinterpret_cast<llaisys::bf16_t *>(out),
			reinterpret_cast<const llaisys::bf16_t *>(gate),
			reinterpret_cast<const llaisys::bf16_t *>(up),
			numel
		);

	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}

} // namespace llaisys::ops::cpu
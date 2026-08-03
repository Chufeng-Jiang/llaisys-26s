#include "linear_cpu.hpp"

#include "matmul.hpp"
#include "vecmul.hpp"

#include "../../../device/cpu/cpu_common.hpp"
#include "../../../utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

void initialize_output(
	float *out,
	const float *bias,
	std::size_t row_count,
	std::size_t output_features
) {
	const std::size_t output_elements =
		row_count * output_features;

	if (bias == nullptr) {
		std::fill(
			out,
			out + output_elements,
			0.0F
		);
		return;
	}

	const bool use_openmp =
		row_count >= 8
		&& output_features >= 256;

#pragma omp parallel for if(use_openmp) schedule(static)
	for (std::size_t row = 0; row < row_count; ++row) {
		std::memcpy(
			out + row * output_features,
			bias,
			output_features * sizeof(float)
		);
	}
}

template <typename T>
void convert_to_float(
	float *destination,
	const T *source,
	std::size_t count
) {
	static_assert(
		std::is_same_v<T, llaisys::fp16_t>
			|| std::is_same_v<T, llaisys::bf16_t>,
		"Linear: unsupported conversion source type."
	);

	CHECK_ARGUMENT(
		count == 0 || destination != nullptr,
		"Linear: conversion destination pointer must not be null."
	);

	CHECK_ARGUMENT(
		count == 0 || source != nullptr,
		"Linear: conversion source pointer must not be null."
	);

	const bool use_openmp =
		count >= llaisys::device::cpu::OPENMP_THRESHOLD;

#pragma omp parallel for if(use_openmp) schedule(static)
	for (std::size_t i = 0; i < count; ++i) {
		destination[i] =
			llaisys::utils::cast<float>(source[i]);
	}
}

template <typename T>
void convert_from_float(
	T *destination,
	const float *source,
	std::size_t count
) {
	static_assert(
		std::is_same_v<T, llaisys::fp16_t>
			|| std::is_same_v<T, llaisys::bf16_t>,
		"Linear: unsupported conversion destination type."
	);

	CHECK_ARGUMENT(
		count == 0 || destination != nullptr,
		"Linear: conversion destination pointer must not be null."
	);

	CHECK_ARGUMENT(
		count == 0 || source != nullptr,
		"Linear: conversion source pointer must not be null."
	);

	const bool use_openmp =
		count >= llaisys::device::cpu::OPENMP_THRESHOLD;

#pragma omp parallel for if(use_openmp) schedule(static)
	for (std::size_t i = 0; i < count; ++i) {
		destination[i] =
			llaisys::utils::cast<T>(source[i]);
	}
}

void linear_float(
	float *out,
	const float *in,
	const float *weight,
	const float *bias,
	std::size_t row_count,
	std::size_t output_features,
	std::size_t input_features
) {
	initialize_output(
		out,
		bias,
		row_count,
		output_features
	);

	// With K == 0, XW^T contributes zero and the initialized bias/zero
	// output is already the final result.
	if (input_features == 0) {
		return;
	}

	if (row_count == 1) {
		vecmul(
			in,
			weight,
			out,
			output_features,
			input_features
		);
		return;
	}

	matmul(
		in,
		weight,
		out,
		row_count,
		output_features,
		input_features
	);
}

template <typename T>
void linear_impl(
	T *out,
	const T *in,
	const T *weight,
	const T *bias,
	std::size_t row_count,
	std::size_t output_features,
	std::size_t input_features
) {
	const std::size_t max_size =
		std::numeric_limits<std::size_t>::max();

	CHECK_ARGUMENT(
		row_count == 0
			|| output_features <= max_size / row_count,
		"Linear: output element count overflows size_t."
	);

	CHECK_ARGUMENT(
		row_count == 0
			|| input_features <= max_size / row_count,
		"Linear: input element count overflows size_t."
	);

	CHECK_ARGUMENT(
		output_features == 0
			|| input_features <= max_size / output_features,
		"Linear: weight element count overflows size_t."
	);

	const std::size_t output_elements =
		row_count * output_features;

	const std::size_t input_elements =
		row_count * input_features;

	const std::size_t weight_elements =
		output_features * input_features;

	CHECK_ARGUMENT(
		output_elements == 0 || out != nullptr,
		"Linear: output pointer must not be null."
	);

	CHECK_ARGUMENT(
		input_elements == 0 || in != nullptr,
		"Linear: input pointer must not be null."
	);

	CHECK_ARGUMENT(
		weight_elements == 0 || weight != nullptr,
		"Linear: weight pointer must not be null."
	);

	if (output_elements == 0) {
		return;
	}

	if constexpr (std::is_same_v<T, float>) {
		linear_float(
			out,
			in,
			weight,
			bias,
			row_count,
			output_features,
			input_features
		);
	} else {
		// Accumulate FP16/BF16 matrix products in FP32. The two low-precision
		// paths share the same implementation; only the conversion routines
		// differ.
		std::vector<float> input_float(input_elements);
		std::vector<float> weight_float(weight_elements);
		std::vector<float> output_float(output_elements);

		if (input_elements > 0) {
			convert_to_float(
				input_float.data(),
				in,
				input_elements
			);
		}

		if (weight_elements > 0) {
			convert_to_float(
				weight_float.data(),
				weight,
				weight_elements
			);
		}

		std::vector<float> bias_float;
		const float *bias_pointer = nullptr;

		if (bias != nullptr) {
			bias_float.resize(output_features);

			convert_to_float(
				bias_float.data(),
				bias,
				output_features
			);

			bias_pointer = bias_float.data();
		}

		linear_float(
			output_float.data(),
			input_float.data(),
			weight_float.data(),
			bias_pointer,
			row_count,
			output_features,
			input_features
		);

		convert_from_float(
			out,
			output_float.data(),
			output_elements
		);
	}
}

} // namespace

namespace llaisys::ops::cpu {

void linear(
	std::byte *out,
	const std::byte *in,
	const std::byte *weight,
	const std::byte *bias,
	llaisysDataType_t type,
	std::size_t nrow,
	std::size_t ncol_out,
	std::size_t ncol_in
) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return linear_impl<float>(
			reinterpret_cast<float *>(out),
			reinterpret_cast<const float *>(in),
			reinterpret_cast<const float *>(weight),
			reinterpret_cast<const float *>(bias),
			nrow,
			ncol_out,
			ncol_in
		);

	case LLAISYS_DTYPE_F16:
		return linear_impl<llaisys::fp16_t>(
			reinterpret_cast<llaisys::fp16_t *>(out),
			reinterpret_cast<const llaisys::fp16_t *>(in),
			reinterpret_cast<const llaisys::fp16_t *>(weight),
			reinterpret_cast<const llaisys::fp16_t *>(bias),
			nrow,
			ncol_out,
			ncol_in
		);

	case LLAISYS_DTYPE_BF16:
		return linear_impl<llaisys::bf16_t>(
			reinterpret_cast<llaisys::bf16_t *>(out),
			reinterpret_cast<const llaisys::bf16_t *>(in),
			reinterpret_cast<const llaisys::bf16_t *>(weight),
			reinterpret_cast<const llaisys::bf16_t *>(bias),
			nrow,
			ncol_out,
			ncol_in
		);

	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}

} // namespace llaisys::ops::cpu
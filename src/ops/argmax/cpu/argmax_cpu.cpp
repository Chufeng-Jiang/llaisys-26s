#include "argmax_cpu.hpp"

#include "../../../device/cpu/cpu_common.hpp"
#include "../../../device/cpu/cpu_dtype.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <omp.h>

namespace {

using llaisys::device::cpu::OPENMP_THRESHOLD;

// FP16 and BF16 values are converted to FP32 for comparison.
// Other supported data types use their original type.
template <typename T>
using accumulator_t = std::conditional_t<std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>,
										 float,
										 T>;

// Store a candidate maximum value and its original index.
template <typename T>
struct MaxResult {
	T value{};
	std::int64_t index{0};
	bool valid{false};
};

// Convert an input value to the type used for comparison.
template <typename T>
accumulator_t<T> to_accumulator(T value) {
	if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
		return llaisys::utils::cast<float>(value);
	} else {
		return value;
	}
}

// Convert the comparison result back to the input data type.
template <typename T>
T from_accumulator(accumulator_t<T> value) {
	if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
		return llaisys::utils::cast<T>(value);
	} else {
		return value;
	}
}

// Return true when the value is NaN.
template <typename T>
bool is_nan(T value) {
	if constexpr (std::is_floating_point_v<T>) {
		return std::isnan(value);
	} else {
		return false;
	}
}

template <typename T>
bool is_better(T candidate_value, std::int64_t candidate_index, T current_value, std::int64_t current_index) {
	const bool candidate_is_nan = is_nan(candidate_value);
	const bool current_is_nan = is_nan(current_value);

	if (candidate_is_nan != current_is_nan) {
		return candidate_is_nan;
	}

	if (candidate_value > current_value) {
		return true;
	}

	if (candidate_value < current_value) {
		return false;
	}

	return candidate_index < current_index;
}

// Merge two partial Argmax results.
// This function is used by the OpenMP custom reduction.
MaxResult<float> merge_max_results(const MaxResult<float> &left, const MaxResult<float> &right) {
	if (!left.valid) {
		return right;
	}

	if (!right.valid) {
		return left;
	}

	if (is_better(right.value, right.index, left.value, left.index)) {
		return right;
	}

	return left;
}

// Define an OpenMP reduction for MaxResult<float>.
//
// All currently supported Argmax data types use float as their
// accumulator type:
// F32  -> float
// F16  -> float
// BF16 -> float
#pragma omp declare reduction(                                                                      \
		llaisys_argmax_reduction : MaxResult<float> : omp_out = merge_max_results(omp_out, omp_in)) \
	initializer(omp_priv = MaxResult<float>())

template <typename T>
void argmax_impl(std::int64_t *max_idx, T *max_val, const T *vals, std::size_t numel) {
	CHECK_ARGUMENT(max_idx != nullptr,
				   "Argmax: max_idx pointer must not be null.");

	CHECK_ARGUMENT(max_val != nullptr,
				   "Argmax: max_val pointer must not be null.");

	CHECK_ARGUMENT(vals != nullptr,
				   "Argmax: vals pointer must not be null.");

	CHECK_ARGUMENT(numel > 0,
				   "Argmax: input tensor must not be empty.");

	CHECK_ARGUMENT(numel <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()),
				   "Argmax: input tensor is too large for int64 indices.");

	using AccumulatorType = accumulator_t<T>;

	// F32, F16, and BF16 all use float as the accumulator.
	static_assert(std::is_same_v<AccumulatorType, float>,
				  "Argmax: unsupported accumulator type.");

	MaxResult<AccumulatorType> result;

	const std::int64_t count = static_cast<std::int64_t>(numel);

	// Use multiple CPU threads only for sufficiently large tensors.
	// Each thread receives a private copy of result, and OpenMP merges
	// the partial results after the loop.
#pragma omp parallel for if (numel >= OPENMP_THRESHOLD) schedule(static) \
	reduction(llaisys_argmax_reduction : result)

	for (std::int64_t i = 0; i < count; ++i) {
		const AccumulatorType value = to_accumulator(vals[i]);

		if (!result.valid || is_better(value, i, result.value, result.index)) {
			result.value = value;
			result.index = i;
			result.valid = true;
		}
	}

	ASSERT(result.valid, "Argmax: failed to produce a valid result.");

	*max_idx = result.index;

	*max_val = from_accumulator<T>(result.value);
}

} // namespace

namespace llaisys::ops::cpu {

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, std::size_t numel) {
	return llaisys::device::cpu::dispatch_cpu_dtype(
		type,
		[&](auto tag) {
			using T = typename decltype(tag)::type;

			return argmax_impl<T>(
				reinterpret_cast<std::int64_t *>(max_idx),
				reinterpret_cast<T *>(max_val),
				reinterpret_cast<const T *>(vals),
				numel
			);
		}
	);
}

} // namespace llaisys::ops::cpu
#include "rearrange_cpu.hpp"

#include "../../../device/cpu/cpu_common.hpp"
#include "../../../device/cpu/cpu_dtype.hpp"
#include "../../../utils.hpp"
#include "../layout_utils.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <omp.h>
#include <utility>
#include <vector>

namespace {

using llaisys::device::cpu::OPENMP_THRESHOLD;

using llaisys::ops::rearrange_utils::layout_numel;
using llaisys::ops::rearrange_utils::validate_layout_common;
using llaisys::ops::rearrange_utils::is_contiguous_layout;
using llaisys::ops::rearrange_utils::is_non_overlapping_layout;
using llaisys::ops::rearrange_utils::ContiguousTail;
using llaisys::ops::rearrange_utils::find_common_contiguous_tail;

inline constexpr std::size_t CONTIGUOUS_TAIL_MIN_BYTES = 32;


// ============================================================
// Incremental logical-index cursor
// ============================================================
//
// Converting every flat index independently requires division and modulo
// operations for every dimension. This cursor performs that conversion only
// once at the beginning of a worker's range. Following offsets are generated
// by incrementing an odometer-like coordinate vector.
class OffsetCursor final {
public:
	OffsetCursor(
		const std::vector<std::size_t> &shape,
		const std::vector<std::ptrdiff_t> &strides,
		std::size_t dimension_count,
		std::size_t flat_index
	)
		: _shape(shape),
		  _strides(strides),
		  _coordinates(dimension_count, 0),
		  _dimension_count(dimension_count) {
		for (
			std::size_t dimension = dimension_count;
			dimension-- > 0;
		) {
			const std::size_t extent =
				_shape[dimension];

			const std::size_t coordinate =
				flat_index % extent;

			flat_index /= extent;

			_coordinates[dimension] =
				coordinate;

			_offset +=
				static_cast<std::ptrdiff_t>(
					coordinate
				)
				* _strides[dimension];
		}
	}

	std::ptrdiff_t offset() const {
		return _offset;
	}

	void advance() {
		for (
			std::size_t dimension = _dimension_count;
			dimension-- > 0;
		) {
			++_coordinates[dimension];

			_offset +=
				_strides[dimension];

			if (
				_coordinates[dimension]
				< _shape[dimension]
			) {
				return;
			}

			_coordinates[dimension] = 0;

			_offset -=
				static_cast<std::ptrdiff_t>(
					_shape[dimension]
				)
				* _strides[dimension];
		}
	}

private:
	const std::vector<std::size_t> &_shape;
	const std::vector<std::ptrdiff_t> &_strides;
	std::vector<std::size_t> _coordinates;
	std::size_t _dimension_count;
	std::ptrdiff_t _offset{0};
};

// ============================================================
// Parallel range helpers
// ============================================================

std::pair<std::size_t, std::size_t> worker_range(
	std::size_t work_items
) {
	const std::size_t thread_index =
		static_cast<std::size_t>(
			omp_get_thread_num()
		);

	const std::size_t thread_count =
		static_cast<std::size_t>(
			omp_get_num_threads()
		);

	const std::size_t begin =
		work_items * thread_index
		/ thread_count;

	const std::size_t end =
		work_items * (thread_index + 1)
		/ thread_count;

	return {
		begin,
		end,
	};
}

// ============================================================
// Fully contiguous path
// ============================================================

template <typename T>
void copy_fully_contiguous(
	T *out,
	const T *in,
	std::size_t numel
) {
	std::memmove(
		out,
		in,
		numel * sizeof(T)
	);
}

// ============================================================
// Contiguous-tail path
// ============================================================
//
// Example:
//
//     shape       = [batch, rows, columns]
//     out strides = [rows * columns, columns, 1]
//     in strides  = [padded_rows, padded_columns, 1]
//
// The last dimension can be copied with one optimized libc memory operation.
// Only the outer coordinates require stride calculations.
template <typename T>
void copy_contiguous_tail(
	T *out,
	const T *in,
	std::size_t numel,
	const std::vector<std::size_t> &shape,
	const std::vector<std::ptrdiff_t> &out_strides,
	const std::vector<std::ptrdiff_t> &in_strides,
	const ContiguousTail &tail,
	bool use_openmp
) {
	const std::size_t block_count =
		numel / tail.element_count;

	const std::size_t block_bytes =
		tail.element_count * sizeof(T);

#pragma omp parallel if(use_openmp)
	{
		const auto [begin, end] =
			worker_range(
				block_count
			);

		if (begin < end) {
			OffsetCursor out_cursor(
				shape,
				out_strides,
				tail.start_dimension,
				begin
			);

			OffsetCursor in_cursor(
				shape,
				in_strides,
				tail.start_dimension,
				begin
			);

			for (
				std::size_t block = begin;
				block < end;
				++block
			) {
				std::memmove(
					out + out_cursor.offset(),
					in + in_cursor.offset(),
					block_bytes
				);

				out_cursor.advance();
				in_cursor.advance();
			}
		}
	}
}

// ============================================================
// Generic incremental path
// ============================================================

template <typename T>
void copy_incrementally(
	T *out,
	const T *in,
	std::size_t numel,
	const std::vector<std::size_t> &out_shape,
	const std::vector<std::ptrdiff_t> &out_strides,
	const std::vector<std::size_t> &in_shape,
	const std::vector<std::ptrdiff_t> &in_strides,
	bool use_openmp
) {
#pragma omp parallel if(use_openmp)
	{
		const auto [begin, end] =
			worker_range(
				numel
			);

		if (begin < end) {
			OffsetCursor out_cursor(
				out_shape,
				out_strides,
				out_shape.size(),
				begin
			);

			OffsetCursor in_cursor(
				in_shape,
				in_strides,
				in_shape.size(),
				begin
			);

			for (
				std::size_t index = begin;
				index < end;
				++index
			) {
				out[out_cursor.offset()] =
					in[in_cursor.offset()];

				out_cursor.advance();
				in_cursor.advance();
			}
		}
	}
}

// ============================================================
// Typed implementation
// ============================================================

template <typename T>
void rearrange_typed(
	T *out,
	const T *in,
	std::size_t numel,
	const std::vector<std::size_t> &out_shape,
	const std::vector<std::ptrdiff_t> &out_strides,
	const std::vector<std::size_t> &in_shape,
	const std::vector<std::ptrdiff_t> &in_strides
) {
	validate_layout_common(
		out_shape,
		out_strides,
		numel,
		"Rearrange: output shape and stride counts must match.",
		"Rearrange: output shape does not match numel.",
		"Rearrange: output shape element count overflows size_t."
	);

	validate_layout_common(
		in_shape,
		in_strides,
		numel,
		"Rearrange: input shape and stride counts must match.",
		"Rearrange: input shape does not match numel.",
		"Rearrange: input shape element count overflows size_t."
	);

	CHECK_ARGUMENT(
		numel == 0 || out != nullptr,
		"Rearrange: output pointer must not be null."
	);

	CHECK_ARGUMENT(
		numel == 0 || in != nullptr,
		"Rearrange: input pointer must not be null."
	);

	if (numel == 0) {
		return;
	}

	CHECK_ARGUMENT(
		numel
			<= std::numeric_limits<std::size_t>::max()
				/ sizeof(T),
		"Rearrange: byte count overflows size_t."
	);

	// Exact same logical view: no data movement is required.
	if (
		out == in
		&& out_shape == in_shape
		&& out_strides == in_strides
	) {
		return;
	}

	// Fastest path: both logical layouts are physically contiguous.
	if (
		is_contiguous_layout(
			out_shape,
			out_strides
		)
		&& is_contiguous_layout(
			in_shape,
			in_strides
		)
	) {
		return copy_fully_contiguous(
			out,
			in,
			numel
		);
	}

	const bool output_is_non_overlapping =
		is_non_overlapping_layout(
			out_shape,
			out_strides
		);

	// Avoid parallel execution for potential in-place rearrangement.
	// Different logical layouts sharing the same pointer can have data
	// dependencies between iterations.
	const bool can_parallelize =
		output_is_non_overlapping
		&& out != in;

	const bool use_openmp =
		can_parallelize
		&& numel >= OPENMP_THRESHOLD;

	// When both layouts have the same logical shape, copy their shared
	// contiguous suffix in blocks instead of moving one element at a time.
	if (out_shape == in_shape) {
		const ContiguousTail tail =
			find_common_contiguous_tail(
				out_shape,
				out_strides,
				in_strides
			);

		if (
			tail.element_count > 1
			&& tail.element_count * sizeof(T)
				>= CONTIGUOUS_TAIL_MIN_BYTES
		) {
			return copy_contiguous_tail(
				out,
				in,
				numel,
				out_shape,
				out_strides,
				in_strides,
				tail,
				use_openmp
			);
		}
	}

	// General path for permutations, transposes, negative strides, and
	// layouts whose input and output shapes differ but have equal numel.
	copy_incrementally(
		out,
		in,
		numel,
		out_shape,
		out_strides,
		in_shape,
		in_strides,
		use_openmp
	);
}

} // namespace

namespace llaisys::ops::cpu {

void rearrange(
	std::byte *out,
	const std::byte *in,
	llaisysDataType_t type,
	std::size_t numel,
	const std::vector<std::size_t> &out_shape,
	const std::vector<std::ptrdiff_t> &out_strides,
	const std::vector<std::size_t> &in_shape,
	const std::vector<std::ptrdiff_t> &in_strides
) {
	return llaisys::device::cpu::dispatch_cpu_dtype(
		type,
		[&](auto tag) {
			using T = typename decltype(tag)::type;

			return rearrange_typed<T>(
				reinterpret_cast<T *>(out),
				reinterpret_cast<const T *>(in),
				numel,
				out_shape,
				out_strides,
				in_shape,
				in_strides
			);
		}
	);
}

} // namespace llaisys::ops::cpu
#include "rms_norm_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../utils.hpp"

#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <limits>

namespace {

using llaisys::device::nvidia::are_aligned;
using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::CUDA_WARP_SIZE;
using llaisys::device::nvidia::from_float;
using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::device::nvidia::Packed128;
using llaisys::device::nvidia::PACKED_128_ALIGNMENT;
using llaisys::device::nvidia::PACKED_128_ELEMENTS;
using llaisys::device::nvidia::to_float;

// CUB BlockReduce requires BLOCK_SIZE to be a compile-time constant.
// These sizes are RMSNorm-specific scheduling choices, so they remain here.
inline constexpr unsigned int SMALL_BLOCK_SIZE =
	static_cast<unsigned int>(CUDA_WARP_SIZE * 2);

inline constexpr unsigned int MEDIUM_BLOCK_SIZE =
	static_cast<unsigned int>(CUDA_WARP_SIZE * 4);

inline constexpr unsigned int LARGE_BLOCK_SIZE =
	static_cast<unsigned int>(CUDA_BLOCK_SIZE);

static_assert(
	SMALL_BLOCK_SIZE <= MEDIUM_BLOCK_SIZE
		&& MEDIUM_BLOCK_SIZE <= LARGE_BLOCK_SIZE,
	"RMSNorm: invalid block-size ordering."
);

// ============================================================
// Scalar kernel
// ============================================================
//
// Used for arbitrary pointer alignment and row widths.
//
// Each block processes one or more rows using a grid-stride loop.
// CUB performs the block-wide FP32 reduction. Normalization,
// multiplication by weight, and output conversion are fused into
// the same kernel.
template <typename T, unsigned int BLOCK_SIZE>
__global__ void rms_norm_scalar_kernel(
	T *out,
	const T *in,
	const T *weight,
	float eps,
	std::size_t nrow,
	std::size_t ncol
) {
	using BlockReduce =
		cub::BlockReduce<float, BLOCK_SIZE>;

	__shared__ typename BlockReduce::TempStorage reduction_storage;
	__shared__ float inverse_rms;

	for (
		std::size_t row = static_cast<std::size_t>(blockIdx.x);
		row < nrow;
		row += static_cast<std::size_t>(gridDim.x)
	) {
		const T *row_in =
			in + row * ncol;

		T *row_out =
			out + row * ncol;

		float thread_square_sum = 0.0F;

		for (
			std::size_t col =
				static_cast<std::size_t>(threadIdx.x);
			col < ncol;
			col += BLOCK_SIZE
		) {
			const float value =
				to_float<T>(row_in[col]);

			thread_square_sum =
				fmaf(
					value,
					value,
					thread_square_sum
				);
		}

		const float row_square_sum =
			BlockReduce(reduction_storage).Sum(
				thread_square_sum
			);

		if (threadIdx.x == 0) {
			const float mean_square =
				row_square_sum
				/ static_cast<float>(ncol);

			// Prefer sqrtf + division over rsqrtf for closer
			// numerical agreement with the reference path.
			inverse_rms =
				1.0F
				/ sqrtf(mean_square + eps);
		}

		__syncthreads();

		for (
			std::size_t col =
				static_cast<std::size_t>(threadIdx.x);
			col < ncol;
			col += BLOCK_SIZE
		) {
			const float input_value =
				to_float<T>(row_in[col]);

			const float weight_value =
				to_float<T>(weight[col]);

			row_out[col] =
				from_float<T>(
					input_value
					* weight_value
					* inverse_rms
				);
		}

		// The shared CUB workspace and inverse_rms are reused by
		// the next row handled by this block.
		__syncthreads();
	}
}

// ============================================================
// 128-bit packed kernel
// ============================================================
//
// Used only when:
// - out, in, and weight are 16-byte aligned;
// - each row contains an integral number of Packed128 values.
//
// Packed128 contains:
// - 4 float elements;
// - 8 half elements;
// - 8 BF16 elements.
template <typename T, unsigned int BLOCK_SIZE>
__global__ void rms_norm_packed_kernel(
	T *out,
	const T *in,
	const T *weight,
	float eps,
	std::size_t nrow,
	std::size_t ncol
) {
	using BlockReduce =
		cub::BlockReduce<float, BLOCK_SIZE>;

	constexpr std::size_t elements_per_pack =
		PACKED_128_ELEMENTS<T>;

	__shared__ typename BlockReduce::TempStorage reduction_storage;
	__shared__ float inverse_rms;

	const std::size_t pack_count =
		ncol / elements_per_pack;

	const Packed128 *packed_weight =
		reinterpret_cast<const Packed128 *>(weight);

	for (
		std::size_t row = static_cast<std::size_t>(blockIdx.x);
		row < nrow;
		row += static_cast<std::size_t>(gridDim.x)
	) {
		const T *row_in =
			in + row * ncol;

		T *row_out =
			out + row * ncol;

		const Packed128 *packed_in =
			reinterpret_cast<const Packed128 *>(row_in);

		Packed128 *packed_out =
			reinterpret_cast<Packed128 *>(row_out);

		float thread_square_sum = 0.0F;

		for (
			std::size_t pack_index =
				static_cast<std::size_t>(threadIdx.x);
			pack_index < pack_count;
			pack_index += BLOCK_SIZE
		) {
			const Packed128 input_pack =
				packed_in[pack_index];

			const T *input_values =
				reinterpret_cast<const T *>(
					&input_pack
				);

#pragma unroll
			for (
				std::size_t item = 0;
				item < elements_per_pack;
				++item
			) {
				const float value =
					to_float<T>(
						input_values[item]
					);

				thread_square_sum =
					fmaf(
						value,
						value,
						thread_square_sum
					);
			}
		}

		const float row_square_sum =
			BlockReduce(reduction_storage).Sum(
				thread_square_sum
			);

		if (threadIdx.x == 0) {
			const float mean_square =
				row_square_sum
				/ static_cast<float>(ncol);

			inverse_rms =
				1.0F
				/ sqrtf(mean_square + eps);
		}

		__syncthreads();

		for (
			std::size_t pack_index =
				static_cast<std::size_t>(threadIdx.x);
			pack_index < pack_count;
			pack_index += BLOCK_SIZE
		) {
			// Load complete input and weight packs before writing
			// the output pack. This keeps out == in safe.
			const Packed128 input_pack =
				packed_in[pack_index];

			const Packed128 weight_pack =
				packed_weight[pack_index];

			Packed128 output_pack{};

			const T *input_values =
				reinterpret_cast<const T *>(
					&input_pack
				);

			const T *weight_values =
				reinterpret_cast<const T *>(
					&weight_pack
				);

			T *output_values =
				reinterpret_cast<T *>(
					&output_pack
				);

#pragma unroll
			for (
				std::size_t item = 0;
				item < elements_per_pack;
				++item
			) {
				const float input_value =
					to_float<T>(
						input_values[item]
					);

				const float weight_value =
					to_float<T>(
						weight_values[item]
					);

				output_values[item] =
					from_float<T>(
						input_value
						* weight_value
						* inverse_rms
					);
			}

			packed_out[pack_index] =
				output_pack;
		}

		__syncthreads();
	}
}

// ============================================================
// Packed-path eligibility
// ============================================================

template <typename T>
bool can_use_packed_path(
	const T *out,
	const T *in,
	const T *weight,
	std::size_t ncol
) {
	constexpr std::size_t elements_per_pack =
		PACKED_128_ELEMENTS<T>;

	// ncol being a complete number of packs guarantees that
	// every following row also begins at a 16-byte boundary.
	return ncol % elements_per_pack == 0
		&& are_aligned<PACKED_128_ALIGNMENT>(
			out,
			in,
			weight
		);
}

// ============================================================
// Launch helpers
// ============================================================

template <typename T, unsigned int BLOCK_SIZE>
void launch_rms_norm_kernel(
	T *out,
	const T *in,
	const T *weight,
	float eps,
	std::size_t nrow,
	std::size_t ncol,
	cudaStream_t stream
) {
	// Each block initially owns one row. The kernels use a
	// row-level grid-stride loop, allowing the grid to be capped.
	const std::size_t grid_size =
		get_capped_grid_size(
			nrow,
			1
		);

	const dim3 grid(
		static_cast<unsigned int>(grid_size)
	);

	const dim3 block(BLOCK_SIZE);

	if (
		can_use_packed_path(
			out,
			in,
			weight,
			ncol
		)
	) {
		rms_norm_packed_kernel<
			T,
			BLOCK_SIZE
		><<<grid, block, 0, stream>>>(
			out,
			in,
			weight,
			eps,
			nrow,
			ncol
		);
	} else {
		rms_norm_scalar_kernel<
			T,
			BLOCK_SIZE
		><<<grid, block, 0, stream>>>(
			out,
			in,
			weight,
			eps,
			nrow,
			ncol
		);
	}

	CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_rms_norm(
	T *out,
	const T *in,
	const T *weight,
	float eps,
	std::size_t nrow,
	std::size_t ncol,
	cudaStream_t stream
) {
	CHECK_ARGUMENT(
		nrow == 0 || out != nullptr,
		"RMSNorm: output pointer must not be null."
	);

	CHECK_ARGUMENT(
		nrow == 0 || in != nullptr,
		"RMSNorm: input pointer must not be null."
	);

	CHECK_ARGUMENT(
		nrow == 0 || weight != nullptr,
		"RMSNorm: weight pointer must not be null."
	);

	if (nrow == 0) {
		return;
	}

	CHECK_ARGUMENT(
		ncol > 0,
		"RMSNorm: row width must be greater than zero."
	);

	CHECK_ARGUMENT(
		std::isfinite(eps) && eps >= 0.0F,
		"RMSNorm: epsilon must be finite and nonnegative."
	);

	CHECK_ARGUMENT(
		ncol
			<= std::numeric_limits<std::size_t>::max()
				/ nrow,
		"RMSNorm: tensor element count overflows size_t."
	);

	// Small rows need fewer warps. Wider rows expose enough
	// independent work to benefit from a full 256-thread block.
	if (ncol <= SMALL_BLOCK_SIZE) {
		return launch_rms_norm_kernel<
			T,
			SMALL_BLOCK_SIZE
		>(
			out,
			in,
			weight,
			eps,
			nrow,
			ncol,
			stream
		);
	}

	if (ncol <= 512) {
		return launch_rms_norm_kernel<
			T,
			MEDIUM_BLOCK_SIZE
		>(
			out,
			in,
			weight,
			eps,
			nrow,
			ncol,
			stream
		);
	}

	return launch_rms_norm_kernel<
		T,
		LARGE_BLOCK_SIZE
	>(
		out,
		in,
		weight,
		eps,
		nrow,
		ncol,
		stream
	);
}

} // namespace

namespace llaisys::ops::nvidia {

void rms_norm(
	std::byte *out,
	const std::byte *in,
	const std::byte *weight,
	float eps,
	llaisysDataType_t type,
	std::size_t nrow,
	std::size_t ncol,
	llaisysStream_t stream
) {

	const cudaStream_t cuda_stream =
		reinterpret_cast<cudaStream_t>(
			stream
		);

	switch (type) {
	case LLAISYS_DTYPE_F32:
		return launch_rms_norm<float>(
			reinterpret_cast<float *>(out),
			reinterpret_cast<const float *>(in),
			reinterpret_cast<const float *>(weight),
			eps,
			nrow,
			ncol,
			cuda_stream
		);

	case LLAISYS_DTYPE_F16:
		return launch_rms_norm<half>(
			reinterpret_cast<half *>(out),
			reinterpret_cast<const half *>(in),
			reinterpret_cast<const half *>(weight),
			eps,
			nrow,
			ncol,
			cuda_stream
		);

	case LLAISYS_DTYPE_BF16:
		return launch_rms_norm<__nv_bfloat16>(
			reinterpret_cast<__nv_bfloat16 *>(out),
			reinterpret_cast<const __nv_bfloat16 *>(in),
			reinterpret_cast<const __nv_bfloat16 *>(weight),
			eps,
			nrow,
			ncol,
			cuda_stream
		);

	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}

} // namespace llaisys::ops::nvidia

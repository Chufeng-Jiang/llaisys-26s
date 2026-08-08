#include "linear_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

namespace {

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::CUDA_DEFAULT_MAX_GRID_SIZE;
using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::utils::checked_product;


template <typename T>
__global__ void broadcast_bias_kernel(
	T *__restrict__ out,
	const T *__restrict__ bias,
	std::size_t output_elements,
	std::size_t output_features
) {
	const std::size_t first =
		static_cast<std::size_t>(blockIdx.x) * blockDim.x
		+ threadIdx.x;

	const std::size_t stride =
		static_cast<std::size_t>(gridDim.x) * blockDim.x;

	for (
		std::size_t index = first;
		index < output_elements;
		index += stride
	) {
		out[index] = bias[index % output_features];
	}
}

template <typename T>
void initialize_output(
	T *out,
	const T *bias,
	std::size_t output_elements,
	std::size_t output_features,
	cudaStream_t stream
) {
	if (output_elements == 0) {
		return;
	}

	if (bias == nullptr) {
		CHECK_ARGUMENT(
			output_elements
				<= std::numeric_limits<std::size_t>::max()
					/ sizeof(T),
			"Linear: output byte size overflows size_t."
		);

		CUDA_CHECK(
			cudaMemsetAsync(
				out,
				0,
				output_elements * sizeof(T),
				stream
			)
		);
		return;
	}

	const std::size_t grid_size =
		get_capped_grid_size(
			output_elements,
			CUDA_BLOCK_SIZE,
			CUDA_DEFAULT_MAX_GRID_SIZE
		);

	broadcast_bias_kernel<T>
		<<<
			static_cast<unsigned int>(grid_size),
			static_cast<unsigned int>(CUDA_BLOCK_SIZE),
			0,
			stream
		>>>(
			out,
			bias,
			output_elements,
			output_features
		);

	CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_linear(
	T *out,
	const T *in,
	const T *weight,
	const T *bias,
	cudaDataType_t data_type,
	std::size_t row_count,
	std::size_t output_features,
	std::size_t input_features,
	llaisysStream_t stream
) {
	const std::size_t output_elements =
		checked_product(
			row_count,
			output_features,
			"Linear: output element count overflows size_t."
		);

	const std::size_t input_elements =
		checked_product(
			row_count,
			input_features,
			"Linear: input element count overflows size_t."
		);

	const std::size_t weight_elements =
		checked_product(
			output_features,
			input_features,
			"Linear: weight element count overflows size_t."
		);

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

	constexpr std::size_t max_cublas_dimension =
		static_cast<std::size_t>(
			std::numeric_limits<int>::max()
		);

	CHECK_ARGUMENT(
		row_count <= max_cublas_dimension,
		"Linear: row count exceeds the cuBLAS int limit."
	);

	CHECK_ARGUMENT(
		output_features <= max_cublas_dimension,
		"Linear: output feature count exceeds the cuBLAS int limit."
	);

	CHECK_ARGUMENT(
		input_features <= max_cublas_dimension,
		"Linear: input feature count exceeds the cuBLAS int limit."
	);

	const cudaStream_t cuda_stream =
		reinterpret_cast<cudaStream_t>(stream);

	// K == 0 means the matrix product contributes zero.
	if (input_features == 0) {
		initialize_output(
			out,
			bias,
			output_elements,
			output_features,
			cuda_stream
		);
		return;
	}

	float beta = 0.0F;

	if (bias != nullptr) {
		initialize_output(
			out,
			bias,
			output_elements,
			output_features,
			cuda_stream
		);

		beta = 1.0F;
	}

	const float alpha = 1.0F;

	cublasHandle_t handle =
		llaisys::device::nvidia::get_cublas_handle(stream);

	// LLAISYS uses row-major tensors:
	//
	//   X: [M, K]
	//   W: [N, K]
	//   Y: [M, N]
	//   Y = X * W^T + bias
	//
	// cuBLAS interprets these same buffers as column-major matrices:
	//
	//   X memory -> X^T [K, M]
	//   W memory -> W^T [K, N]
	//   Y memory -> Y^T [N, M]
	//
	// Therefore calculate Y^T = W * X^T.
	CUBLAS_CHECK(
		cublasGemmEx(
			handle,
			CUBLAS_OP_T,
			CUBLAS_OP_N,
			static_cast<int>(output_features),
			static_cast<int>(row_count),
			static_cast<int>(input_features),
			&alpha,
			weight,
			data_type,
			static_cast<int>(input_features),
			in,
			data_type,
			static_cast<int>(input_features),
			&beta,
			out,
			data_type,
			static_cast<int>(output_features),
			CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT
		)
	);
}

} // namespace

namespace llaisys::ops::nvidia {

void linear(
	std::byte *out,
	const std::byte *in,
	const std::byte *weight,
	const std::byte *bias,
	llaisysDataType_t type,
	std::size_t nrow,
	std::size_t ncol_out,
	std::size_t ncol_in,
	llaisysStream_t stream
) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return launch_linear<float>(
			reinterpret_cast<float *>(out),
			reinterpret_cast<const float *>(in),
			reinterpret_cast<const float *>(weight),
			reinterpret_cast<const float *>(bias),
			CUDA_R_32F,
			nrow,
			ncol_out,
			ncol_in,
			stream
		);

	case LLAISYS_DTYPE_F16:
		return launch_linear<half>(
			reinterpret_cast<half *>(out),
			reinterpret_cast<const half *>(in),
			reinterpret_cast<const half *>(weight),
			reinterpret_cast<const half *>(bias),
			CUDA_R_16F,
			nrow,
			ncol_out,
			ncol_in,
			stream
		);

	case LLAISYS_DTYPE_BF16:
		return launch_linear<__nv_bfloat16>(
			reinterpret_cast<__nv_bfloat16 *>(out),
			reinterpret_cast<const __nv_bfloat16 *>(in),
			reinterpret_cast<const __nv_bfloat16 *>(weight),
			reinterpret_cast<const __nv_bfloat16 *>(bias),
			CUDA_R_16BF,
			nrow,
			ncol_out,
			ncol_in,
			stream
		);

	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}

} // namespace llaisys::ops::nvidia

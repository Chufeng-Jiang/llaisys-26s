#pragma once

#include "../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <utility>

namespace llaisys::device::nvidia {

// ============================================================
// CUDA type tag
// ============================================================

template <typename T>
struct CudaTypeTag {
	using type = T;
};

// ============================================================
// LLAISYS dtype -> CUDA C++ type dispatch
// ============================================================

template <typename Fn>
decltype(auto) dispatch_cuda_dtype(
	llaisysDataType_t type,
	Fn &&fn
) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return std::forward<Fn>(fn)(
			CudaTypeTag<float>{}
		);

	case LLAISYS_DTYPE_F16:
		return std::forward<Fn>(fn)(
			CudaTypeTag<half>{}
		);

	case LLAISYS_DTYPE_BF16:
		return std::forward<Fn>(fn)(
			CudaTypeTag<__nv_bfloat16>{}
		);

	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}

// ============================================================
// CUDA C++ type -> CUDA library data type
// ============================================================

template <typename T>
struct CudaTypeTraits;

template <>
struct CudaTypeTraits<float> {
	static constexpr cudaDataType_t data_type =
		CUDA_R_32F;
};

template <>
struct CudaTypeTraits<half> {
	static constexpr cudaDataType_t data_type =
		CUDA_R_16F;
};

template <>
struct CudaTypeTraits<__nv_bfloat16> {
	static constexpr cudaDataType_t data_type =
		CUDA_R_16BF;
};

} // namespace llaisys::device::nvidia

#pragma once

#if defined(ENABLE_NVIDIA_API) && defined(ENABLE_METAX_API)
#error "Only one CUDA-compatible GPU backend may be enabled per build."
#endif

#if defined(ENABLE_METAX_API)

#include <mc_common.h>
#include <mc_runtime.h>

#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>

#elif defined(ENABLE_NVIDIA_API)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#else

#error "A CUDA-compatible GPU backend must be enabled."

#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaisys::ops::cuda_compat {

// ============================================================
// CUDA-compatible scalar/vector type aliases
// ============================================================

using fp16_t = __half;
using fp16x2_t = __half2;

#ifdef ENABLE_METAX_API

using bf16_t = __maca_bfloat16;
using bf16x2_t = __maca_bfloat162;

#else

using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;

#endif

// ============================================================
// Compile-time ABI checks
// ============================================================

static_assert(
	sizeof(fp16_t) == 2,
	"CUDA-compatible FP16 type must occupy 2 bytes."
);

static_assert(
	sizeof(fp16x2_t) == 4,
	"CUDA-compatible FP16x2 type must occupy 4 bytes."
);

static_assert(
	sizeof(bf16_t) == 2,
	"CUDA-compatible BF16 type must occupy 2 bytes."
);

static_assert(
	sizeof(bf16x2_t) == 4,
	"CUDA-compatible BF16x2 type must occupy 4 bytes."
);

static_assert(
	sizeof(float4) == 16,
	"CUDA-compatible float4 must occupy 16 bytes."
);

static_assert(
	sizeof(uint4) == 16,
	"CUDA-compatible uint4 must occupy 16 bytes."
);

// ============================================================
// Template utilities
// ============================================================

template <typename>
inline constexpr bool DEPENDENT_FALSE = false;

// ============================================================
// Integer helpers
// ============================================================

__host__ __device__
constexpr std::size_t div_ceil(
	std::size_t value,
	std::size_t divisor
) {
	return value / divisor
		+ static_cast<std::size_t>(
			value % divisor != 0
		);
}

// ============================================================
// CUDA-compatible data-type conversion
// ============================================================

template <typename T>
__device__ __forceinline__
float to_float(
	T value
) {
	if constexpr (
		std::is_same_v<T, float>
	) {
		return value;
	} else if constexpr (
		std::is_same_v<T, fp16_t>
	) {
		return __half2float(
			value
		);
	} else if constexpr (
		std::is_same_v<T, bf16_t>
	) {
		return __bfloat162float(
			value
		);
	} else {
		static_assert(
			DEPENDENT_FALSE<T>,
			"Unsupported CUDA-compatible type for conversion to float."
		);
	}
}

template <typename T>
__device__ __forceinline__
T from_float(
	float value
) {
	if constexpr (
		std::is_same_v<T, float>
	) {
		return value;
	} else if constexpr (
		std::is_same_v<T, fp16_t>
	) {
		return __float2half(
			value
		);
	} else if constexpr (
		std::is_same_v<T, bf16_t>
	) {
		return __float2bfloat16(
			value
		);
	} else {
		static_assert(
			DEPENDENT_FALSE<T>,
			"Unsupported CUDA-compatible type for conversion from float."
		);
	}
}

// ============================================================
// Address-alignment helpers
// ============================================================

template <
	std::size_t Alignment,
	typename T
>
inline bool is_aligned(
	const T *pointer
) {
	static_assert(
		Alignment > 0
			&& (Alignment & (Alignment - 1)) == 0,
		"Alignment must be a nonzero power of two."
	);

	const std::uintptr_t address =
		reinterpret_cast<std::uintptr_t>(
			pointer
		);

	return address % Alignment == 0;
}

template <
	std::size_t Alignment,
	typename... PointerTypes
>
inline bool are_aligned(
	PointerTypes... pointers
) {
	return (
		is_aligned<Alignment>(
			pointers
		)
		&& ...
	);
}

// ============================================================
// 128-bit raw-memory pack
// ============================================================

using Packed128 = uint4;

inline constexpr std::size_t PACKED_128_BYTES =
	sizeof(Packed128);

inline constexpr std::size_t PACKED_128_ALIGNMENT =
	alignof(Packed128);

static_assert(
	PACKED_128_BYTES == 16,
	"Packed128 must occupy exactly 16 bytes."
);

static_assert(
	PACKED_128_ALIGNMENT == 16,
	"Packed128 must require 16-byte alignment."
);

template <typename T>
inline constexpr std::size_t PACKED_128_ELEMENTS =
	PACKED_128_BYTES / sizeof(T);

} // namespace llaisys::ops::cuda_compat
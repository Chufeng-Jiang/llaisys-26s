#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace llaisys::ops::cuda_compat {

// ============================================================
// Template utilities
// ============================================================

template <typename>
inline constexpr bool DEPENDENT_FALSE = false;

// ============================================================
// Integer helpers
// ============================================================

__host__ __device__ constexpr std::size_t div_ceil(
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
//
// These helpers operate only on CUDA-compatible scalar types.
// They do not depend on an NVIDIA Runtime API.
//
// Future CUDA-compatible toolchains can compile the same code
// if they provide compatible float / half / BF16 types.
// ============================================================

template <typename T>
__device__ __forceinline__ float to_float(
	T value
) {
	if constexpr (
		std::is_same_v<T, float>
	) {
		return value;
	} else if constexpr (
		std::is_same_v<T, half>
	) {
		return __half2float(value);
	} else if constexpr (
		std::is_same_v<T, __nv_bfloat16>
	) {
		return __bfloat162float(value);
	} else {
		static_assert(
			DEPENDENT_FALSE<T>,
			"Unsupported CUDA-compatible type for conversion to float."
		);
	}
}

template <typename T>
__device__ __forceinline__ T from_float(
	float value
) {
	if constexpr (
		std::is_same_v<T, float>
	) {
		return value;
	} else if constexpr (
		std::is_same_v<T, half>
	) {
		return __float2half(value);
	} else if constexpr (
		std::is_same_v<T, __nv_bfloat16>
	) {
		return __float2bfloat16(value);
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
		is_aligned<Alignment>(pointers)
		&& ...
	);
}

// ============================================================
// 128-bit raw-memory pack
//
// uint4 is part of the CUDA-compatible source layer.
//
// This is a memory-access representation only. It does not
// depend on CUDA Runtime device management.
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
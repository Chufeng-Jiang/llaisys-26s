#pragma once

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cuda_compat {

// ============================================================
// Compile-time helper
// ============================================================

template <typename>
inline constexpr bool DEPENDENT_FALSE = false;

// ============================================================
// Pointer alignment helpers
//
// These are not NVIDIA-specific.
//
// MetaX, Iluvatar and Moore Threads can reuse the same logic.
// ============================================================

template <std::size_t Alignment>
inline bool is_aligned(
	const void *pointer
) {
	if (pointer == nullptr) {
		return false;
	}

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

} // namespace llaisys::ops::cuda_compat
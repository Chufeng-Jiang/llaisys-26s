#pragma once

#include "../../utils.hpp"

#include <utility>

namespace llaisys::device::cpu {

template <typename T>
struct CpuTypeTag {
	using type = T;
};

template <typename Fn>
decltype(auto) dispatch_cpu_dtype(
	llaisysDataType_t type,
	Fn &&fn
) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return std::forward<Fn>(fn)(
			CpuTypeTag<float>{}
		);

	case LLAISYS_DTYPE_F16:
		return std::forward<Fn>(fn)(
			CpuTypeTag<llaisys::fp16_t>{}
		);

	case LLAISYS_DTYPE_BF16:
		return std::forward<Fn>(fn)(
			CpuTypeTag<llaisys::bf16_t>{}
		);

	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}

} // namespace llaisys::device::cpu
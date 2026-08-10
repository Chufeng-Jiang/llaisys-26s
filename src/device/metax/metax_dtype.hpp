#pragma once

#include "llaisys.h"

#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>

#include <stdexcept>
#include <utility>

namespace llaisys::device::metax {

template <typename T>
struct DTypeTag {
	using type = T;
};

template <typename Function>
decltype(auto) dispatch_metax_dtype(
	llaisysDataType_t type,
	Function &&function
) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return std::forward<Function>(
			function
		)(
			DTypeTag<float>{}
		);

	case LLAISYS_DTYPE_F16:
		return std::forward<Function>(
			function
		)(
			DTypeTag<__half>{}
		);

	case LLAISYS_DTYPE_BF16:
		return std::forward<Function>(
			function
		)(
			DTypeTag<__maca_bfloat16>{}
		);

	default:
		throw std::invalid_argument(
			"Unsupported MetaX data type."
		);
	}
}

} // namespace llaisys::device::metax

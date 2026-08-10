#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::metax {

// Apply non-interleaved rotary position embedding:
//
//     input/output shape: [seqlen, nhead, d]
//     position ids shape: [seqlen]
//
// F32, F16, and BF16 are supported.
// Arithmetic is performed in FP32.
void rope(
	std::byte *out,
	const std::byte *in,
	const std::byte *pos_ids,
	float theta,
	llaisysDataType_t type,
	std::size_t seqlen,
	std::size_t nhead,
	std::size_t d,
	llaisysStream_t stream
);

} // namespace llaisys::ops::metax
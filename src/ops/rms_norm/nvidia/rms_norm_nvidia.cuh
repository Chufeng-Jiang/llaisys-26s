#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

// Compute RMS normalization over each contiguous row:
//
//   out[row, col] =
//       in[row, col] * weight[col]
//       / sqrt(mean(in[row, :]^2) + eps)
//
// FP16 and BF16 inputs are accumulated in FP32.
// The kernel is launched on the supplied LLAISYS Runtime stream.
void rms_norm(
	std::byte *out,
	const std::byte *in,
	const std::byte *weight,
	float eps,
	llaisysDataType_t type,
	std::size_t nrow,
	std::size_t ncol,
	llaisysStream_t stream
);

} // namespace llaisys::ops::nvidia

#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void add(
	std::byte *c,
	const std::byte *a,
	const std::byte *b,
	llaisysDataType_t type,
	std::size_t numel,
	llaisysStream_t stream
);

} // namespace llaisys::ops::nvidia
#pragma once

#include "../../utils/check.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace llaisys::ops::rearrange_utils {

using llaisys::utils::checked_product;

inline std::size_t
layout_numel(const std::vector<std::size_t> &shape, const char *overflow_message) {
    std::size_t result = 1;

    for (const std::size_t extent : shape) {
        if (extent == 0) { return 0; }

        result = checked_product(result, extent, overflow_message);
    }

    return result;
}

inline void validate_layout_common(
    const std::vector<std::size_t> &shape,
    const std::vector<std::ptrdiff_t> &strides,
    std::size_t expected_numel,
    const char *metadata_message,
    const char *numel_message,
    const char *overflow_message) {
    CHECK_ARGUMENT(shape.size() == strides.size(), metadata_message);

    CHECK_ARGUMENT(layout_numel(shape, overflow_message) == expected_numel, numel_message);
}

inline bool is_contiguous_layout(
    const std::vector<std::size_t> &shape, const std::vector<std::ptrdiff_t> &strides) {
    std::size_t expected_stride = 1;

    for (std::size_t dimension = shape.size(); dimension-- > 0;) {
        const std::size_t extent = shape[dimension];

        if (extent == 0) { return true; }

        if (extent != 1 && strides[dimension] != static_cast<std::ptrdiff_t>(expected_stride)) {
            return false;
        }

        if (expected_stride > std::numeric_limits<std::size_t>::max() / extent) { return false; }

        expected_stride *= extent;
    }

    return true;
}

inline std::size_t absolute_stride(std::ptrdiff_t stride) {
    if (stride >= 0) { return static_cast<std::size_t>(stride); }

    return static_cast<std::size_t>(-(stride + 1)) + 1;
}

inline bool is_non_overlapping_layout(
    const std::vector<std::size_t> &shape, const std::vector<std::ptrdiff_t> &strides) {
    struct DimensionInfo {
        std::size_t stride;
        std::size_t extent;
    };

    std::vector<DimensionInfo> dimensions;
    dimensions.reserve(shape.size());

    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        const std::size_t extent = shape[dimension];

        if (extent <= 1) { continue; }

        const std::size_t stride = absolute_stride(strides[dimension]);

        if (stride == 0) { return false; }

        dimensions.push_back(
            DimensionInfo{
                stride,
                extent,
            });
    }

    std::sort(
        dimensions.begin(),
        dimensions.end(),
        [](const DimensionInfo &left, const DimensionInfo &right) {
            return left.stride < right.stride;
        });

    std::size_t occupied_span = 0;

    for (const DimensionInfo &dimension : dimensions) {
        if (dimension.stride <= occupied_span) { return false; }

        const std::size_t repetitions = dimension.extent - 1;

        if (repetitions != 0
            && dimension.stride
                   > (std::numeric_limits<std::size_t>::max() - occupied_span) / repetitions) {
            return false;
        }

        occupied_span += repetitions * dimension.stride;
    }

    return true;
}

// ============================================================
// Common contiguous tail
// ============================================================

struct ContiguousTail {
    std::size_t start_dimension;
    std::size_t element_count;
};

inline ContiguousTail find_common_contiguous_tail(
    const std::vector<std::size_t> &shape,
    const std::vector<std::ptrdiff_t> &out_strides,
    const std::vector<std::ptrdiff_t> &in_strides) {
    ContiguousTail result{
        shape.size(),
        1,
    };

    std::size_t expected_stride = 1;

    for (std::size_t dimension = shape.size(); dimension-- > 0;) {
        const std::size_t extent = shape[dimension];

        if (extent == 1) {
            result.start_dimension = dimension;
            continue;
        }

        if (out_strides[dimension] != static_cast<std::ptrdiff_t>(expected_stride)
            || in_strides[dimension] != static_cast<std::ptrdiff_t>(expected_stride)) {
            break;
        }

        if (expected_stride > std::numeric_limits<std::size_t>::max() / extent) { break; }

        expected_stride *= extent;

        result.start_dimension = dimension;

        result.element_count = expected_stride;
    }

    return result;
}

} // namespace llaisys::ops::rearrange_utils
#include "rms_norm_cpu.hpp"

#include "sdot.hpp"

#include "../../../device/cpu/cpu_common.hpp"
#include "../../../device/cpu/cpu_dtype.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

using llaisys::device::cpu::OPENMP_THRESHOLD;

// Converting a shared low-precision weight once becomes worthwhile when it is
// reused by several rows. For decode-like M == 1 calls, avoiding the allocation
// is generally more important than avoiding one conversion pass.
inline constexpr std::size_t PRECONVERT_WEIGHT_MIN_ROWS = 4;

template <typename T>
inline constexpr bool IS_LOW_PRECISION
    = std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>;

template <typename T> float to_float(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else {
        static_assert(IS_LOW_PRECISION<T>, "RMSNorm: unsupported conversion source type.");

        return llaisys::utils::cast<float>(value);
    }
}

template <typename T> T from_float(float value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else {
        static_assert(IS_LOW_PRECISION<T>, "RMSNorm: unsupported conversion destination type.");

        return llaisys::utils::cast<T>(value);
    }
}

template <typename T> float calculate_square_sum(const T *input, std::size_t count) {
    if constexpr (std::is_same_v<T, float>) {
        return llaisys::ops::cpu::sdot(input, input, count);
    } else {
        float sum0 = 0.0F;
        float sum1 = 0.0F;
        float sum2 = 0.0F;
        float sum3 = 0.0F;

        std::size_t i = 0;

        // Independent accumulators shorten the dependency chain. The conversion
        // still follows the project's canonical utils::cast<T>() path.
        for (; i + 4 <= count; i += 4) {
            const float value0 = to_float(input[i]);
            const float value1 = to_float(input[i + 1]);
            const float value2 = to_float(input[i + 2]);
            const float value3 = to_float(input[i + 3]);

            sum0 += value0 * value0;
            sum1 += value1 * value1;
            sum2 += value2 * value2;
            sum3 += value3 * value3;
        }

        float sum = (sum0 + sum1) + (sum2 + sum3);

        for (; i < count; ++i) {
            const float value = to_float(input[i]);
            sum += value * value;
        }

        return sum;
    }
}

template <typename T>
void apply_rms_norm(
    T *out,
    const T *in,
    const T *weight,
    const float *converted_weight,
    float inverse_rms,
    std::size_t count) {
    if constexpr (std::is_same_v<T, float>) {
#pragma omp simd
        for (std::size_t i = 0; i < count; ++i) { out[i] = in[i] * weight[i] * inverse_rms; }
    } else if (converted_weight != nullptr) {
        for (std::size_t i = 0; i < count; ++i) {
            const float value = to_float(in[i]) * converted_weight[i] * inverse_rms;

            out[i] = from_float<T>(value);
        }
    } else {
        // Decode-like single-row calls avoid allocating a temporary FP32
        // weight.
        for (std::size_t i = 0; i < count; ++i) {
            const float value = to_float(in[i]) * to_float(weight[i]) * inverse_rms;

            out[i] = from_float<T>(value);
        }
    }
}

template <typename T>
void rms_norm_impl(
    T *out,
    const T *in,
    const T *weight,
    float eps,
    std::size_t row_count,
    std::size_t column_count) {
    if (row_count == 0) { return; }

    CHECK_ARGUMENT(column_count > 0, "RMSNorm: column count must be greater than zero.");

    CHECK_ARGUMENT(
        row_count <= std::numeric_limits<std::size_t>::max() / column_count,
        "RMSNorm: tensor element count overflow.");

    const std::size_t element_count = row_count * column_count;

    CHECK_ARGUMENT(out != nullptr, "RMSNorm: output pointer must not be null.");

    CHECK_ARGUMENT(in != nullptr, "RMSNorm: input pointer must not be null.");

    CHECK_ARGUMENT(weight != nullptr, "RMSNorm: weight pointer must not be null.");

    CHECK_ARGUMENT(eps >= 0.0F, "RMSNorm: epsilon must not be negative.");

    std::vector<float> converted_weight_storage;
    const float *converted_weight = nullptr;

    if constexpr (IS_LOW_PRECISION<T>) {
        if (row_count >= PRECONVERT_WEIGHT_MIN_ROWS) {
            converted_weight_storage.resize(column_count);

#pragma omp simd
            for (std::size_t column = 0; column < column_count; ++column) {
                converted_weight_storage[column] = to_float(weight[column]);
            }

            converted_weight = converted_weight_storage.data();
        }
    }

    const bool use_openmp = element_count >= OPENMP_THRESHOLD && row_count > 1;

#pragma omp parallel for if (use_openmp) schedule(static)
    for (std::size_t row = 0; row < row_count; ++row) {
        T *out_row = out + row * column_count;

        const T *input_row = in + row * column_count;

        const float square_sum = calculate_square_sum(input_row, column_count);

        const float mean_square = square_sum / static_cast<float>(column_count);

        // Compute the reciprocal once. The output loop then uses multiplication
        // instead of one division per element.
        const float inverse_rms = 1.0F / std::sqrt(mean_square + eps);

        apply_rms_norm(out_row, input_row, weight, converted_weight, inverse_rms, column_count);
    }
}

} // namespace

namespace llaisys::ops::cpu {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    float eps,
    llaisysDataType_t type,
    std::size_t nrow,
    std::size_t ncol) {
    return llaisys::device::cpu::dispatch_cpu_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return rms_norm_impl<T>(
            reinterpret_cast<T *>(out),
            reinterpret_cast<const T *>(in),
            reinterpret_cast<const T *>(weight),
            eps,
            nrow,
            ncol);
    });
}

} // namespace llaisys::ops::cpu

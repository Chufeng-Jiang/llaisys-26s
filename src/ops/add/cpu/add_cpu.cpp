#include <cstddef>
#include <type_traits>

#include "../../../device/cpu/cpu_common.hpp"
#include "../../../device/cpu/cpu_dtype.hpp"
#include "../../../utils.hpp"
#include "add_cpu.hpp"

namespace {

using llaisys::device::cpu::OPENMP_THRESHOLD;

/**
 * @brief Performs element-wise addition of two contiguous arrays on CPU.
 * For sufficiently large tensors, OpenMP is used to divide the loop
 * iterations among multiple CPU threads. Each iteration is independent,
 * so different threads can safely compute different output elements.
 *
 * For FP16 and BF16 values, each input element is first converted to
 * float32, the addition is performed in float32, and the result is then
 * converted back to the original data type.
 *
 * This avoids relying on native C++ arithmetic operators for the custom
 * FP16/BF16 storage types and performs the arithmetic with higher
 * intermediate precision.
 *
 * @tparam T Element data type.
 * @param c Output array.
 * @param a First input array.
 * @param b Second input array.
 * @param numel Number of elements to process.
 */
template <typename T> void add_(T *c, const T *a, const T *b, std::size_t numel) {
    CHECK_ARGUMENT(numel == 0 || c != nullptr, "Add: output pointer must not be null.");
    CHECK_ARGUMENT(numel == 0 || a != nullptr, "Add: input pointer a must not be null.");
    CHECK_ARGUMENT(numel == 0 || b != nullptr, "Add: input pointer b must not be null.");

#pragma omp parallel for if (numel >= OPENMP_THRESHOLD) schedule(static)
    for (std::size_t i = 0; i < numel; ++i) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            const float a_value = llaisys::utils::cast<float>(a[i]);
            const float b_value = llaisys::utils::cast<float>(b[i]);
            c[i] = llaisys::utils::cast<T>(a_value + b_value);
        } else {
            c[i] = a[i] + b[i];
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    std::size_t numel) {
    return llaisys::device::cpu::dispatch_cpu_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return add_<T>(
            reinterpret_cast<T *>(c), reinterpret_cast<const T *>(a),
            reinterpret_cast<const T *>(b), numel);
    });
}

} // namespace llaisys::ops::cpu
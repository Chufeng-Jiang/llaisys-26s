#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "../../../device/cpu/cpu_common.hpp"
#include "../../../device/cpu/cpu_dtype.hpp"
#include "../../../utils.hpp"
#include "embedding_cpu.hpp"

namespace {

using llaisys::device::cpu::EMBEDDING_OPENMP_MIN_BYTES;
using llaisys::device::cpu::EMBEDDING_OPENMP_MIN_ROWS;

// Copy the embedding row selected by each index into the output.
//
// Example:
//
//     index = [2, 0]
//     len = 4
//
//     weight:
//         row 0: [a, b, c, d]
//         row 1: [e, f, g, h]
//         row 2: [i, j, k, l]
//
//     output:
//         [i, j, k, l,
//          a, b, c, d]
template <typename T>
void embedding_impl(
    T *out,
    const std::int64_t *index,
    const T *weight,
    std::size_t numel,
    std::size_t len,
    std::size_t vocabulary_size) {
    CHECK_ARGUMENT(len > 0, "Embedding: embedding length must be greater than zero.");

    CHECK_ARGUMENT(
        numel % len == 0,
        "Embedding: output element count must be divisible by embedding "
        "length.");

    CHECK_ARGUMENT(numel == 0 || out != nullptr, "Embedding: output pointer must not be null.");

    CHECK_ARGUMENT(numel == 0 || index != nullptr, "Embedding: index pointer must not be null.");

    CHECK_ARGUMENT(numel == 0 || weight != nullptr, "Embedding: weight pointer must not be null.");

    CHECK_ARGUMENT(
        numel == 0 || vocabulary_size > 0, "Embedding: vocabulary size must be greater than zero.");

    CHECK_ARGUMENT(
        len <= std::numeric_limits<std::size_t>::max() / sizeof(T),
        "Embedding: embedding row size overflows size_t.");

    CHECK_ARGUMENT(
        numel <= std::numeric_limits<std::size_t>::max() / sizeof(T),
        "Embedding: output byte size overflows size_t.");

    CHECK_ARGUMENT(
        vocabulary_size <= std::numeric_limits<std::size_t>::max() / len,
        "Embedding: weight element count overflows size_t.");

    const std::size_t index_count = numel / len;
    const std::size_t row_bytes = len * sizeof(T);
    const std::size_t total_bytes = numel * sizeof(T);

    // Validate indices before entering the OpenMP region.
    //
    // Checking index[i] before converting it to size_t is important:
    // converting a negative int64_t directly to size_t would produce
    // a very large positive value.
    for (std::size_t i = 0; i < index_count; ++i) {
        const std::int64_t raw_index = index[i];

        CHECK_ARGUMENT(raw_index >= 0, "Embedding: index must not be negative.");

        CHECK_ARGUMENT(
            static_cast<std::uint64_t>(raw_index) < static_cast<std::uint64_t>(vocabulary_size),
            "Embedding: index is out of vocabulary range.");
    }

    const bool use_openmp
        = index_count >= EMBEDDING_OPENMP_MIN_ROWS && total_bytes >= EMBEDDING_OPENMP_MIN_BYTES;

#pragma omp parallel for if (use_openmp) schedule(static)
    for (std::size_t i = 0; i < index_count; ++i) {
        const std::size_t row_index = static_cast<std::size_t>(index[i]);

        const std::size_t output_offset = i * len;
        const std::size_t weight_offset = row_index * len;

        std::memcpy(out + output_offset, weight + weight_offset, row_bytes);
    }
}

} // namespace

namespace llaisys::ops::cpu {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    std::size_t numel,
    std::size_t len,
    std::size_t vocabulary_size) {
    return llaisys::device::cpu::dispatch_cpu_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return embedding_impl<T>(
            reinterpret_cast<T *>(out),
            reinterpret_cast<const std::int64_t *>(index),
            reinterpret_cast<const T *>(weight),
            numel,
            len,
            vocabulary_size);
    });
}

} // namespace llaisys::ops::cpu
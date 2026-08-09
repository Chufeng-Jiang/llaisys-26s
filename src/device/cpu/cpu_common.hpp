#pragma once

#include <cstddef>

namespace llaisys::device::cpu {

// Minimum number of elements required to enable OpenMP for
// element-wise and reduction operators such as Add and Argmax.
inline constexpr std::size_t OPENMP_THRESHOLD = 32768;

// Embedding parallelizes over index rows rather than individual elements.
// At least this many embedding rows must be available before OpenMP parallel
// execution is considered.
inline constexpr std::size_t EMBEDDING_OPENMP_MIN_ROWS = 64;

// OpenMP is enabled only when the total amount of copied embedding data
// is sufficiently large to offset thread scheduling and synchronization
// overhead.
inline constexpr std::size_t EMBEDDING_OPENMP_MIN_BYTES = 256 * 1024;

} // namespace llaisys::device::cpu
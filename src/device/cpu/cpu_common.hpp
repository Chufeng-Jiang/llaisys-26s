#pragma once

#include <cstddef>

namespace llaisys::device::cpu {

// Minimum number of elements required to enable OpenMP parallel execution.
// OpenMP thread creation and synchronization introduce overhead,
// so small workloads are executed using a single thread.
inline constexpr std::size_t OPENMP_THRESHOLD = 32768;

} // namespace llaisys::device::cpu
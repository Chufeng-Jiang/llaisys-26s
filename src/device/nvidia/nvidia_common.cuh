#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

// Number of threads in one CUDA block.
//
// One CUDA warp normally contains 32 threads.
// Therefore, 256 threads correspond to 8 warps.
inline constexpr std::size_t CUDA_BLOCK_SIZE = 256;

// Integer ceiling division.
//
// Examples:
// div_ceil(10, 4) == 3
// div_ceil(8, 4)  == 2
//
// The divisor b must be greater than zero.
__host__ __device__ constexpr std::size_t div_ceil(std::size_t a,
                                                   std::size_t b) {
  return a / b + static_cast<std::size_t>(a % b != 0);
}

// Check the return value of a CUDA Runtime API call.
inline void check_cuda(cudaError_t error, const char *expression,
                       const char *file, int line, const char *function) {
  if (error == cudaSuccess) {
    return;
  }

  throw std::runtime_error(std::string("CUDA error: ") +
                           cudaGetErrorString(error) + " while executing " +
                           expression + " in " + function + " at " + file +
                           ":" + std::to_string(line));
}

}  // namespace llaisys::device::nvidia

#define CUDA_CHECK(CALL) ::llaisys::device::nvidia::check_cuda((CALL), #CALL, __FILE__, __LINE__, __func__)
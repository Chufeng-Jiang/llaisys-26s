#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::device::nvidia {

// ============================================================
// CUDA C++ type -> CUDA library data type
// ============================================================

template <typename T> struct CudaTypeTraits;

template <> struct CudaTypeTraits<float> {
    static constexpr cudaDataType_t data_type = CUDA_R_32F;
};

template <> struct CudaTypeTraits<half> {
    static constexpr cudaDataType_t data_type = CUDA_R_16F;
};

template <> struct CudaTypeTraits<__nv_bfloat16> {
    static constexpr cudaDataType_t data_type = CUDA_R_16BF;
};

} // namespace llaisys::device::nvidia

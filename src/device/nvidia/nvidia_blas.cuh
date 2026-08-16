#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace llaisys::device::nvidia {

const char *cublas_status_name(cublasStatus_t status) noexcept;

void check_cublas(
    cublasStatus_t status,
    const char *expression,
    const char *file,
    int line,
    const char *function);

cublasHandle_t get_cublas_handle(cudaStream_t stream);

} // namespace llaisys::device::nvidia

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(CALL)                                                                         \
    ::llaisys::device::nvidia::check_cublas((CALL), #CALL, __FILE__, __LINE__, __func__)
#endif
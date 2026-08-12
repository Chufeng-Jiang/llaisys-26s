#include "linear_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../../utils.hpp"
#include "../../cuda_compat/common.cuh"
#include "../cuda_compat/linear_cuda_compat.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

namespace {

namespace cuda_compat = llaisys::ops::cuda_compat;

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::CUDA_DEFAULT_MAX_GRID_SIZE;
using llaisys::device::nvidia::to_cuda_stream;
using llaisys::ops::cuda_compat::get_capped_grid_size;
using llaisys::utils::checked_product;

// ============================================================
// NVIDIA output initialization
// ============================================================
//
// Linear computes:
//
//     Y = X W^T + bias
//
// cuBLAS beta can scale an existing C matrix, but it does not
// directly broadcast a one-dimensional bias vector.
//
// Therefore:
//
// bias != nullptr:
//     initialize Y by broadcasting bias,
//     then GEMM uses beta = 1.
//
// bias == nullptr:
//     GEMM uses beta = 0, so initialization is unnecessary.
//
// The input_features == 0 special case is different because no
// GEMM is executed. In that case the output must explicitly be:
//
//     bias broadcast
//
// or:
//
//     zero
// ============================================================

template <typename T>
void initialize_linear_output(
    T *out,
    const T *bias,
    std::size_t output_elements,
    std::size_t output_features,
    cudaStream_t stream) {
    if (output_elements == 0) { return; }

    if (bias == nullptr) {
        CHECK_ARGUMENT(
            output_elements <= std::numeric_limits<std::size_t>::max() / sizeof(T),
            "Linear: output byte size overflows size_t.");

        CUDA_CHECK(cudaMemsetAsync(out, 0, output_elements * sizeof(T), stream));

        return;
    }

    // ========================================================
    // NVIDIA-specific scheduling
    // ========================================================

    const std::size_t grid_size
        = get_capped_grid_size(output_elements, CUDA_BLOCK_SIZE, CUDA_DEFAULT_MAX_GRID_SIZE);

    // ========================================================
    // Shared CUDA-compatible bias broadcast
    // ========================================================

    cuda_compat::launch_linear_bias_broadcast<T>(
        out, bias, output_elements, output_features, static_cast<unsigned int>(CUDA_BLOCK_SIZE),
        grid_size, stream);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// NVIDIA Linear implementation
// ============================================================
//
// GEMM is deliberately vendor-library-backed.
//
// NVIDIA:
//     cuBLAS
//
// Future backends:
//
//     MetaX       -> vendor BLAS
//     Iluvatar    -> vendor BLAS
//     MThreads    -> muBLAS / vendor BLAS
//
// We do not implement a portable handwritten GEMM merely to
// make the source tree look uniform.
// ============================================================

template <typename T>
void launch_nvidia_linear(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    std::size_t row_count,
    std::size_t output_features,
    std::size_t input_features,
    cudaStream_t stream) {
    // ========================================================
    // Element-count validation
    // ========================================================

    const std::size_t output_elements = checked_product(
        row_count, output_features, "Linear: output element count overflows size_t.");

    const std::size_t input_elements = checked_product(
        row_count, input_features, "Linear: input element count overflows size_t.");

    const std::size_t weight_elements = checked_product(
        output_features, input_features, "Linear: weight element count overflows size_t.");

    CHECK_ARGUMENT(
        output_elements == 0 || out != nullptr, "Linear: output pointer must not be null.");

    CHECK_ARGUMENT(input_elements == 0 || in != nullptr, "Linear: input pointer must not be null.");

    CHECK_ARGUMENT(
        weight_elements == 0 || weight != nullptr, "Linear: weight pointer must not be null.");

    if (output_elements == 0) { return; }

    // ========================================================
    // cuBLAS dimension limits
    // ========================================================

    constexpr std::size_t max_cublas_dimension
        = static_cast<std::size_t>(std::numeric_limits<int>::max());

    CHECK_ARGUMENT(
        row_count <= max_cublas_dimension, "Linear: row count exceeds the cuBLAS int limit.");

    CHECK_ARGUMENT(
        output_features <= max_cublas_dimension,
        "Linear: output feature count exceeds the cuBLAS int limit.");

    CHECK_ARGUMENT(
        input_features <= max_cublas_dimension,
        "Linear: input feature count exceeds the cuBLAS int limit.");

    // ========================================================
    // K == 0
    // ========================================================
    //
    // The matrix product contributes zero:
    //
    //     Y = bias
    //
    // or:
    //
    //     Y = 0
    //
    // No GEMM should be launched.
    // ========================================================

    if (input_features == 0) {
        initialize_linear_output(out, bias, output_elements, output_features, stream);

        return;
    }

    // ========================================================
    // Bias handling
    // ========================================================

    float beta = 0.0F;

    if (bias != nullptr) {
        initialize_linear_output(out, bias, output_elements, output_features, stream);

        // Y was initialized to the broadcast bias:
        //
        //     Y = bias
        //
        // GEMM then performs:
        //
        //     Y = XW^T + 1 * Y
        beta = 1.0F;
    }

    const float alpha = 1.0F;

    // ========================================================
    // NVIDIA dtype mapping
    // ========================================================

    constexpr cudaDataType_t data_type = llaisys::device::nvidia::CudaTypeTraits<T>::data_type;

    // ========================================================
    // NVIDIA cuBLAS handle
    // ========================================================

    cublasHandle_t handle = llaisys::device::nvidia::get_cublas_handle(stream);

    // ========================================================
    // GEMM
    // ========================================================
    //
    // LLAISYS stores row-major tensors:
    //
    //     X: [M, K]
    //     W: [N, K]
    //     Y: [M, N]
    //
    //     Y = X W^T + bias
    //
    // cuBLAS interprets the same memory as column-major:
    //
    //     X memory -> X^T [K, M]
    //     W memory -> W^T [K, N]
    //     Y memory -> Y^T [N, M]
    //
    // Therefore calculate:
    //
    //     Y^T = W X^T
    // ========================================================

    CUBLAS_CHECK(cublasGemmEx(
        handle,

        CUBLAS_OP_T, CUBLAS_OP_N,

        static_cast<int>(output_features),

        static_cast<int>(row_count),

        static_cast<int>(input_features),

        &alpha,

        weight, data_type, static_cast<int>(input_features),

        in, data_type, static_cast<int>(input_features),

        &beta,

        out, data_type, static_cast<int>(output_features),

        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

} // namespace

// ============================================================
// Public NVIDIA backend
// ============================================================

namespace llaisys::ops::nvidia {

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    std::size_t nrow,
    std::size_t ncol_out,
    std::size_t ncol_in,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = to_cuda_stream(stream);

    return llaisys::device::nvidia::dispatch_cuda_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_nvidia_linear<T>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
            reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(bias), nrow, ncol_out,
            ncol_in, cuda_stream);
    });
}

} // namespace llaisys::ops::nvidia
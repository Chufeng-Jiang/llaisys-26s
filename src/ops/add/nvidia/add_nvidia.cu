#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"
#include "../../cuda_compat/common.cuh"
#include "../cuda_compat/add_cuda_compat.cuh"
#include "add_nvidia.cuh"

#include <cstddef>

namespace {

namespace cuda_compat = llaisys::ops::cuda_compat;

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::CUDA_DEFAULT_MAX_GRID_SIZE;
using llaisys::device::nvidia::to_cuda_stream;

// ============================================================
// NVIDIA Add adapter
//
// Shared:
//   - scalar addition
//   - scalar kernel
//   - vector traits
//   - vectorized kernel
//   - tail handling
//
// NVIDIA-specific:
//   - CUDA block size
//   - CUDA grid cap
//   - cudaStream_t
//   - CUDA launch error checking
// ============================================================

/**
 * @brief Launches the element-wise Add kernel on an NVIDIA GPU.
 *
 * This function does not perform the addition itself. Instead, it prepares
 * the CUDA launch configuration and dispatches the actual Add kernel.
 *
 * The logical operation is:
 *
 *     c[i] = a[i] + b[i]
 *
 * for every element i in [0, numel).
 *
 * The launch process is:
 *
 *     1. Return immediately if the tensor is empty.
 *
 *     2. Decide whether the vectorized implementation can be used.
 *        The vectorized kernel processes multiple elements per GPU thread
 *        or memory transaction when pointer alignment and tensor size allow it.
 *
 *     3. Choose the NVIDIA CUDA block size.
 *
 *     4. Compute the number of logical work items.
 *        For a scalar kernel, this is typically close to numel.
 *        For a vectorized kernel, one work item may represent multiple elements.
 *
 *     5. Compute the required grid size:
 *
 *            number of blocks
 *                =
 *            ceil(work_items / block_size)
 *
 *        get_capped_grid_size() may additionally limit the grid size to
 *        a reasonable NVIDIA-specific maximum.
 *
 *     6. Launch the shared CUDA-compatible Add kernel on the provided
 *        CUDA stream.
 *
 *     7. Check whether the CUDA kernel launch produced an error.
 *
 * @tparam T Element data type, such as float, fp16, or bf16.
 * @param c Output device pointer.
 * @param a First input device pointer.
 * @param b Second input device pointer.
 * @param numel Number of tensor elements.
 * @param stream CUDA stream on which the kernel is launched.
 */
template <typename T>
void launch_nvidia_add(T *c, const T *a, const T *b, std::size_t numel, cudaStream_t stream) {
    if (numel == 0) { return; }

    const bool use_vectorized_kernel = cuda_compat::can_use_vectorized_add<T>(c, a, b, numel);

    constexpr std::size_t block_size = CUDA_BLOCK_SIZE;
    const std::size_t work_items = cuda_compat::get_add_work_items<T>(numel, use_vectorized_kernel);

    const std::size_t grid_size
        = cuda_compat::get_capped_grid_size(work_items, block_size, CUDA_DEFAULT_MAX_GRID_SIZE);

    cuda_compat::launch_add_kernel<T>(
        c, a, b, numel, block_size, grid_size, use_vectorized_kernel, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    std::size_t numel,
    llaisysStream_t stream) {
    CHECK_ARGUMENT(numel == 0 || c != nullptr, "Add: output pointer c must not be null.");
    CHECK_ARGUMENT(numel == 0 || a != nullptr, "Add: input pointer a must not be null.");
    CHECK_ARGUMENT(numel == 0 || b != nullptr, "Add: input pointer b must not be null.");

    const cudaStream_t cuda_stream = to_cuda_stream(stream);

    return llaisys::device::nvidia::dispatch_cuda_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_nvidia_add<T>(
            reinterpret_cast<T *>(c), reinterpret_cast<const T *>(a),
            reinterpret_cast<const T *>(b), numel, cuda_stream);
    });
}

} // namespace llaisys::ops::nvidia
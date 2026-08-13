#include "rope_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"
#include "../../cuda_compat/common.cuh"
#include "../cuda_compat/rope_cuda_compat.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {

namespace cuda_compat = llaisys::ops::cuda_compat;
using llaisys::device::nvidia::CUDA_DEFAULT_MAX_GRID_SIZE;
using llaisys::device::nvidia::get_warp_aligned_block_size;
using llaisys::device::nvidia::to_cuda_stream;
using llaisys::ops::cuda_compat::get_capped_grid_size;
using llaisys::utils::checked_product;

// ============================================================
// NVIDIA RoPE implementation selection
// ============================================================
//
// Environment variable:
//
//     LLAISYS_NVIDIA_ROPE_IMPL
//
// Values:
//
//     auto
//         Use the current NVIDIA tuning policy.
//
//     direct
//         Force Implementation 0:
//         direct RoPE kernel.
//
//     cached
//         Force Implementation 1:
//         shared-memory cached RoPE kernel.
//
// The environment variable is intentionally read on every
// operator invocation. This allows Direct/Cached interleaved
// A/B benchmarking within the same process.
// ============================================================

enum class NvidiaRopeImpl {
    AUTO,
    DIRECT,
    CACHED,
};

NvidiaRopeImpl get_nvidia_rope_impl() {
    const char *value = std::getenv("LLAISYS_NVIDIA_ROPE_IMPL");

    if (value == nullptr || std::strcmp(value, "auto") == 0) { return NvidiaRopeImpl::AUTO; }

    if (std::strcmp(value, "direct") == 0) { return NvidiaRopeImpl::DIRECT; }

    if (std::strcmp(value, "cached") == 0) { return NvidiaRopeImpl::CACHED; }

    throw std::invalid_argument("LLAISYS_NVIDIA_ROPE_IMPL must be auto, direct, or cached.");
}

// ============================================================
// NVIDIA RoPE tuning
// ============================================================
//
// Cache cosine/sine values in dynamic shared memory while the
// half dimension remains reasonably small.
//
// Shared-memory requirement:
//
//     2 * half_dimension * sizeof(float)
//
// For 2048:
//
//     2 * 2048 * 4 = 16384 bytes
//
// This threshold is an NVIDIA scheduling/tuning choice and is
// therefore deliberately kept outside cuda_compat.
//
// IMPORTANT:
//
// This threshold is used only by AUTO.
//
// If the user explicitly selects "cached", the cached
// implementation is forced so that the A/B benchmark actually
// measures the cached implementation.
// ============================================================

inline constexpr std::size_t NVIDIA_MAX_CACHED_HALF_DIMENSION = 2048;

// ============================================================
// NVIDIA implementation selector
// ============================================================

bool select_cached_kernel(
    NvidiaRopeImpl implementation, std::size_t head_count, std::size_t half_dimension) {
    switch (implementation) {
    case NvidiaRopeImpl::DIRECT:
        return false;

    case NvidiaRopeImpl::CACHED:
        CHECK_ARGUMENT(
            half_dimension <= NVIDIA_MAX_CACHED_HALF_DIMENSION,
            "RoPE: forced NVIDIA cached implementation exceeds "
            "NVIDIA_MAX_CACHED_HALF_DIMENSION.");

        return true;

    case NvidiaRopeImpl::AUTO:
        return head_count > 1 && half_dimension <= NVIDIA_MAX_CACHED_HALF_DIMENSION;
    }

    CHECK_ARGUMENT(false, "RoPE: invalid NVIDIA implementation.");

    return false;
}

// ============================================================
// NVIDIA launcher
// ============================================================

template <typename T>
void launch_nvidia_rope(
    T *out,
    const T *in,
    const std::int64_t *position_ids,
    float theta,
    std::size_t sequence_length,
    std::size_t head_count,
    std::size_t dimension,
    cudaStream_t stream) {
    // ========================================================
    // Validation
    // ========================================================

    const std::size_t vector_count = checked_product(
        sequence_length, head_count, "RoPE: sequence/head count overflows size_t.");

    const std::size_t element_count
        = checked_product(vector_count, dimension, "RoPE: tensor element count overflows size_t.");

    CHECK_ARGUMENT(element_count == 0 || out != nullptr, "RoPE: output pointer must not be null.");

    CHECK_ARGUMENT(element_count == 0 || in != nullptr, "RoPE: input pointer must not be null.");

    CHECK_ARGUMENT(
        sequence_length == 0 || position_ids != nullptr,
        "RoPE: position-id pointer must not be null.");

    CHECK_ARGUMENT(
        sequence_length == 0 || head_count > 0,
        "RoPE: head count must be greater than zero for a nonempty sequence.");

    CHECK_ARGUMENT(dimension > 0, "RoPE: head dimension must be greater than zero.");

    CHECK_ARGUMENT(dimension % 2 == 0, "RoPE: head dimension must be even.");

    CHECK_ARGUMENT(
        std::isfinite(theta) && theta > 0.0F, "RoPE: theta must be finite and greater than zero.");

    if (element_count == 0) { return; }

    // ========================================================
    // Algorithm dimensions
    // ========================================================

    const std::size_t half_dimension = dimension / 2;

    const std::size_t pair_count
        = checked_product(head_count, half_dimension, "RoPE: head/pair count overflows size_t.");

    // ========================================================
    // NVIDIA-specific launch tuning
    // ========================================================
    //
    // Both Direct and Cached deliberately use the same:
    //
    //     block_size
    //     grid_size
    //     stream
    //
    // This keeps the A/B comparison focused on the execution
    // strategy rather than changing the launch configuration.
    // ========================================================

    const unsigned int block_size
        = get_warp_aligned_block_size(std::max(half_dimension, pair_count));

    const std::size_t grid_size
        = get_capped_grid_size(sequence_length, 1, CUDA_DEFAULT_MAX_GRID_SIZE);

    // ========================================================
    // Implementation selection
    // ========================================================

    const NvidiaRopeImpl implementation = get_nvidia_rope_impl();

    const bool use_cached_kernel = select_cached_kernel(implementation, head_count, half_dimension);

    // ========================================================
    // Shared CUDA-compatible implementation
    // ========================================================
    //
    // Implementation 0:
    //
    //     use_cached_kernel = false
    //
    // Implementation 1:
    //
    //     use_cached_kernel = true
    //
    // The actual RoPE mathematical implementation remains in:
    //
    //     ../cuda_compat/rope_cuda_compat.cuh
    //
    // so NVIDIA and MetaX can reuse exactly the same algorithm.
    // ========================================================

    cuda_compat::launch_rope_kernel<T>(
        out, in, position_ids, theta, sequence_length, head_count, dimension, block_size, grid_size,
        use_cached_kernel, stream);

    // ========================================================
    // NVIDIA-specific launch error handling
    // ========================================================

    CUDA_CHECK(cudaGetLastError());
}

} // namespace

// ============================================================
// Public NVIDIA backend
// ============================================================

namespace llaisys::ops::nvidia {

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    float theta,
    llaisysDataType_t type,
    std::size_t seqlen,
    std::size_t nhead,
    std::size_t d,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = to_cuda_stream(stream);

    return llaisys::device::nvidia::dispatch_cuda_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_nvidia_rope<T>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
            reinterpret_cast<const std::int64_t *>(pos_ids), theta, seqlen, nhead, d, cuda_stream);
    });
}

} // namespace llaisys::ops::nvidia
#pragma once

#include "../../cuda/common.cuh"

#include "rearrange_cuda.hpp"
#include "../../../utils.hpp"
#include "../layout_utils.hpp"
#include "../rearrange_config.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace llaisys::ops::cuda {
namespace {

using rearrange_utils::absolute_stride;
using rearrange_utils::ContiguousTail;
using rearrange_utils::find_common_contiguous_tail;
using rearrange_utils::is_contiguous_layout;
using rearrange_utils::is_non_overlapping_layout;
using rearrange_utils::validate_layout_common;

// ============================================================
// Shared device-layout metadata
// ============================================================
//
// The same metadata representation is passed by value to kernels
// compiled by NVCC and MXCC.
//
// Rank 32 keeps the two layout descriptors comfortably below the
// common kernel-parameter budget used by the previous backends.
// ============================================================

inline constexpr std::size_t MAX_REARRANGE_RANK = 32;

struct DeviceLayout {
    std::size_t ndim{0};
    std::size_t shape[MAX_REARRANGE_RANK]{};
    std::ptrdiff_t strides[MAX_REARRANGE_RANK]{};
};

static_assert(
    sizeof(DeviceLayout) * 2 < 4096,
    "Rearrange: layout metadata exceeds the shared CUDA-compatible kernel-parameter budget.");

void validate_layout(
    const std::vector<std::size_t> &shape,
    const std::vector<std::ptrdiff_t> &strides,
    std::size_t expected_numel,
    const char *metadata_message,
    const char *rank_message,
    const char *numel_message,
    const char *overflow_message) {
    CHECK_ARGUMENT(shape.size() <= MAX_REARRANGE_RANK, rank_message);

    validate_layout_common(
        shape, strides, expected_numel, metadata_message, numel_message, overflow_message);
}

DeviceLayout make_device_layout(
    const std::vector<std::size_t> &shape, const std::vector<std::ptrdiff_t> &strides) {
    DeviceLayout layout{};
    layout.ndim = shape.size();

    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        layout.shape[dimension] = shape[dimension];
        layout.strides[dimension] = strides[dimension];
    }

    return layout;
}

// ============================================================
// Shared address-range overlap detection
// ============================================================

struct AddressRange {
    __int128 begin;
    __int128 end;
};

template <typename T>
AddressRange layout_address_range(
    const T *pointer,
    const std::vector<std::size_t> &shape,
    const std::vector<std::ptrdiff_t> &strides) {
    __int128 minimum_offset = 0;
    __int128 maximum_offset = 0;

    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (shape[dimension] <= 1) { continue; }

        const __int128 delta = static_cast<__int128>(shape[dimension] - 1)
                             * static_cast<__int128>(strides[dimension]);

        if (delta < 0) {
            minimum_offset += delta;
        } else {
            maximum_offset += delta;
        }
    }

    const __int128 base_address = static_cast<__int128>(reinterpret_cast<std::uintptr_t>(pointer));

    const __int128 element_bytes = static_cast<__int128>(sizeof(T));

    return AddressRange{
        base_address + minimum_offset * element_bytes,
        base_address + maximum_offset * element_bytes + element_bytes,
    };
}

bool address_ranges_overlap(const AddressRange &left, const AddressRange &right) {
    return left.begin < right.end && right.begin < left.end;
}

// ============================================================
// Shared Packed128 eligibility
// ============================================================
//
// Packed128 is either enabled on both CUDA-compatible backends or
// disabled by the same runtime eligibility check. This removes the
// previous NVIDIA-only vectorization difference from the main study.
// ============================================================

template <typename T>
bool can_use_packed_tail(
    const T *out,
    const T *in,
    const std::vector<std::size_t> &shape,
    const std::vector<std::ptrdiff_t> &out_strides,
    const std::vector<std::ptrdiff_t> &in_strides,
    const ContiguousTail &tail) {
    constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;

    if (tail.element_count % elements_per_pack != 0
        || !are_aligned<PACKED_128_ALIGNMENT>(out, in)) {
        return false;
    }

    for (std::size_t dimension = 0; dimension < tail.start_dimension; ++dimension) {
        if (shape[dimension] <= 1) { continue; }

        if (absolute_stride(out_strides[dimension]) % elements_per_pack != 0
            || absolute_stride(in_strides[dimension]) % elements_per_pack != 0) {
            return false;
        }
    }

    return true;
}

// ============================================================
// Device-side indexing
// ============================================================

__device__ __forceinline__ std::ptrdiff_t
logical_offset(std::size_t flat_index, const DeviceLayout &layout, std::size_t dimension_count) {
    std::ptrdiff_t offset = 0;

    for (std::size_t dimension = dimension_count; dimension-- > 0;) {
        const std::size_t extent = layout.shape[dimension];
        const std::size_t coordinate = flat_index % extent;

        flat_index /= extent;

        offset += static_cast<std::ptrdiff_t>(coordinate) * layout.strides[dimension];
    }

    return offset;
}

// ============================================================
// Shared kernels
// ============================================================

template <typename T>
__global__ void rearrange_generic_kernel(
    T *out, const T *in, std::size_t numel, DeviceLayout out_layout, DeviceLayout in_layout) {
    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t grid_stride
        = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);

    for (std::size_t logical_index = thread_index; logical_index < numel;
         logical_index += grid_stride) {
        const std::ptrdiff_t out_offset
            = logical_offset(logical_index, out_layout, out_layout.ndim);

        const std::ptrdiff_t in_offset = logical_offset(logical_index, in_layout, in_layout.ndim);

        out[out_offset] = in[in_offset];
    }
}

template <typename T>
__global__ void
gather_to_contiguous_kernel(T *temporary, const T *in, std::size_t numel, DeviceLayout in_layout) {
    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t grid_stride
        = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);

    for (std::size_t logical_index = thread_index; logical_index < numel;
         logical_index += grid_stride) {
        const std::ptrdiff_t in_offset = logical_offset(logical_index, in_layout, in_layout.ndim);

        temporary[logical_index] = in[in_offset];
    }
}

template <typename T>
__global__ void scatter_from_contiguous_kernel(
    T *out, const T *temporary, std::size_t numel, DeviceLayout out_layout) {
    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t grid_stride
        = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);

    for (std::size_t logical_index = thread_index; logical_index < numel;
         logical_index += grid_stride) {
        const std::ptrdiff_t out_offset
            = logical_offset(logical_index, out_layout, out_layout.ndim);

        out[out_offset] = temporary[logical_index];
    }
}

template <typename T>
__global__ void contiguous_copy_kernel(T *out, const T *in, std::size_t numel) {
    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t grid_stride
        = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);

    for (std::size_t index = thread_index; index < numel; index += grid_stride) {
        out[index] = in[index];
    }
}

template <typename T>
__global__ void rearrange_contiguous_tail_kernel(
    T *out,
    const T *in,
    std::size_t outer_block_count,
    std::size_t tail_element_count,
    std::size_t outer_dimension_count,
    DeviceLayout out_layout,
    DeviceLayout in_layout) {
    for (std::size_t outer_block = static_cast<std::size_t>(blockIdx.x);
         outer_block < outer_block_count; outer_block += static_cast<std::size_t>(gridDim.x)) {
        const std::ptrdiff_t out_offset
            = logical_offset(outer_block, out_layout, outer_dimension_count);

        const std::ptrdiff_t in_offset
            = logical_offset(outer_block, in_layout, outer_dimension_count);

        for (std::size_t item = static_cast<std::size_t>(threadIdx.x); item < tail_element_count;
             item += static_cast<std::size_t>(blockDim.x)) {
            out[out_offset + static_cast<std::ptrdiff_t>(item)]
                = in[in_offset + static_cast<std::ptrdiff_t>(item)];
        }
    }
}

template <typename T>
__global__ void rearrange_contiguous_tail_packed_kernel(
    T *out,
    const T *in,
    std::size_t outer_block_count,
    std::size_t packs_per_outer_block,
    std::size_t outer_dimension_count,
    DeviceLayout out_layout,
    DeviceLayout in_layout) {
    for (std::size_t outer_block = static_cast<std::size_t>(blockIdx.x);
         outer_block < outer_block_count; outer_block += static_cast<std::size_t>(gridDim.x)) {
        const std::ptrdiff_t out_offset
            = logical_offset(outer_block, out_layout, outer_dimension_count);

        const std::ptrdiff_t in_offset
            = logical_offset(outer_block, in_layout, outer_dimension_count);

        auto *packed_out = reinterpret_cast<Packed128 *>(out + out_offset);
        const auto *packed_in = reinterpret_cast<const Packed128 *>(in + in_offset);

        for (std::size_t pack = static_cast<std::size_t>(threadIdx.x); pack < packs_per_outer_block;
             pack += static_cast<std::size_t>(blockDim.x)) {
            packed_out[pack] = packed_in[pack];
        }
    }
}

// ============================================================
// Shared launch policy
// ============================================================

std::size_t get_grid_size(std::size_t work_items) {
    return get_capped_grid_size(
        work_items, rearrange_config::block_size(), rearrange_config::MAX_BLOCKS);
}

std::size_t get_outer_grid_size(std::size_t outer_block_count) {
    return cap_grid_size(outer_block_count, rearrange_config::MAX_BLOCKS);
}

template <typename T>
void launch_generic_copy(
    T *out,
    const T *in,
    std::size_t numel,
    const DeviceLayout &out_layout,
    const DeviceLayout &in_layout,
    cudaStream_t stream) {
    const std::size_t block_size = rearrange_config::block_size();
    const std::size_t grid_size = get_grid_size(numel);

    rearrange_generic_kernel<T>
        <<<static_cast<unsigned int>(grid_size), static_cast<unsigned int>(block_size), 0,
           stream>>>(out, in, numel, out_layout, in_layout);

    check_kernel("Rearrange generic kernel");
}

template <typename T>
void launch_gather(
    T *temporary,
    const T *in,
    std::size_t numel,
    const DeviceLayout &in_layout,
    cudaStream_t stream) {
    const std::size_t block_size = rearrange_config::block_size();
    const std::size_t grid_size = get_grid_size(numel);

    gather_to_contiguous_kernel<T>
        <<<static_cast<unsigned int>(grid_size), static_cast<unsigned int>(block_size), 0,
           stream>>>(temporary, in, numel, in_layout);

    check_kernel("Rearrange gather kernel");
}

template <typename T>
void launch_scatter(
    T *out,
    const T *temporary,
    std::size_t numel,
    const DeviceLayout &out_layout,
    cudaStream_t stream) {
    const std::size_t block_size = rearrange_config::block_size();
    const std::size_t grid_size = get_grid_size(numel);

    scatter_from_contiguous_kernel<T>
        <<<static_cast<unsigned int>(grid_size), static_cast<unsigned int>(block_size), 0,
           stream>>>(out, temporary, numel, out_layout);

    check_kernel("Rearrange scatter kernel");
}

template <typename T>
void launch_contiguous_copy(T *out, const T *in, std::size_t numel, cudaStream_t stream) {
    const std::size_t block_size = rearrange_config::block_size();
    const std::size_t grid_size = get_grid_size(numel);

    contiguous_copy_kernel<T>
        <<<static_cast<unsigned int>(grid_size), static_cast<unsigned int>(block_size), 0,
           stream>>>(out, in, numel);

    check_kernel("Rearrange contiguous-copy kernel");
}

template <typename T>
void launch_contiguous_tail_copy(
    T *out,
    const T *in,
    const DeviceLayout &out_layout,
    const DeviceLayout &in_layout,
    const std::vector<std::size_t> &shape,
    const std::vector<std::ptrdiff_t> &out_strides,
    const std::vector<std::ptrdiff_t> &in_strides,
    std::size_t numel,
    const ContiguousTail &tail,
    cudaStream_t stream) {
    const std::size_t outer_block_count = numel / tail.element_count;
    const std::size_t grid_size = get_outer_grid_size(outer_block_count);
    const std::size_t block_size = rearrange_config::block_size();

    if (can_use_packed_tail(out, in, shape, out_strides, in_strides, tail)) {
        constexpr std::size_t elements_per_pack = PACKED_128_ELEMENTS<T>;
        const std::size_t packs_per_outer_block = tail.element_count / elements_per_pack;

        rearrange_contiguous_tail_packed_kernel<T>
            <<<static_cast<unsigned int>(grid_size), static_cast<unsigned int>(block_size), 0,
               stream>>>(
                out, in, outer_block_count, packs_per_outer_block, tail.start_dimension, out_layout,
                in_layout);

        check_kernel("Rearrange packed contiguous-tail kernel");

        if (config::debug_enabled()) {
            std::fprintf(
                stderr,
                "[Rearrange][%s] implementation=shared path=contiguous_tail_packed "
                "numel=%zu block=%zu grid=%zu tail=%zu packs=%zu\n",
                GPU_BACKEND_NAME, numel, block_size, grid_size, tail.element_count,
                packs_per_outer_block);
        }

        return;
    }

    rearrange_contiguous_tail_kernel<T>
        <<<static_cast<unsigned int>(grid_size), static_cast<unsigned int>(block_size), 0,
           stream>>>(
            out, in, outer_block_count, tail.element_count, tail.start_dimension, out_layout,
            in_layout);

    check_kernel("Rearrange contiguous-tail kernel");

    if (config::debug_enabled()) {
        std::fprintf(
            stderr,
            "[Rearrange][%s] implementation=shared path=contiguous_tail "
            "numel=%zu block=%zu grid=%zu tail=%zu\n",
            GPU_BACKEND_NAME, numel, block_size, grid_size, tail.element_count);
    }
}

// ============================================================
// Shared overlap temporary
// ============================================================
//
// V1 intentionally uses the same allocation and synchronization policy
// on both CUDA-compatible backends. The overlap path is therefore not
// polluted by NVIDIA cudaMallocAsync versus MetaX mcMalloc behavior.
//
// This path is correctness-oriented. If overlap becomes performance
// critical, allocator policy should be studied as a separate variable.
// ============================================================

template <typename T> T *allocate_temporary(std::size_t numel) {
    T *temporary = nullptr;

    check_cuda(
        cudaMalloc(reinterpret_cast<void **>(&temporary), numel * sizeof(T)),
        "Rearrange temporary cudaMalloc");

    return temporary;
}

template <typename T> void release_temporary_after_stream(T *temporary, cudaStream_t stream) {
    check_cuda(cudaStreamSynchronize(stream), "Rearrange temporary stream synchronize");
    check_cuda(cudaFree(temporary), "Rearrange temporary cudaFree");
}

// ============================================================
// Shared typed implementation
// ============================================================

template <typename T>
void rearrange_typed(
    T *out,
    const T *in,
    std::size_t numel,
    const std::vector<std::size_t> &out_shape,
    const std::vector<std::ptrdiff_t> &out_strides,
    const std::vector<std::size_t> &in_shape,
    const std::vector<std::ptrdiff_t> &in_strides,
    cudaStream_t stream) {
    validate_layout(
        out_shape, out_strides, numel, "Rearrange: output shape and stride counts must match.",
        "Rearrange: output rank exceeds the shared CUDA-compatible metadata limit.",
        "Rearrange: output shape does not match numel.",
        "Rearrange: output shape element count overflows size_t.");

    validate_layout(
        in_shape, in_strides, numel, "Rearrange: input shape and stride counts must match.",
        "Rearrange: input rank exceeds the shared CUDA-compatible metadata limit.",
        "Rearrange: input shape does not match numel.",
        "Rearrange: input shape element count overflows size_t.");

    CHECK_ARGUMENT(numel == 0 || out != nullptr, "Rearrange: output pointer must not be null.");
    CHECK_ARGUMENT(numel == 0 || in != nullptr, "Rearrange: input pointer must not be null.");

    if (numel == 0) { return; }

    CHECK_ARGUMENT(
        numel <= std::numeric_limits<std::size_t>::max() / sizeof(T),
        "Rearrange: byte count overflows size_t.");

    CHECK_ARGUMENT(
        is_non_overlapping_layout(out_shape, out_strides),
        "Rearrange: output layout must be non-overlapping.");

    if (out == in && out_shape == in_shape && out_strides == in_strides) {
        if (config::debug_enabled()) {
            std::fprintf(
                stderr, "[Rearrange][%s] implementation=shared path=noop numel=%zu\n",
                GPU_BACKEND_NAME, numel);
        }

        return;
    }

    const DeviceLayout out_layout = make_device_layout(out_shape, out_strides);
    const DeviceLayout in_layout = make_device_layout(in_shape, in_strides);

    const bool out_is_contiguous = is_contiguous_layout(out_shape, out_strides);
    const bool in_is_contiguous = is_contiguous_layout(in_shape, in_strides);

    const bool storage_overlaps = address_ranges_overlap(
        layout_address_range(out, out_shape, out_strides),
        layout_address_range(in, in_shape, in_strides));

    if (storage_overlaps) {
        T *temporary = allocate_temporary<T>(numel);

        if (in_is_contiguous) {
            launch_contiguous_copy(temporary, in, numel, stream);
        } else {
            launch_gather(temporary, in, numel, in_layout, stream);
        }

        if (out_is_contiguous) {
            launch_contiguous_copy(out, temporary, numel, stream);
        } else {
            launch_scatter(out, temporary, numel, out_layout, stream);
        }

        if (config::debug_enabled()) {
            std::fprintf(
                stderr,
                "[Rearrange][%s] implementation=shared path=overlap_temporary "
                "numel=%zu block=%zu\n",
                GPU_BACKEND_NAME, numel, rearrange_config::block_size());
        }

        release_temporary_after_stream(temporary, stream);
        return;
    }

    if (out_is_contiguous && in_is_contiguous) {
        if (config::debug_enabled()) {
            std::fprintf(
                stderr,
                "[Rearrange][%s] implementation=shared path=contiguous "
                "numel=%zu block=%zu grid=%zu\n",
                GPU_BACKEND_NAME, numel, rearrange_config::block_size(), get_grid_size(numel));
        }

        return launch_contiguous_copy(out, in, numel, stream);
    }

    if (out_shape == in_shape) {
        const ContiguousTail tail = find_common_contiguous_tail(out_shape, out_strides, in_strides);

        if (tail.element_count > 1) {
            return launch_contiguous_tail_copy(
                out, in, out_layout, in_layout, out_shape, out_strides, in_strides, numel, tail,
                stream);
        }
    }

    if (config::debug_enabled()) {
        std::fprintf(
            stderr,
            "[Rearrange][%s] implementation=shared path=generic "
            "numel=%zu block=%zu grid=%zu\n",
            GPU_BACKEND_NAME, numel, rearrange_config::block_size(), get_grid_size(numel));
    }

    launch_generic_copy(out, in, numel, out_layout, in_layout, stream);
}

} // namespace

void rearrange(
    std::byte *out,
    const std::byte *in,
    llaisysDataType_t type,
    std::size_t numel,
    const std::vector<std::size_t> &out_shape,
    const std::vector<std::ptrdiff_t> &out_strides,
    const std::vector<std::size_t> &in_shape,
    const std::vector<std::ptrdiff_t> &in_strides,
    llaisysStream_t stream) {
    static_assert(
        sizeof(float) == sizeof(std::uint32_t), "Rearrange: Float32 must occupy four bytes.");

    static_assert(
        sizeof(llaisys::fp16_t) == sizeof(std::uint16_t), "Rearrange: FP16 must occupy two bytes.");

    static_assert(
        sizeof(llaisys::bf16_t) == sizeof(std::uint16_t), "Rearrange: BF16 must occupy two bytes.");

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_typed<std::uint32_t>(
            reinterpret_cast<std::uint32_t *>(out), reinterpret_cast<const std::uint32_t *>(in),
            numel, out_shape, out_strides, in_shape, in_strides, cuda_stream);

    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return rearrange_typed<std::uint16_t>(
            reinterpret_cast<std::uint16_t *>(out), reinterpret_cast<const std::uint16_t *>(in),
            numel, out_shape, out_strides, in_shape, in_strides, cuda_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cuda

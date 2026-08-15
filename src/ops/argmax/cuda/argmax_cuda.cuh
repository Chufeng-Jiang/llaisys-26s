#pragma once

#include "../../cuda/common.cuh"
#include "argmax_cuda.hpp"
#include "../../../utils.hpp"
#include "../argmax_config.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace llaisys::ops::cuda {
namespace {

inline constexpr std::uint32_t ARGMAX_INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

struct ArgmaxResult {
    float value;
    std::uint32_t index;
};

static_assert(sizeof(std::int64_t) == 8, "Argmax requires a 64-bit output index.");

__host__ __device__ constexpr ArgmaxResult invalid_argmax_result() {
    return ArgmaxResult{0.0F, ARGMAX_INVALID_INDEX};
}

__host__ __device__ constexpr bool is_valid_argmax_result(const ArgmaxResult &result) {
    return result.index != ARGMAX_INVALID_INDEX;
}

__device__ __forceinline__ bool is_better_argmax(
    float candidate_value,
    std::uint32_t candidate_index,
    float current_value,
    std::uint32_t current_index) {
    const bool candidate_is_nan = isnan(candidate_value);
    const bool current_is_nan = isnan(current_value);

    if (candidate_is_nan != current_is_nan) { return candidate_is_nan; }
    if (candidate_value > current_value) { return true; }
    if (candidate_value < current_value) { return false; }

    return candidate_index < current_index;
}

__device__ __forceinline__ void
update_argmax_result(ArgmaxResult &result, float candidate_value, std::uint32_t candidate_index) {
    if (!is_valid_argmax_result(result)
        || is_better_argmax(candidate_value, candidate_index, result.value, result.index)) {
        result.value = candidate_value;
        result.index = candidate_index;
    }
}

__device__ __forceinline__ ArgmaxResult
block_reduce_argmax(ArgmaxResult thread_result, ArgmaxResult *shared_results) {
    const unsigned int thread_index = threadIdx.x;
    shared_results[thread_index] = thread_result;
    __syncthreads();

    unsigned int active_count = blockDim.x;

    while (active_count > 1) {
        const unsigned int next_count = (active_count + 1U) / 2U;
        const unsigned int pair_count = active_count / 2U;

        if (thread_index < pair_count) {
            const ArgmaxResult candidate = shared_results[thread_index + next_count];

            if (is_valid_argmax_result(candidate)) {
                update_argmax_result(
                    shared_results[thread_index], candidate.value, candidate.index);
            }
        }

        __syncthreads();
        active_count = next_count;
    }

    return shared_results[0];
}

template <typename T>
__global__ void argmax_stage1_kernel(
    ArgmaxResult *__restrict__ partial_results, const T *__restrict__ vals, std::size_t numel) {
    extern __shared__ ArgmaxResult shared_results[];

    const std::size_t thread_index
        = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);

    const std::size_t grid_stride
        = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);

    ArgmaxResult thread_result = invalid_argmax_result();

    for (std::size_t index = thread_index; index < numel; index += grid_stride) {
        const float value = to_float<T>(vals[index]);
        update_argmax_result(thread_result, value, static_cast<std::uint32_t>(index));
    }

    const ArgmaxResult block_result = block_reduce_argmax(thread_result, shared_results);

    if (threadIdx.x == 0) { partial_results[blockIdx.x] = block_result; }
}

template <typename T>
__global__ void argmax_stage2_kernel(
    std::int64_t *__restrict__ max_idx,
    T *__restrict__ max_val,
    const ArgmaxResult *__restrict__ partial_results,
    std::size_t partial_count) {
    extern __shared__ ArgmaxResult shared_results[];

    ArgmaxResult thread_result = invalid_argmax_result();

    for (std::size_t partial_index = static_cast<std::size_t>(threadIdx.x);
         partial_index < partial_count; partial_index += static_cast<std::size_t>(blockDim.x)) {
        const ArgmaxResult candidate = partial_results[partial_index];

        if (is_valid_argmax_result(candidate)) {
            update_argmax_result(thread_result, candidate.value, candidate.index);
        }
    }

    const ArgmaxResult final_result = block_reduce_argmax(thread_result, shared_results);

    if (threadIdx.x == 0 && is_valid_argmax_result(final_result)) {
        *max_idx = static_cast<std::int64_t>(final_result.index);
        *max_val = from_float<T>(final_result.value);
    }
}

class ArgmaxWorkspace final {
public:
    ArgmaxWorkspace() = default;

    ~ArgmaxWorkspace() noexcept { release(); }

    ArgmaxWorkspace(const ArgmaxWorkspace &) = delete;
    ArgmaxWorkspace &operator=(const ArgmaxWorkspace &) = delete;
    ArgmaxWorkspace(ArgmaxWorkspace &&) = delete;
    ArgmaxWorkspace &operator=(ArgmaxWorkspace &&) = delete;

    ArgmaxResult *get(int device_id, std::size_t required_count) {
        CHECK_ARGUMENT(device_id >= 0, "Argmax: invalid GPU device id.");
        CHECK_ARGUMENT(required_count > 0, "Argmax: workspace count must be greater than zero.");

        if (_device_id >= 0 && _device_id != device_id) { release(); }

        check_cuda(cudaSetDevice(device_id), "Argmax cudaSetDevice");

        if (_device_id < 0) { _device_id = device_id; }

        if (_pointer != nullptr && _capacity >= required_count) { return _pointer; }

        if (_pointer != nullptr) {
            check_cuda(cudaFree(_pointer), "Argmax workspace cudaFree");
            _pointer = nullptr;
            _capacity = 0;
        }

        check_cuda(
            cudaMalloc(reinterpret_cast<void **>(&_pointer), required_count * sizeof(ArgmaxResult)),
            "Argmax workspace cudaMalloc");

        _capacity = required_count;
        return _pointer;
    }

private:
    void release() noexcept {
        if (_pointer == nullptr) {
            _capacity = 0;
            _device_id = -1;
            return;
        }

        int previous_device = -1;
        const cudaError_t get_device_status = cudaGetDevice(&previous_device);

        if (_device_id >= 0) { (void)cudaSetDevice(_device_id); }

        (void)cudaFree(_pointer);

        _pointer = nullptr;
        _capacity = 0;

        if (get_device_status == cudaSuccess && previous_device >= 0
            && previous_device != _device_id) {
            (void)cudaSetDevice(previous_device);
        }

        _device_id = -1;
    }

private:
    int _device_id{-1};
    ArgmaxResult *_pointer{nullptr};
    std::size_t _capacity{0};
};

thread_local ArgmaxWorkspace ARGMAX_WORKSPACE;

template <typename T>
void launch_argmax(
    std::int64_t *max_idx,
    T *max_val,
    const T *vals,
    std::size_t numel,
    int device_id,
    cudaStream_t stream) {
    const argmax_config::LaunchConfig launch_config = argmax_config::get_launch_config(numel);

    ArgmaxResult *partial_results
        = ARGMAX_WORKSPACE.get(device_id, static_cast<std::size_t>(launch_config.grid_size));

    CHECK_ARGUMENT(
        partial_results != nullptr, "Argmax: partial-result workspace must not be null.");

    if (config::debug_enabled()) {
        std::fprintf(
            stderr, "[Argmax][%s] implementation=shared numel=%zu block=%u grid=%u\n",
            GPU_BACKEND_NAME, numel, launch_config.block_size, launch_config.grid_size);
    }

    const std::size_t shared_memory_bytes
        = static_cast<std::size_t>(launch_config.block_size) * sizeof(ArgmaxResult);

    argmax_stage1_kernel<T>
        <<<launch_config.grid_size, launch_config.block_size, shared_memory_bytes, stream>>>(
            partial_results, vals, numel);

    argmax_stage2_kernel<T><<<1, launch_config.block_size, shared_memory_bytes, stream>>>(
        max_idx, max_val, partial_results, static_cast<std::size_t>(launch_config.grid_size));

    check_kernel("Argmax kernel");
}

} // namespace

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    std::size_t numel,
    int device_id,
    llaisysStream_t stream) {
    CHECK_ARGUMENT(max_idx != nullptr, "Argmax: max_idx pointer must not be null.");
    CHECK_ARGUMENT(max_val != nullptr, "Argmax: max_val pointer must not be null.");
    CHECK_ARGUMENT(vals != nullptr, "Argmax: vals pointer must not be null.");
    CHECK_ARGUMENT(numel > 0, "Argmax: input tensor must not be empty.");
    CHECK_ARGUMENT(
        numel <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),
        "Argmax: shared CUDA-compatible implementation supports at most UINT32_MAX elements.");
    CHECK_ARGUMENT(device_id >= 0, "Argmax: GPU device id must be non-negative.");

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    return dispatch_dtype(type, [&](auto tag) {
        using T = typename decltype(tag)::type;

        return launch_argmax<T>(
            reinterpret_cast<std::int64_t *>(max_idx), reinterpret_cast<T *>(max_val),
            reinterpret_cast<const T *>(vals), numel, device_id, cuda_stream);
    });
}

} // namespace llaisys::ops::cuda

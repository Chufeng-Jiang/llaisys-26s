#include "op.hpp"

#include "../../core/context/context.hpp"
#include "../../device/device_utils.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
#include "cuda/rearrange_cuda.hpp"
#endif

#include <cstddef>

namespace llaisys::ops {

void rearrange(tensor_t out, tensor_t in) {
    CHECK_ARGUMENT(out != nullptr, "Rearrange: output tensor must not be null.");
    CHECK_ARGUMENT(in != nullptr, "Rearrange: input tensor must not be null.");

    CHECK_ARGUMENT(
        out->numel() == in->numel(),
        "Rearrange: input and output tensors must have the same number of elements.");

    CHECK_ARGUMENT(
        out->dtype() == in->dtype(),
        "Rearrange: input and output tensors must use the same data type.");

    CHECK_ARGUMENT(
        out->deviceType() == in->deviceType(),
        "Rearrange: input and output tensors must use the same device type.");

    CHECK_ARGUMENT(
        out->deviceId() == in->deviceId(),
        "Rearrange: input and output tensors must be on the same device.");

    const std::size_t numel = out->numel();

    if (numel == 0) { return; }

    const auto device_type = out->deviceType();

    if (device_type == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(
            out->data(), in->data(), out->dtype(), numel, out->shape(), out->strides(), in->shape(),
            in->strides());
    }

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    if (device::is_cuda_compatible_gpu(device_type)) {
        core::context().setDevice(device_type, out->deviceId());

        auto &runtime = core::context().runtime();

        return cuda::rearrange(
            out->data(), in->data(), out->dtype(), numel, out->shape(), out->strides(), in->shape(),
            in->strides(), runtime.stream());
    }
#endif

    CHECK_ARGUMENT(false, "Rearrange: unsupported device type.");
}

} // namespace llaisys::ops

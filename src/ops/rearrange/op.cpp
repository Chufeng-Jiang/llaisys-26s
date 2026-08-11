#include "op.hpp"

#include "../../core/context/context.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/rearrange_metax.hpp"
#endif

#include <cstddef>

namespace llaisys::ops {

void rearrange(tensor_t out, tensor_t in) {
    CHECK_ARGUMENT(out != nullptr, "Rearrange: output tensor must not be null.");

    CHECK_ARGUMENT(in != nullptr, "Rearrange: input tensor must not be null.");

    CHECK_ARGUMENT(
        out->numel() == in->numel(),
        "Rearrange: input and output tensors must have the same number of "
        "elements.");

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

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(
            out->data(),
            in->data(),
            out->dtype(),
            numel,
            out->shape(),
            out->strides(),
            in->shape(),
            in->strides());

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        core::context().setDevice(out->deviceType(), out->deviceId());

        auto &runtime = core::context().runtime();

        return nvidia::rearrange(
            out->data(),
            in->data(),
            out->dtype(),
            numel,
            out->shape(),
            out->strides(),
            in->shape(),
            in->strides(),
            runtime.stream());
    }
#endif

#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX: {
        core::context().setDevice(out->deviceType(), out->deviceId());

        auto &runtime = core::context().runtime();

        return metax::rearrange(
            out->data(),
            in->data(),
            out->dtype(),
            numel,
            out->shape(),
            out->strides(),
            in->shape(),
            in->strides(),
            runtime.stream());
    }
#endif

    default:
        CHECK_ARGUMENT(false, "Rearrange: unsupported device type.");

        return;
    }
}

} // namespace llaisys::ops
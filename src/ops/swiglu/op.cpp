#include "op.hpp"

#include "../../core/context/context.hpp"
#include "../../device/device_utils.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
#include "cuda/swiglu_cuda.hpp"
#endif

#include <cstddef>

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_ARGUMENT(out != nullptr, "SwiGLU: output tensor must not be null.");

    CHECK_ARGUMENT(gate != nullptr, "SwiGLU: gate tensor must not be null.");

    CHECK_ARGUMENT(up != nullptr, "SwiGLU: up tensor must not be null.");

    CHECK_ARGUMENT(out->ndim() == 2, "SwiGLU: output tensor must be two-dimensional.");

    CHECK_ARGUMENT(gate->ndim() == 2, "SwiGLU: gate tensor must be two-dimensional.");

    CHECK_ARGUMENT(up->ndim() == 2, "SwiGLU: up tensor must be two-dimensional.");

    CHECK_ARGUMENT(
        out->shape() == gate->shape(), "SwiGLU: output and gate tensors must have the same shape.");

    CHECK_ARGUMENT(
        out->shape() == up->shape(), "SwiGLU: output and up tensors must have the same shape.");

    CHECK_ARGUMENT(
        out->numel() == gate->numel(),
        "SwiGLU: output and gate tensors must have the same number of elements.");

    CHECK_ARGUMENT(
        out->numel() == up->numel(),
        "SwiGLU: output and up tensors must have the same number of elements.");

    CHECK_ARGUMENT(
        out->dtype() == gate->dtype(),
        "SwiGLU: output and gate tensors must use the same data type.");

    CHECK_ARGUMENT(
        out->dtype() == up->dtype(), "SwiGLU: output and up tensors must use the same data type.");

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    CHECK_ARGUMENT(
        out->deviceType() == gate->deviceType(),
        "SwiGLU: output and gate tensors must use the same device type.");

    CHECK_ARGUMENT(
        out->deviceType() == up->deviceType(),
        "SwiGLU: output and up tensors must use the same device type.");

    CHECK_ARGUMENT(
        out->deviceId() == gate->deviceId(),
        "SwiGLU: output and gate tensors must be located on the same device.");

    CHECK_ARGUMENT(
        out->deviceId() == up->deviceId(),
        "SwiGLU: output and up tensors must be located on the same device.");

    CHECK_ARGUMENT(out->isContiguous(), "SwiGLU: output tensor must be contiguous.");

    CHECK_ARGUMENT(gate->isContiguous(), "SwiGLU: gate tensor must be contiguous.");

    CHECK_ARGUMENT(up->isContiguous(), "SwiGLU: up tensor must be contiguous.");

    const std::size_t numel = out->numel();

    if (numel == 0) { return; }

    const auto device_type = out->deviceType();

    if (device_type == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);
    }

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    if (device::is_cuda_compatible_gpu(device_type)) {
        core::context().setDevice(device_type, out->deviceId());

        auto &runtime = core::context().runtime();

        return cuda::swiglu(
            out->data(), gate->data(), up->data(), out->dtype(), numel, runtime.stream());
    }
#endif

    CHECK_ARGUMENT(false, "SwiGLU: unsupported device type.");
}

} // namespace llaisys::ops

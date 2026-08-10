#include "op.hpp"

#include "../../core/context/context.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/swiglu_metax.hpp"
#endif

#include <cstddef>

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // ============================================================
    // Null checks
    // ============================================================

    CHECK_ARGUMENT(out != nullptr, "SwiGLU: output tensor must not be null.");

    CHECK_ARGUMENT(gate != nullptr, "SwiGLU: gate tensor must not be null.");

    CHECK_ARGUMENT(up != nullptr, "SwiGLU: up tensor must not be null.");

    // ============================================================
    // Dimension checks
    // ============================================================

    CHECK_ARGUMENT(out->ndim() == 2, "SwiGLU: output tensor must be two-dimensional.");

    CHECK_ARGUMENT(gate->ndim() == 2, "SwiGLU: gate tensor must be two-dimensional.");

    CHECK_ARGUMENT(up->ndim() == 2, "SwiGLU: up tensor must be two-dimensional.");

    // ============================================================
    // Shape checks
    // ============================================================

    CHECK_ARGUMENT(
        out->shape() == gate->shape(), "SwiGLU: output and gate tensors must have the same shape.");

    CHECK_ARGUMENT(
        out->shape() == up->shape(), "SwiGLU: output and up tensors must have the same shape.");

    CHECK_ARGUMENT(
        out->numel() == gate->numel(),
        "SwiGLU: output and gate tensors must have the same number of "
        "elements.");

    CHECK_ARGUMENT(
        out->numel() == up->numel(),
        "SwiGLU: output and up tensors must have the same number of elements.");

    // ============================================================
    // Data-type checks
    // ============================================================

    CHECK_ARGUMENT(
        out->dtype() == gate->dtype(),
        "SwiGLU: output and gate tensors must use the same data type.");

    CHECK_ARGUMENT(
        out->dtype() == up->dtype(), "SwiGLU: output and up tensors must use the same data type.");

    // ============================================================
    // Device-type checks
    // ============================================================

    CHECK_ARGUMENT(
        out->deviceType() == gate->deviceType(),
        "SwiGLU: output and gate tensors must use the same device type.");

    CHECK_ARGUMENT(
        out->deviceType() == up->deviceType(),
        "SwiGLU: output and up tensors must use the same device type.");

    // ============================================================
    // Device-ID checks
    // ============================================================

    CHECK_ARGUMENT(
        out->deviceId() == gate->deviceId(),
        "SwiGLU: output and gate tensors must be located on the same device.");

    CHECK_ARGUMENT(
        out->deviceId() == up->deviceId(),
        "SwiGLU: output and up tensors must be located on the same device.");

    // ============================================================
    // Contiguity checks
    // ============================================================

    CHECK_ARGUMENT(out->isContiguous(), "SwiGLU: output tensor must be contiguous.");

    CHECK_ARGUMENT(gate->isContiguous(), "SwiGLU: gate tensor must be contiguous.");

    CHECK_ARGUMENT(up->isContiguous(), "SwiGLU: up tensor must be contiguous.");

    const std::size_t numel = out->numel();

    // CUDA does not allow launching a kernel with zero blocks.
    // The CPU implementation also has no work to perform.
    if (numel == 0) { return; }

    // ============================================================
    // Device dispatch
    // ============================================================

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), numel);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        // Select the correct CUDA device before obtaining its
        // Runtime and stream.
        core::context().setDevice(out->deviceType(), out->deviceId());

        auto &runtime = core::context().runtime();

        return nvidia::swiglu(
            out->data(), gate->data(), up->data(), out->dtype(), numel, runtime.stream());
    }
#endif

#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX: {
        core::context().setDevice(
            out->deviceType(),
            out->deviceId()
        );

        auto &runtime =
            core::context().runtime();

        return metax::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            numel,
            runtime.stream()
        );
    }
#endif


    default:
        CHECK_ARGUMENT(false, "SwiGLU: unsupported device type.");

        return;
    }
}

} // namespace llaisys::ops
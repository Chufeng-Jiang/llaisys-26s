#include "op.hpp"

#include "../../core/context/context.hpp"
#include "../../device/device_utils.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
#include "cuda/rms_norm_cuda.hpp"
#endif

#include <cmath>
#include <cstddef>

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_ARGUMENT(out != nullptr, "RMSNorm: output tensor must not be null.");
    CHECK_ARGUMENT(in != nullptr, "RMSNorm: input tensor must not be null.");
    CHECK_ARGUMENT(weight != nullptr, "RMSNorm: weight tensor must not be null.");

    CHECK_ARGUMENT(out->ndim() == 2, "RMSNorm: output tensor must be two-dimensional.");

    CHECK_ARGUMENT(in->ndim() == 2, "RMSNorm: input tensor must be two-dimensional.");

    CHECK_ARGUMENT(weight->ndim() == 1, "RMSNorm: weight tensor must be one-dimensional.");

    const std::size_t row_count = in->shape()[0];
    const std::size_t column_count = in->shape()[1];

    CHECK_ARGUMENT(column_count > 0, "RMSNorm: input row length must be greater than zero.");

    CHECK_ARGUMENT(
        out->shape()[0] == row_count, "RMSNorm: output row count must match input row count.");

    CHECK_ARGUMENT(
        out->shape()[1] == column_count,
        "RMSNorm: output column count must match input column count.");

    CHECK_ARGUMENT(
        weight->shape()[0] == column_count, "RMSNorm: weight length must match input row length.");

    CHECK_ARGUMENT(std::isfinite(eps), "RMSNorm: epsilon must be finite.");

    CHECK_ARGUMENT(eps >= 0.0F, "RMSNorm: epsilon must not be negative.");

    CHECK_ARGUMENT(
        out->dtype() == in->dtype(), "RMSNorm: output and input must use the same data type.");

    CHECK_ARGUMENT(
        weight->dtype() == in->dtype(), "RMSNorm: weight and input must use the same data type.");

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    CHECK_ARGUMENT(
        out->deviceType() == in->deviceType(),
        "RMSNorm: output and input must use the same device type.");

    CHECK_ARGUMENT(
        weight->deviceType() == in->deviceType(),
        "RMSNorm: weight and input must use the same device type.");

    CHECK_ARGUMENT(
        out->deviceId() == in->deviceId(),
        "RMSNorm: output and input must be located on the same device.");

    CHECK_ARGUMENT(
        weight->deviceId() == in->deviceId(),
        "RMSNorm: weight and input must be located on the same device.");

    CHECK_ARGUMENT(out->isContiguous(), "RMSNorm: output tensor must be contiguous.");

    CHECK_ARGUMENT(in->isContiguous(), "RMSNorm: input tensor must be contiguous.");

    CHECK_ARGUMENT(weight->isContiguous(), "RMSNorm: weight tensor must be contiguous.");

    const auto device_type = out->deviceType();

    if (device_type == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(), in->data(), weight->data(), eps, out->dtype(), row_count, column_count);
    }

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    if (device::is_cuda_compatible_gpu(device_type)) {
        core::context().setDevice(device_type, out->deviceId());

        auto &runtime = core::context().runtime();

        return cuda::rms_norm(
            out->data(), in->data(), weight->data(), eps, out->dtype(), row_count, column_count,
            runtime.stream());
    }
#endif

    CHECK_ARGUMENT(false, "RMSNorm: unsupported device type.");
}

} // namespace llaisys::ops

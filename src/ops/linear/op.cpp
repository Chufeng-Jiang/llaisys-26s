#include "op.hpp"

#include "../../core/context/context.hpp"
#include "../../device/device_utils.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
#include "cuda/linear_cuda.hpp"
#endif

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_ARGUMENT(out != nullptr, "Linear: output tensor must not be null.");
    CHECK_ARGUMENT(in != nullptr, "Linear: input tensor must not be null.");
    CHECK_ARGUMENT(weight != nullptr, "Linear: weight tensor must not be null.");

    CHECK_ARGUMENT(out->ndim() == 2, "Linear: output tensor must be two-dimensional.");
    CHECK_ARGUMENT(in->ndim() == 2, "Linear: input tensor must be two-dimensional.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight tensor must be two-dimensional.");

    const std::size_t row_count = in->shape()[0];
    const std::size_t input_features = in->shape()[1];
    const std::size_t output_features = weight->shape()[0];

    CHECK_ARGUMENT(
        weight->shape()[1] == input_features,
        "Linear: input feature count must match weight row length.");

    CHECK_ARGUMENT(
        out->shape()[0] == row_count, "Linear: output row count must match input row count.");

    CHECK_ARGUMENT(
        out->shape()[1] == output_features,
        "Linear: output feature count must match weight row count.");

    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias tensor must be one-dimensional.");

        CHECK_ARGUMENT(
            bias->shape()[0] == output_features,
            "Linear: bias length must match output feature count.");
    }

    CHECK_ARGUMENT(
        out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
        "Linear: output, input, and weight must use the same data type.");

    if (bias != nullptr) {
        CHECK_ARGUMENT(
            bias->dtype() == out->dtype(), "Linear: bias must use the same data type as output.");
    }

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        break;

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    CHECK_ARGUMENT(
        out->deviceType() == in->deviceType() && out->deviceType() == weight->deviceType(),
        "Linear: output, input, and weight must use the same device type.");

    CHECK_ARGUMENT(
        out->deviceId() == in->deviceId() && out->deviceId() == weight->deviceId(),
        "Linear: output, input, and weight must use the same device.");

    if (bias != nullptr) {
        CHECK_ARGUMENT(
            bias->deviceType() == out->deviceType() && bias->deviceId() == out->deviceId(),
            "Linear: bias must be located on the same device as output.");
    }

    CHECK_ARGUMENT(out->isContiguous(), "Linear: output tensor must be contiguous.");
    CHECK_ARGUMENT(in->isContiguous(), "Linear: input tensor must be contiguous.");
    CHECK_ARGUMENT(weight->isContiguous(), "Linear: weight tensor must be contiguous.");

    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->isContiguous(), "Linear: bias tensor must be contiguous.");
    }

    const auto device_type = out->deviceType();

    if (device_type == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(),
            out->dtype(), row_count, output_features, input_features);
    }

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    if (device::is_cuda_compatible_gpu(device_type)) {
        core::context().setDevice(device_type, out->deviceId());

        auto &runtime = core::context().runtime();

        return cuda::linear(
            out->data(), in->data(), weight->data(), bias == nullptr ? nullptr : bias->data(),
            out->dtype(), row_count, output_features, input_features, runtime.stream());
    }
#endif

    CHECK_ARGUMENT(false, "Linear: unsupported device type.");
}

} // namespace llaisys::ops

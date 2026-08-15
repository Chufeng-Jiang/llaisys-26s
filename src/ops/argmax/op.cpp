#include "op.hpp"

#include "../../core/context/context.hpp"
#include "../../device/device_utils.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
#include "cuda/argmax_cuda.hpp"
#endif

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_ARGUMENT(max_idx != nullptr, "Argmax: max_idx tensor must not be null.");
    CHECK_ARGUMENT(max_val != nullptr, "Argmax: max_val tensor must not be null.");
    CHECK_ARGUMENT(vals != nullptr, "Argmax: vals tensor must not be null.");

    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor.");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax: vals must not be empty.");
    CHECK_ARGUMENT(max_idx->ndim() == 1, "Argmax: max_idx must be a 1D tensor.");
    CHECK_ARGUMENT(max_idx->numel() == 1, "Argmax: max_idx must contain exactly one element.");
    CHECK_ARGUMENT(max_val->ndim() == 1, "Argmax: max_val must be a 1D tensor.");
    CHECK_ARGUMENT(max_val->numel() == 1, "Argmax: max_val must contain exactly one element.");

    CHECK_ARGUMENT(vals->isContiguous(), "Argmax: vals must be contiguous.");
    CHECK_ARGUMENT(max_idx->isContiguous(), "Argmax: max_idx must be contiguous.");
    CHECK_ARGUMENT(max_val->isContiguous(), "Argmax: max_val must be contiguous.");

    CHECK_ARGUMENT(
        max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must have Int64 data type.");
    CHECK_ARGUMENT(
        max_val->dtype() == vals->dtype(),
        "Argmax: max_val and vals must have the same data type.");

    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }

    CHECK_SAME_DEVICE(vals, max_idx, max_val);

    const auto device_type = vals->deviceType();

    if (device_type == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    if (device::is_cuda_compatible_gpu(device_type)) {
        core::context().setDevice(device_type, vals->deviceId());
        auto &runtime = core::context().runtime();

        return cuda::argmax(
            max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel(),
            vals->deviceId(), runtime.stream());
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops

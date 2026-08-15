#include "../../core/llaisys_core.hpp"
#include "../../device/device_utils.hpp"
#include "../../utils.hpp"

#include "cpu/add_cpu.hpp"
#include "op.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
#include "cuda/add_cuda.hpp"
#endif

namespace llaisys::ops {

void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());

    ASSERT(
        c->isContiguous() && a->isContiguous() && b->isContiguous(),
        "Add: all tensors must be contiguous.");

    const auto device_type = c->deviceType();

    if (device_type == LLAISYS_DEVICE_CPU) {
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    if (device::is_cuda_compatible_gpu(device_type)) {
        core::context().setDevice(device_type, c->deviceId());

        auto &runtime = core::context().runtime();

        return cuda::add(c->data(), a->data(), b->data(), c->dtype(), c->numel(), runtime.stream());
    }
#endif

    EXCEPTION_UNSUPPORTED_DEVICE;
}

} // namespace llaisys::ops
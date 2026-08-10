#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/add_cpu.hpp"
#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/add_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/add_metax.hpp"
#endif


namespace llaisys::ops {

void add(tensor_t c, tensor_t a, tensor_t b) {
    // ============================================================
    // Validation
    // ============================================================

    CHECK_SAME_DEVICE(c, a, b);

    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());

    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());

    ASSERT(
        c->isContiguous() && a->isContiguous() && b->isContiguous(),
        "Add: all tensors must be contiguous.");

    // ============================================================
    // Device dispatch
    // ============================================================

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());

#ifdef ENABLE_NVIDIA_API

    case LLAISYS_DEVICE_NVIDIA: {
        // Select the correct NVIDIA device before obtaining its
        // Runtime object and Runtime-owned CUDA stream.
        core::context().setDevice(c->deviceType(), c->deviceId());

        auto &runtime = core::context().runtime();

        return nvidia::add(
            c->data(), a->data(), b->data(), c->dtype(), c->numel(), runtime.stream());
    }

#endif

#ifdef ENABLE_METAX_API

	case LLAISYS_DEVICE_METAX: {
		core::context().setDevice(
			c->deviceType(),
			c->deviceId()
		);

		auto &runtime =
			core::context().runtime();

		return metax::add(
			c->data(),
			a->data(),
			b->data(),
			c->dtype(),
			c->numel(),
			runtime.stream()
		);
	}

#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
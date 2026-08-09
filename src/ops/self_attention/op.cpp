#include "op.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

#include <cmath>
#include <cstddef>

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // ============================================================
    // Null checks
    // ============================================================

    CHECK_ARGUMENT(attn_val != nullptr, "SelfAttention: output tensor must not be null.");

    CHECK_ARGUMENT(q != nullptr, "SelfAttention: query tensor must not be null.");

    CHECK_ARGUMENT(k != nullptr, "SelfAttention: key tensor must not be null.");

    CHECK_ARGUMENT(v != nullptr, "SelfAttention: value tensor must not be null.");

    // ============================================================
    // Dimension checks
    // ============================================================

    CHECK_ARGUMENT(
        attn_val->ndim() == 3, "SelfAttention: output tensor must be three-dimensional.");

    CHECK_ARGUMENT(q->ndim() == 3, "SelfAttention: query tensor must be three-dimensional.");

    CHECK_ARGUMENT(k->ndim() == 3, "SelfAttention: key tensor must be three-dimensional.");

    CHECK_ARGUMENT(v->ndim() == 3, "SelfAttention: value tensor must be three-dimensional.");

    // Q shape:
    // [seqlen, nhead, d]
    const std::size_t seqlen = q->shape()[0];

    const std::size_t nhead = q->shape()[1];

    const std::size_t d = q->shape()[2];

    // K shape:
    // [total_len, nkvhead, d]
    const std::size_t total_len = k->shape()[0];

    const std::size_t nkvhead = k->shape()[1];

    // V shape:
    // [total_len, nkvhead, dv]
    const std::size_t dv = v->shape()[2];

    // ============================================================
    // Basic shape-parameter checks
    // ============================================================

    CHECK_ARGUMENT(nhead > 0, "SelfAttention: query head count must be greater than zero.");

    CHECK_ARGUMENT(nkvhead > 0, "SelfAttention: KV head count must be greater than zero.");

    CHECK_ARGUMENT(d > 0, "SelfAttention: query/key head dimension must be greater than zero.");

    CHECK_ARGUMENT(dv > 0, "SelfAttention: value head dimension must be greater than zero.");

    // The current query tokens are assumed to be located at the end
    // of the KV cache.
    CHECK_ARGUMENT(
        total_len >= seqlen,
        "SelfAttention: total KV length must not be smaller than query "
        "sequence length.");

    // GQA/MQA requires every KV head to serve an equal number
    // of query heads.
    CHECK_ARGUMENT(
        nhead % nkvhead == 0,
        "SelfAttention: query head count must be divisible by KV head count.");

    // ============================================================
    // K shape checks
    // ============================================================

    CHECK_ARGUMENT(k->shape()[2] == d, "SelfAttention: query and key head dimensions must match.");

    // ============================================================
    // V shape checks
    // ============================================================

    CHECK_ARGUMENT(
        v->shape()[0] == total_len, "SelfAttention: key and value sequence lengths must match.");

    CHECK_ARGUMENT(
        v->shape()[1] == nkvhead, "SelfAttention: key and value head counts must match.");

    // ============================================================
    // Output shape checks
    // ============================================================

    CHECK_ARGUMENT(
        attn_val->shape()[0] == seqlen,
        "SelfAttention: output sequence length must match query sequence "
        "length.");

    CHECK_ARGUMENT(
        attn_val->shape()[1] == nhead,
        "SelfAttention: output head count must match query head count.");

    CHECK_ARGUMENT(
        attn_val->shape()[2] == dv,
        "SelfAttention: output head dimension must match value head "
        "dimension.");

    // ============================================================
    // Scale checks
    // ============================================================

    CHECK_ARGUMENT(std::isfinite(scale), "SelfAttention: scale must be finite.");

    // A zero scale is mathematically valid. It produces uniform
    // attention over the visible keys.
    //
    // Negative scale is also mathematically defined, so it is not
    // rejected here.

    // ============================================================
    // Data type checks
    // ============================================================

    CHECK_ARGUMENT(
        attn_val->dtype() == q->dtype(),
        "SelfAttention: output and query must use the same data type.");

    CHECK_ARGUMENT(
        k->dtype() == q->dtype(), "SelfAttention: key and query must use the same data type.");

    CHECK_ARGUMENT(
        v->dtype() == q->dtype(), "SelfAttention: value and query must use the same data type.");

    // ============================================================
    // Device-type checks
    // ============================================================

    CHECK_ARGUMENT(
        attn_val->deviceType() == q->deviceType(),
        "SelfAttention: output and query must use the same device type.");

    CHECK_ARGUMENT(
        k->deviceType() == q->deviceType(),
        "SelfAttention: key and query must use the same device type.");

    CHECK_ARGUMENT(
        v->deviceType() == q->deviceType(),
        "SelfAttention: value and query must use the same device type.");

    // ============================================================
    // Device-ID checks
    // ============================================================

    CHECK_ARGUMENT(
        attn_val->deviceId() == q->deviceId(),
        "SelfAttention: output and query must be located on the same device.");

    CHECK_ARGUMENT(
        k->deviceId() == q->deviceId(),
        "SelfAttention: key and query must be located on the same device.");

    CHECK_ARGUMENT(
        v->deviceId() == q->deviceId(),
        "SelfAttention: value and query must be located on the same device.");

    // ============================================================
    // Contiguity checks
    // ============================================================

    CHECK_ARGUMENT(attn_val->isContiguous(), "SelfAttention: output tensor must be contiguous.");

    CHECK_ARGUMENT(q->isContiguous(), "SelfAttention: query tensor must be contiguous.");

    CHECK_ARGUMENT(k->isContiguous(), "SelfAttention: key tensor must be contiguous.");

    CHECK_ARGUMENT(v->isContiguous(), "SelfAttention: value tensor must be contiguous.");

    // ============================================================
    // Empty-query fast path
    // ============================================================

    if (seqlen == 0) { return; }

    // When seqlen > 0, total_len >= seqlen guarantees that at
    // least one KV element exists.
    CHECK_ARGUMENT(total_len > 0, "SelfAttention: total KV length must be greater than zero.");

    // ============================================================
    // Device dispatch
    // ============================================================

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            scale,
            attn_val->dtype(),
            seqlen,
            nhead,
            dv,
            total_len,
            nkvhead,
            d);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        // Select the tensor's CUDA device before obtaining the
        // device-specific Runtime and CUDA stream.
        core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

        auto &runtime = core::context().runtime();

        return nvidia::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            scale,
            attn_val->dtype(),
            seqlen,
            nhead,
            dv,
            total_len,
            nkvhead,
            d,
            runtime.stream());
    }
#endif

    default:
        CHECK_ARGUMENT(false, "SelfAttention: unsupported device type.");

        return;
    }
}

} // namespace llaisys::ops
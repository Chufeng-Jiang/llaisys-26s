#include "op.hpp"
#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/embedding_metax.hpp"
#endif

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // ============================================================
    // Tensor checks
    // ============================================================

    CHECK_ARGUMENT(out != nullptr, "Embedding: output tensor must not be null.");

    CHECK_ARGUMENT(index != nullptr, "Embedding: index tensor must not be null.");

    CHECK_ARGUMENT(weight != nullptr, "Embedding: weight tensor must not be null.");

    // ============================================================
    // Shape checks
    // ============================================================

    // index shape:
    //
    //     [index_count]
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index tensor must be one-dimensional.");

    // weight shape:
    //
    //     [vocabulary_size, embedding_length]
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight tensor must be two-dimensional.");

    // out shape:
    //
    //     [index_count, embedding_length]
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: output tensor must be two-dimensional.");

    const std::size_t index_count = index->shape()[0];

    const std::size_t vocabulary_size = weight->shape()[0];

    const std::size_t embedding_length = weight->shape()[1];

    CHECK_ARGUMENT(embedding_length > 0, "Embedding: embedding length must be greater than zero.");

    CHECK_ARGUMENT(
        index_count == 0 || vocabulary_size > 0,
        "Embedding: vocabulary size must be greater than zero "
        "when indices are provided.");

    // Number of output rows must equal the number of indices.
    CHECK_ARGUMENT(
        out->shape()[0] == index_count, "Embedding: output row count must match index count.");

    // Output row length must equal weight row length.
    CHECK_ARGUMENT(
        out->shape()[1] == embedding_length, "Embedding: output embedding length must match "
                                             "weight embedding length.");

    // These checks are technically implied by the shape checks,
    // but make the intended relationship explicit.
    CHECK_ARGUMENT(
        index->numel() == index_count,
        "Embedding: index element count is inconsistent with its shape.");

    CHECK_ARGUMENT(
        out->numel() == index_count * embedding_length,
        "Embedding: output element count is inconsistent with its shape.");

    // ============================================================
    // Data-type checks
    // ============================================================

    // Embedding indices must be signed 64-bit integers.
    CHECK_ARGUMENT(
        index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must use int64 data type.");

    // The output stores rows copied directly from weight.
    CHECK_ARGUMENT(
        out->dtype() == weight->dtype(), "Embedding: output and weight tensors must have "
                                         "the same data type.");

    // ============================================================
    // Device checks
    // ============================================================

    CHECK_ARGUMENT(
        out->deviceType() == index->deviceType() && out->deviceType() == weight->deviceType(),
        "Embedding: all tensors must use the same device type.");

    CHECK_ARGUMENT(
        out->deviceId() == index->deviceId() && out->deviceId() == weight->deviceId(),
        "Embedding: all tensors must be located on the same device.");

    // ============================================================
    // Memory-layout checks
    // ============================================================

    // Both backend implementations calculate addresses using:
    //
    //     row * embedding_length
    //
    // Therefore, they currently require contiguous storage.
    CHECK_ARGUMENT(out->isContiguous(), "Embedding: output tensor must be contiguous.");

    CHECK_ARGUMENT(index->isContiguous(), "Embedding: index tensor must be contiguous.");

    CHECK_ARGUMENT(weight->isContiguous(), "Embedding: weight tensor must be contiguous.");

    // ============================================================
    // Backend dispatch
    // ============================================================

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(), index->data(), weight->data(), weight->dtype(), out->numel(),
            embedding_length, vocabulary_size);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
        // Select the device containing the tensors.
        core::context().setDevice(out->deviceType(), out->deviceId());

        auto &runtime = core::context().runtime();

        return nvidia::embedding(
            out->data(), index->data(), weight->data(), weight->dtype(), out->numel(),
            embedding_length, vocabulary_size, runtime.stream());
    }
#endif

#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX: {
        // Select the device containing the tensors.
        core::context().setDevice(out->deviceType(), out->deviceId());

        auto &runtime = core::context().runtime();

        return metax::embedding(
            out->data(), index->data(), weight->data(), weight->dtype(), out->numel(),
            embedding_length, vocabulary_size, runtime.stream());
    }
#endif

    default:
        CHECK_ARGUMENT(false, "Embedding: unsupported device type.");
    }
}

} // namespace llaisys::ops
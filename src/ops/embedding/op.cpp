#include "op.hpp"

#include "../../core/context/context.hpp"
#include "../../device/device_utils.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
#include "cuda/embedding_cuda.hpp"
#endif

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_ARGUMENT(out != nullptr, "Embedding: output tensor must not be null.");
    CHECK_ARGUMENT(index != nullptr, "Embedding: index tensor must not be null.");
    CHECK_ARGUMENT(weight != nullptr, "Embedding: weight tensor must not be null.");

    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index tensor must be one-dimensional.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight tensor must be two-dimensional.");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: output tensor must be two-dimensional.");

    const std::size_t index_count = index->shape()[0];
    const std::size_t vocabulary_size = weight->shape()[0];
    const std::size_t embedding_length = weight->shape()[1];

    CHECK_ARGUMENT(embedding_length > 0, "Embedding: embedding length must be greater than zero.");

    CHECK_ARGUMENT(
        index_count == 0 || vocabulary_size > 0,
        "Embedding: vocabulary size must be greater than zero when indices are provided.");

    CHECK_ARGUMENT(
        out->shape()[0] == index_count, "Embedding: output row count must match index count.");

    CHECK_ARGUMENT(
        out->shape()[1] == embedding_length,
        "Embedding: output embedding length must match weight embedding length.");

    CHECK_ARGUMENT(
        index->numel() == index_count,
        "Embedding: index element count is inconsistent with its shape.");

    CHECK_ARGUMENT(
        out->numel() == index_count * embedding_length,
        "Embedding: output element count is inconsistent with its shape.");

    CHECK_ARGUMENT(
        index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index tensor must use int64 data type.");

    CHECK_ARGUMENT(
        out->dtype() == weight->dtype(),
        "Embedding: output and weight tensors must have the same data type.");

    CHECK_ARGUMENT(
        out->deviceType() == index->deviceType() && out->deviceType() == weight->deviceType(),
        "Embedding: all tensors must use the same device type.");

    CHECK_ARGUMENT(
        out->deviceId() == index->deviceId() && out->deviceId() == weight->deviceId(),
        "Embedding: all tensors must be located on the same device.");

    CHECK_ARGUMENT(out->isContiguous(), "Embedding: output tensor must be contiguous.");
    CHECK_ARGUMENT(index->isContiguous(), "Embedding: index tensor must be contiguous.");
    CHECK_ARGUMENT(weight->isContiguous(), "Embedding: weight tensor must be contiguous.");

    const auto device_type = out->deviceType();

    if (device_type == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(), index->data(), weight->data(), weight->dtype(), out->numel(),
            embedding_length, vocabulary_size);
    }

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    if (device::is_cuda_compatible_gpu(device_type)) {
        core::context().setDevice(device_type, out->deviceId());
        auto &runtime = core::context().runtime();

        return cuda::embedding(
            out->data(), index->data(), weight->data(), weight->dtype(), out->numel(),
            embedding_length, vocabulary_size, runtime.stream());
    }
#endif

    CHECK_ARGUMENT(false, "Embedding: unsupported device type.");
}

} // namespace llaisys::ops

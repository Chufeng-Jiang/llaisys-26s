#include "../../core/context/context.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
#include "op.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

#include <cmath>
#include <cstddef>

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
  // ============================================================
  // Null checks
  // ============================================================

  CHECK_ARGUMENT(out != nullptr, "RoPE: output tensor must not be null.");
  CHECK_ARGUMENT(in != nullptr, "RoPE: input tensor must not be null.");
  CHECK_ARGUMENT(pos_ids != nullptr, "RoPE: position ID tensor must not be null.");

  // ============================================================
  // Dimension checks
  // ============================================================

  CHECK_ARGUMENT(out->ndim() == 3, "RoPE: output tensor must be three-dimensional.");
  CHECK_ARGUMENT(in->ndim() == 3, "RoPE: input tensor must be three-dimensional.");
  CHECK_ARGUMENT(pos_ids->ndim() == 1, "RoPE: position ID tensor must be one-dimensional.");

  const std::size_t sequence_length = in->shape()[0];
  const std::size_t head_count = in->shape()[1];
  const std::size_t head_dimension = in->shape()[2];

  CHECK_ARGUMENT(head_dimension > 0, "RoPE: head dimension must be greater than zero.");
  CHECK_ARGUMENT(head_dimension % 2 == 0, "RoPE: head dimension must be even.");

  // Output shape must exactly match input shape.
  CHECK_ARGUMENT(out->shape()[0] == sequence_length, "RoPE: output sequence length must match input sequence length.");
  CHECK_ARGUMENT(out->shape()[1] == head_count, "RoPE: output head count must match input head count.");
  CHECK_ARGUMENT(out->shape()[2] == head_dimension, "RoPE: output head dimension must match input head dimension.");

  // One position ID is required for every input token.
  CHECK_ARGUMENT(pos_ids->shape()[0] == sequence_length, "RoPE: position ID count must match input sequence length.");

  // ============================================================
  // Theta checks
  // ============================================================

  CHECK_ARGUMENT(std::isfinite(theta), "RoPE: theta must be finite.");
  CHECK_ARGUMENT(theta > 0.0F, "RoPE: theta must be greater than zero.");

  // ============================================================
  // Data type checks
  // ============================================================

  CHECK_ARGUMENT(out->dtype() == in->dtype(), "RoPE: output and input must use the same data type.");
  CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: position IDs must use the Int64 data type.");

  // ============================================================
  // Device checks
  // ============================================================

  CHECK_ARGUMENT(out->deviceType() == in->deviceType(), "RoPE: output and input must use the same device type.");
  CHECK_ARGUMENT(pos_ids->deviceType() == in->deviceType(), "RoPE: position IDs and input must use the same device type.");
  CHECK_ARGUMENT(out->deviceId() == in->deviceId(), "RoPE: output and input must be located on the same device.");
  CHECK_ARGUMENT(pos_ids->deviceId() == in->deviceId(), "RoPE: position IDs and input must be located on the same device.");

  // ============================================================
  // Contiguity checks
  // ============================================================

  CHECK_ARGUMENT(out->isContiguous(), "RoPE: output tensor must be contiguous.");
  CHECK_ARGUMENT(in->isContiguous(), "RoPE: input tensor must be contiguous.");
  CHECK_ARGUMENT(pos_ids->isContiguous(), "RoPE: position ID tensor must be contiguous.");

  // ============================================================
  // Device dispatch
  // ============================================================

  switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
      return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), sequence_length, head_count, head_dimension);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA: {
      // Select the correct CUDA device before retrieving its Runtime.
      core::context().setDevice(out->deviceType(), out->deviceId());

      auto &runtime = core::context().runtime();

      return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), sequence_length, head_count, head_dimension, runtime.stream());
    }
#endif

    default:
      CHECK_ARGUMENT(false, "RoPE: unsupported device type.");
  }
}

}  // namespace llaisys::ops
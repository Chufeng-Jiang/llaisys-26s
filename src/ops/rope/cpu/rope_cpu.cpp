#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "../../../device/cpu/cpu_common.hpp"
#include "../../../utils.hpp"
#include "rope_cpu.hpp"

namespace {

using llaisys::device::cpu::OPENMP_THRESHOLD;

struct RopeWorkspace {
  std::size_t cached_dimension{0};
  float cached_theta{0.0F};
  bool frequency_cache_valid{false};

  // frequency_denominator[k] = theta^(2k/d)
  // Store the denominator rather than its reciprocal so the angle uses
  // the same Float32 operation order as the PyTorch reference.
  std::vector<float> frequency_denominator;

  // For every token, store d/2 cosine values followed by d/2 sine values.
  // Capacity is reused across repeated inference calls on the same host thread.
  std::vector<float> trig_cache;
};

RopeWorkspace &rope_workspace() {
  thread_local RopeWorkspace workspace;
  return workspace;
}

bool multiplication_fits(std::size_t left, std::size_t right) { return left == 0 || right <= std::numeric_limits<std::size_t>::max() / left; }

void prepare_frequency_denominator(RopeWorkspace &workspace, float theta, std::size_t dimension) {
  if (workspace.frequency_cache_valid && workspace.cached_dimension == dimension && workspace.cached_theta == theta) {
    return;
  }

  const std::size_t half_dimension = dimension / 2;
  workspace.frequency_denominator.resize(half_dimension);

  for (std::size_t index = 0; index < half_dimension; ++index) {
    // Match the PyTorch reference exactly at the operation level:
    //
    //     denominator = theta ** (2 * index / dimension)
    //     angle       = position / denominator
    //
    // Precomputing the reciprocal and multiplying is mathematically
    // equivalent, but it follows a different Float32 rounding path.
    const float exponent = 2.0F * static_cast<float>(index) / static_cast<float>(dimension);

    workspace.frequency_denominator[index] = std::pow(theta, exponent);
  }

  workspace.cached_dimension = dimension;
  workspace.cached_theta = theta;
  workspace.frequency_cache_valid = true;
}

inline void calculate_sine_cosine(float angle, float &sine, float &cosine) {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_sincosf(angle, &sine, &cosine);
#else
  sine = std::sin(angle);
  cosine = std::cos(angle);
#endif
}

template <typename T>
inline float load_as_float(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else {
    return llaisys::utils::cast<float>(value);
  }
}

template <typename T>
inline T store_from_float(float value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else {
    return llaisys::utils::cast<T>(value);
  }
}

template <typename T>
void apply_rope_vector(T *out, const T *in, const float *cosine, const float *sine, std::size_t half_dimension) {
  // Read both values before either output location is overwritten. This keeps
  // exact in-place execution (out == in) safe.
#pragma omp simd
  for (std::size_t index = 0; index < half_dimension; ++index) {
    const float first = load_as_float(in[index]);

    const float second = load_as_float(in[index + half_dimension]);

    const float cosine_value = cosine[index];
    const float sine_value = sine[index];

    out[index] = store_from_float<T>(first * cosine_value - second * sine_value);

    out[index + half_dimension] = store_from_float<T>(second * cosine_value + first * sine_value);
  }
}

template <typename T>
void rope_impl(T *out, const T *in, const std::int64_t *pos_ids, float theta, std::size_t sequence_length, std::size_t head_count, std::size_t dimension) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>, "RoPE: unsupported CPU element type.");

  CHECK_ARGUMENT(dimension > 0, "RoPE: head dimension must be greater than zero.");
  CHECK_ARGUMENT(dimension % 2 == 0, "RoPE: head dimension must be even.");
  CHECK_ARGUMENT(std::isfinite(theta) && theta > 0.0F, "RoPE: theta must be finite and greater than zero.");
  CHECK_ARGUMENT(multiplication_fits(sequence_length, head_count), "RoPE: sequence length and head count overflow size_t.");

  const std::size_t vector_count = sequence_length * head_count;

  CHECK_ARGUMENT(multiplication_fits(vector_count, dimension), "RoPE: tensor element count overflows size_t.");

  const std::size_t element_count = vector_count * dimension;

  CHECK_ARGUMENT(element_count == 0 || out != nullptr, "RoPE: output pointer must not be null.");
  CHECK_ARGUMENT(element_count == 0 || in != nullptr, "RoPE: input pointer must not be null.");
  CHECK_ARGUMENT(sequence_length == 0 || pos_ids != nullptr, "RoPE: position ID pointer must not be null.");

  if (element_count == 0) {
    return;
  }

  const std::size_t half_dimension = dimension / 2;

  CHECK_ARGUMENT(multiplication_fits(sequence_length, dimension), "RoPE: trigonometric cache size overflows size_t.");

  RopeWorkspace &workspace = rope_workspace();

  prepare_frequency_denominator(workspace, theta, dimension);

  // One row of d floats stores both cosine and sine values:
  // [cos(0..d/2), sin(0..d/2)].
  workspace.trig_cache.resize(sequence_length * dimension);

  const bool use_openmp = element_count >= OPENMP_THRESHOLD && vector_count > 1;

  // Use one OpenMP team for both phases. The implicit barrier after the first
  // omp-for guarantees that every trigonometric row is ready before output
  // vectors begin reading it.
#pragma omp parallel if (use_openmp)
  {
#pragma omp for schedule(static)
    for (std::size_t token = 0; token < sequence_length; ++token) {
      float *token_trig = workspace.trig_cache.data() + token * dimension;
      float *token_cosine = token_trig;
      float *token_sine = token_trig + half_dimension;
      const float position = static_cast<float>(pos_ids[token]);

#pragma omp simd
      for (std::size_t index = 0; index < half_dimension; ++index) {
        const float angle = position / workspace.frequency_denominator[index];
        calculate_sine_cosine(angle, token_sine[index], token_cosine[index]);
      }
    }

#pragma omp for collapse(2) schedule(static)
    for (std::size_t token = 0; token < sequence_length; ++token) {
      for (std::size_t head = 0; head < head_count; ++head) {
        const std::size_t vector_index = token * head_count + head;
        T *out_vector = out + vector_index * dimension;
        const T *in_vector = in + vector_index * dimension;
        const float *token_trig = workspace.trig_cache.data() + token * dimension;
        const float *token_cosine = token_trig;
        const float *token_sine = token_trig + half_dimension;
        apply_rope_vector(out_vector, in_vector, token_cosine, token_sine, half_dimension);
      }
    }
  }
}

}  // namespace

namespace llaisys::ops::cpu {

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, std::size_t seqlen, std::size_t nhead,
          std::size_t d) {
  switch (type) {
    case LLAISYS_DTYPE_F32:
      return rope_impl<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const std::int64_t *>(pos_ids), theta,
                              seqlen, nhead, d);

    case LLAISYS_DTYPE_F16:
      return rope_impl<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                                        reinterpret_cast<const std::int64_t *>(pos_ids), theta, seqlen, nhead, d);

    case LLAISYS_DTYPE_BF16:
      return rope_impl<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                                        reinterpret_cast<const std::int64_t *>(pos_ids), theta, seqlen, nhead, d);

    default:
      EXCEPTION_UNSUPPORTED_DATATYPE(type);
  }
}

}  // namespace llaisys::ops::cpu
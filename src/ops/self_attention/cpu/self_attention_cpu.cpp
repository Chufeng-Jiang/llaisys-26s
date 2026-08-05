#include "self_attention_cpu.hpp"

#include "../../../device/cpu/cpu_common.hpp"
#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

using llaisys::device::cpu::OPENMP_THRESHOLD;

inline constexpr std::size_t MAX_QUERY_TILE_SIZE = 16;
inline constexpr std::size_t TARGET_WORKSPACE_BYTES = 64 * 1024;

struct AttentionWorkspace {
	std::vector<float> query;
	std::vector<float> key;
	std::vector<float> value;
	std::vector<float> output;
	std::vector<float> maximum;
	std::vector<float> denominator;
};

std::size_t checked_product(
	std::size_t left,
	std::size_t right,
	const char *message
) {
	CHECK_ARGUMENT(
		left == 0
			|| right <= std::numeric_limits<std::size_t>::max() / left,
		message
	);

	return left * right;
}

std::size_t saturating_sum(
	std::size_t left,
	std::size_t right
) {
	if (right > std::numeric_limits<std::size_t>::max() - left) {
		return std::numeric_limits<std::size_t>::max();
	}

	return left + right;
}

std::size_t saturating_product(
	std::size_t left,
	std::size_t right
) {
	if (
		left != 0
		&& right > std::numeric_limits<std::size_t>::max() / left
	) {
		return std::numeric_limits<std::size_t>::max();
	}

	return left * right;
}

std::size_t choose_query_tile_size(
	std::size_t query_dimension,
	std::size_t value_dimension
) {
	const std::size_t floats_per_query =
		query_dimension
		+ value_dimension
		+ 2;

	if (
		floats_per_query
		> std::numeric_limits<std::size_t>::max() / sizeof(float)
	) {
		return 1;
	}

	const std::size_t bytes_per_query =
		floats_per_query * sizeof(float);

	if (bytes_per_query == 0) {
		return 1;
	}

	const std::size_t candidate =
		TARGET_WORKSPACE_BYTES / bytes_per_query;

	return std::clamp<std::size_t>(
		candidate,
		1,
		MAX_QUERY_TILE_SIZE
	);
}

template <typename T>
void convert_to_float(
	float *destination,
	const T *source,
	std::size_t count
) {
#pragma omp simd
	for (std::size_t index = 0; index < count; ++index) {
		destination[index] =
			llaisys::utils::cast<float>(source[index]);
	}
}

float dot_product(
	const float *left,
	const float *right,
	std::size_t count
) {
	float result = 0.0F;

#pragma omp simd reduction(+ : result)
	for (std::size_t index = 0; index < count; ++index) {
		result += left[index] * right[index];
	}

	return result;
}

template <typename T>
void store_from_float(
	T *destination,
	const float *source,
	std::size_t count
) {
#pragma omp simd
	for (std::size_t index = 0; index < count; ++index) {
		if constexpr (std::is_same_v<T, float>) {
			destination[index] = source[index];
		} else {
			destination[index] =
				llaisys::utils::cast<T>(source[index]);
		}
	}
}

template <typename T>
void process_attention_tile(
	AttentionWorkspace &workspace,
	T *attn_val,
	const T *q,
	const T *k,
	const T *v,
	float scale,
	std::size_t query_start,
	std::size_t query_count,
	std::size_t head,
	std::size_t sequence_length,
	std::size_t head_count,
	std::size_t value_dimension,
	std::size_t total_length,
	std::size_t kv_head_count,
	std::size_t query_dimension,
	std::size_t group_size
) {
	const std::size_t kv_head =
		head / group_size;

	const std::size_t output_count =
		query_count * value_dimension;

	workspace.output.resize(output_count);
	workspace.maximum.resize(query_count);
	workspace.denominator.resize(query_count);

	std::fill(
		workspace.output.begin(),
		workspace.output.begin()
			+ static_cast<std::ptrdiff_t>(output_count),
		0.0F
	);

	std::fill(
		workspace.maximum.begin(),
		workspace.maximum.begin()
			+ static_cast<std::ptrdiff_t>(query_count),
		-std::numeric_limits<float>::infinity()
	);

	std::fill(
		workspace.denominator.begin(),
		workspace.denominator.begin()
			+ static_cast<std::ptrdiff_t>(query_count),
		0.0F
	);

	if constexpr (!std::is_same_v<T, float>) {
		workspace.query.resize(
			query_count * query_dimension
		);

		for (
			std::size_t local_query = 0;
			local_query < query_count;
			++local_query
		) {
			const std::size_t query_index =
				query_start + local_query;

			const T *source =
				q
				+ (query_index * head_count + head)
					* query_dimension;

			convert_to_float(
				workspace.query.data()
					+ local_query * query_dimension,
				source,
				query_dimension
			);
		}

		workspace.key.resize(query_dimension);
		workspace.value.resize(value_dimension);
	}

	const std::size_t prefix_length =
		total_length - sequence_length;

	// The last query in this tile may attend through this exclusive K/V index.
	const std::size_t key_end =
		prefix_length + query_start + query_count;

	for (
		std::size_t key_index = 0;
		key_index < key_end;
		++key_index
	) {
		const T *key_source =
			k
			+ (key_index * kv_head_count + kv_head)
				* query_dimension;

		const T *value_source =
			v
			+ (key_index * kv_head_count + kv_head)
				* value_dimension;

		const float *key_float = nullptr;
		const float *value_float = nullptr;

		if constexpr (std::is_same_v<T, float>) {
			key_float = key_source;
			value_float = value_source;
		} else {
			// A K/V row is shared by every query in this tile. Convert it only
			// once instead of repeating the conversion for each query.
			convert_to_float(
				workspace.key.data(),
				key_source,
				query_dimension
			);

			convert_to_float(
				workspace.value.data(),
				value_source,
				value_dimension
			);

			key_float = workspace.key.data();
			value_float = workspace.value.data();
		}

		// Query i may attend key j exactly when:
		//
		//   j <= prefix_length + i
		//
		// Determine the first query in the tile that satisfies this condition,
		// avoiding a causal-mask branch inside every query iteration.
		const std::size_t tile_causal_start =
			prefix_length + query_start;

		const std::size_t first_local_query =
			key_index <= tile_causal_start
				? 0
				: key_index - tile_causal_start;

		for (
			std::size_t local_query = first_local_query;
			local_query < query_count;
			++local_query
		) {
			const float *query_float = nullptr;

			if constexpr (std::is_same_v<T, float>) {
				const std::size_t query_index =
					query_start + local_query;

				query_float =
					q
					+ (query_index * head_count + head)
						* query_dimension;
			} else {
				query_float =
					workspace.query.data()
					+ local_query * query_dimension;
			}

			const float score =
				dot_product(
					query_float,
					key_float,
					query_dimension
				)
				* scale;

			float &maximum =
				workspace.maximum[local_query];

			float &denominator =
				workspace.denominator[local_query];

			float *output =
				workspace.output.data()
				+ local_query * value_dimension;

			if (score > maximum) {
				// Rescale the previous online-softmax state to the new maximum.
				// For the first key, maximum is -inf and old_scale becomes 0.
				const float old_scale =
					std::exp(maximum - score);

				denominator =
					denominator * old_scale + 1.0F;

#pragma omp simd
				for (
					std::size_t value_index = 0;
					value_index < value_dimension;
					++value_index
				) {
					output[value_index] =
						output[value_index] * old_scale
						+ value_float[value_index];
				}

				maximum = score;
			} else {
				const float weight =
					std::exp(score - maximum);

				denominator += weight;

#pragma omp simd
				for (
					std::size_t value_index = 0;
					value_index < value_dimension;
					++value_index
				) {
					output[value_index] +=
						value_float[value_index] * weight;
				}
			}
		}
	}

	for (
		std::size_t local_query = 0;
		local_query < query_count;
		++local_query
	) {
		const float denominator =
			workspace.denominator[local_query];

		// total_length >= sequence_length and every query includes at least its
		// own current K/V entry, so denominator is guaranteed to be positive.
		const float inverse_denominator =
			1.0F / denominator;

		float *output =
			workspace.output.data()
			+ local_query * value_dimension;

#pragma omp simd
		for (
			std::size_t value_index = 0;
			value_index < value_dimension;
			++value_index
		) {
			output[value_index] *= inverse_denominator;
		}

		const std::size_t query_index =
			query_start + local_query;

		T *destination =
			attn_val
			+ (query_index * head_count + head)
				* value_dimension;

		store_from_float(
			destination,
			output,
			value_dimension
		);
	}
}

template <typename T>
void self_attention_impl(
	T *attn_val,
	const T *q,
	const T *k,
	const T *v,
	float scale,
	std::size_t sequence_length,
	std::size_t head_count,
	std::size_t value_dimension,
	std::size_t total_length,
	std::size_t kv_head_count,
	std::size_t query_dimension
) {
	CHECK_ARGUMENT(
		head_count > 0,
		"SelfAttention: query head count must be greater than zero."
	);

	CHECK_ARGUMENT(
		kv_head_count > 0,
		"SelfAttention: K/V head count must be greater than zero."
	);

	CHECK_ARGUMENT(
		head_count % kv_head_count == 0,
		"SelfAttention: query head count must be divisible by K/V head count."
	);

	CHECK_ARGUMENT(
		total_length >= sequence_length,
		"SelfAttention: total K/V length must not be shorter than query length."
	);

	CHECK_ARGUMENT(
		std::isfinite(scale),
		"SelfAttention: scale must be finite."
	);

	if (sequence_length == 0) {
		return;
	}

	CHECK_ARGUMENT(
		query_dimension > 0,
		"SelfAttention: query/key dimension must be greater than zero."
	);

	CHECK_ARGUMENT(
		value_dimension > 0,
		"SelfAttention: value dimension must be greater than zero."
	);

	CHECK_ARGUMENT(
		total_length > 0,
		"SelfAttention: total K/V length must be greater than zero."
	);

	const std::size_t output_elements =
		checked_product(
			checked_product(
				sequence_length,
				head_count,
				"SelfAttention: output vector count overflows size_t."
			),
			value_dimension,
			"SelfAttention: output element count overflows size_t."
		);

	const std::size_t query_elements =
		checked_product(
			checked_product(
				sequence_length,
				head_count,
				"SelfAttention: query vector count overflows size_t."
			),
			query_dimension,
			"SelfAttention: query element count overflows size_t."
		);

	const std::size_t key_elements =
		checked_product(
			checked_product(
				total_length,
				kv_head_count,
				"SelfAttention: key vector count overflows size_t."
			),
			query_dimension,
			"SelfAttention: key element count overflows size_t."
		);

	const std::size_t value_elements =
		checked_product(
			checked_product(
				total_length,
				kv_head_count,
				"SelfAttention: value vector count overflows size_t."
			),
			value_dimension,
			"SelfAttention: value element count overflows size_t."
		);

	CHECK_ARGUMENT(
		output_elements == 0 || attn_val != nullptr,
		"SelfAttention: output pointer must not be null."
	);

	CHECK_ARGUMENT(
		query_elements == 0 || q != nullptr,
		"SelfAttention: query pointer must not be null."
	);

	CHECK_ARGUMENT(
		key_elements == 0 || k != nullptr,
		"SelfAttention: key pointer must not be null."
	);

	CHECK_ARGUMENT(
		value_elements == 0 || v != nullptr,
		"SelfAttention: value pointer must not be null."
	);

	const std::size_t query_tile_size =
		choose_query_tile_size(
			query_dimension,
			value_dimension
		);

	const std::size_t tile_count =
		sequence_length / query_tile_size
		+ static_cast<std::size_t>(
			sequence_length % query_tile_size != 0
		);

	const std::size_t task_count =
		checked_product(
			tile_count,
			head_count,
			"SelfAttention: task count overflows size_t."
		);

	CHECK_ARGUMENT(
		task_count
			<= static_cast<std::size_t>(
				std::numeric_limits<std::ptrdiff_t>::max()
			),
		"SelfAttention: task count exceeds OpenMP loop range."
	);

	const std::size_t estimated_work =
		saturating_product(
			saturating_product(
				sequence_length,
				head_count
			),
			saturating_product(
				total_length,
				saturating_sum(
					query_dimension,
					value_dimension
				)
			)
		);

	const bool use_openmp =
		task_count > 1
		&& estimated_work >= OPENMP_THRESHOLD;

	const std::size_t group_size =
		head_count / kv_head_count;

#pragma omp parallel if(use_openmp)
	{
		// One workspace per worker avoids allocations for every query tile while
		// keeping different workers independent.
		AttentionWorkspace workspace;

#pragma omp for schedule(guided)
		for (
			std::ptrdiff_t task = 0;
			task < static_cast<std::ptrdiff_t>(task_count);
			++task
		) {
			const std::size_t tile =
				static_cast<std::size_t>(task)
				/ head_count;

			const std::size_t head =
				static_cast<std::size_t>(task)
				% head_count;

			const std::size_t query_start =
				tile * query_tile_size;

			const std::size_t query_count =
				std::min(
					query_tile_size,
					sequence_length - query_start
				);

			process_attention_tile(
				workspace,
				attn_val,
				q,
				k,
				v,
				scale,
				query_start,
				query_count,
				head,
				sequence_length,
				head_count,
				value_dimension,
				total_length,
				kv_head_count,
				query_dimension,
				group_size
			);
		}
	}
}

} // namespace

namespace llaisys::ops::cpu {

void self_attention(
	std::byte *attn_val,
	const std::byte *q,
	const std::byte *k,
	const std::byte *v,
	float scale,
	llaisysDataType_t type,
	std::size_t seqlen,
	std::size_t nhead,
	std::size_t dv,
	std::size_t total_len,
	std::size_t nkvhead,
	std::size_t d
) {
	switch (type) {
	case LLAISYS_DTYPE_F32:
		return self_attention_impl(
			reinterpret_cast<float *>(attn_val),
			reinterpret_cast<const float *>(q),
			reinterpret_cast<const float *>(k),
			reinterpret_cast<const float *>(v),
			scale,
			seqlen,
			nhead,
			dv,
			total_len,
			nkvhead,
			d
		);

	case LLAISYS_DTYPE_F16:
		return self_attention_impl(
			reinterpret_cast<llaisys::fp16_t *>(attn_val),
			reinterpret_cast<const llaisys::fp16_t *>(q),
			reinterpret_cast<const llaisys::fp16_t *>(k),
			reinterpret_cast<const llaisys::fp16_t *>(v),
			scale,
			seqlen,
			nhead,
			dv,
			total_len,
			nkvhead,
			d
		);

	case LLAISYS_DTYPE_BF16:
		return self_attention_impl(
			reinterpret_cast<llaisys::bf16_t *>(attn_val),
			reinterpret_cast<const llaisys::bf16_t *>(q),
			reinterpret_cast<const llaisys::bf16_t *>(k),
			reinterpret_cast<const llaisys::bf16_t *>(v),
			scale,
			seqlen,
			nhead,
			dv,
			total_len,
			nkvhead,
			d
		);

	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}

} // namespace llaisys::ops::cpu
#include "self_attention_nvidia.cuh"

#include "../../../device/nvidia/nvidia_common.cuh"
#include "../../../device/nvidia/nvidia_dtype.cuh"
#include "../../../utils.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#if defined(ENABLE_CUDNN_API) \
	&& __has_include(<cudnn.h>) \
	&& __has_include(<cudnn_frontend.h>)
#define LLAISYS_HAS_CUDNN_SDPA 1
#include <cudnn.h>
#include <cudnn_frontend.h>
#else
#define LLAISYS_HAS_CUDNN_SDPA 0
#endif

namespace {

using llaisys::device::nvidia::CUDA_BLOCK_SIZE;
using llaisys::device::nvidia::CUDA_DEFAULT_MAX_GRID_SIZE;
using llaisys::device::nvidia::CUDA_WARP_SIZE;
using llaisys::device::nvidia::from_float;
using llaisys::device::nvidia::get_capped_grid_size;
using llaisys::device::nvidia::to_float;
using llaisys::utils::checked_product;

inline constexpr std::size_t TILE_KV = 128;
inline constexpr std::size_t WARP_COUNT =
	CUDA_BLOCK_SIZE / CUDA_WARP_SIZE;

static_assert(
	CUDA_BLOCK_SIZE % CUDA_WARP_SIZE == 0,
	"SelfAttention: CUDA block size must be warp aligned."
);

static_assert(
	WARP_COUNT
		<= llaisys::device::nvidia::CUDA_MAX_WARPS_PER_BLOCK,
	"SelfAttention: warp count exceeds the common CUDA limit."
);

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
	for (
		int offset = static_cast<int>(CUDA_WARP_SIZE / 2);
		offset > 0;
		offset /= 2
	) {
		value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
	}

	return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
	for (
		int offset = static_cast<int>(CUDA_WARP_SIZE / 2);
		offset > 0;
		offset /= 2
	) {
		value = fmaxf(
			value,
			__shfl_down_sync(0xFFFFFFFFU, value, offset)
		);
	}

	return value;
}

__device__ __forceinline__ float block_reduce_sum(
	float value,
	float *shared_reduction
) {
	const unsigned int lane =
		threadIdx.x % CUDA_WARP_SIZE;
	const unsigned int warp =
		threadIdx.x / CUDA_WARP_SIZE;

	value = warp_reduce_sum(value);

	if (lane == 0) {
		shared_reduction[warp] = value;
	}

	__syncthreads();

	float block_value = 0.0F;

	if (warp == 0) {
		block_value = lane < WARP_COUNT
			? shared_reduction[lane]
			: 0.0F;

		block_value = warp_reduce_sum(block_value);

		if (lane == 0) {
			shared_reduction[0] = block_value;
		}
	}

	__syncthreads();
	return shared_reduction[0];
}

__device__ __forceinline__ float block_reduce_max(
	float value,
	float *shared_reduction
) {
	const unsigned int lane =
		threadIdx.x % CUDA_WARP_SIZE;
	const unsigned int warp =
		threadIdx.x / CUDA_WARP_SIZE;

	value = warp_reduce_max(value);

	if (lane == 0) {
		shared_reduction[warp] = value;
	}

	__syncthreads();

	float block_value = -FLT_MAX;

	if (warp == 0) {
		block_value = lane < WARP_COUNT
			? shared_reduction[lane]
			: -FLT_MAX;

		block_value = warp_reduce_max(block_value);

		if (lane == 0) {
			shared_reduction[0] = block_value;
		}
	}

	__syncthreads();
	return shared_reduction[0];
}

// Portable fused fallback used for:
//
// - FP32;
// - GPUs or dimensions outside the cuDNN SDPA support surface;
// - builds without cuDNN Frontend;
// - a cuDNN graph build failure.
//
// One block processes one or more (query, query-head) tasks. The full output
// accumulator lives in shared memory, so dv is not limited by blockDim.x.
template <typename T>
__global__ void self_attention_fallback_kernel(
	T *__restrict__ attn_val,
	const T *__restrict__ q,
	const T *__restrict__ k,
	const T *__restrict__ v,
	float scale,
	std::size_t seqlen,
	std::size_t nhead,
	std::size_t dv,
	std::size_t total_len,
	std::size_t nkvhead,
	std::size_t d
) {
	extern __shared__ float shared[];

	float *shared_q = shared;
	float *shared_scores = shared_q + d;
	float *shared_output = shared_scores + TILE_KV;
	float *shared_reduction = shared_output + dv;
	float *shared_state = shared_reduction + WARP_COUNT;

	const std::size_t task_count = seqlen * nhead;
	const std::size_t group_size = nhead / nkvhead;
	const std::size_t prefix_length = total_len - seqlen;
	const unsigned int warp = threadIdx.x / CUDA_WARP_SIZE;
	const unsigned int lane = threadIdx.x % CUDA_WARP_SIZE;

	for (
		std::size_t task = blockIdx.x;
		task < task_count;
		task += gridDim.x
	) {
		const std::size_t query_index = task / nhead;
		const std::size_t query_head = task % nhead;
		const std::size_t kv_head = query_head / group_size;
		const std::size_t causal_length =
			prefix_length + query_index + 1;

		const T *query =
			q + (query_index * nhead + query_head) * d;

		for (
			std::size_t index = threadIdx.x;
			index < d;
			index += blockDim.x
		) {
			shared_q[index] = to_float<T>(query[index]);
		}

		for (
			std::size_t index = threadIdx.x;
			index < dv;
			index += blockDim.x
		) {
			shared_output[index] = 0.0F;
		}

		if (threadIdx.x == 0) {
			shared_state[0] = -FLT_MAX; // running max
			shared_state[1] = 0.0F;    // running sum
		}

		__syncthreads();

		for (
			std::size_t key_start = 0;
			key_start < causal_length;
			key_start += TILE_KV
		) {
			const std::size_t remaining_keys =
				causal_length - key_start;

			const std::size_t tile_length =
				remaining_keys < TILE_KV
					? remaining_keys
					: TILE_KV;

			// Each warp computes one score at a time.
			for (
				std::size_t local_key = warp;
				local_key < tile_length;
				local_key += WARP_COUNT
			) {
				const std::size_t key_index =
					key_start + local_key;

				const T *key =
					k + (key_index * nkvhead + kv_head) * d;

				float dot = 0.0F;

				for (
					std::size_t index = lane;
					index < d;
					index += CUDA_WARP_SIZE
				) {
					dot += shared_q[index] * to_float<T>(key[index]);
				}

				dot = warp_reduce_sum(dot);

				if (lane == 0) {
					shared_scores[local_key] = dot * scale;
				}
			}

			__syncthreads();

			float local_max = -FLT_MAX;

			for (
				std::size_t index = threadIdx.x;
				index < tile_length;
				index += blockDim.x
			) {
				local_max = fmaxf(
					local_max,
					shared_scores[index]
				);
			}

			const float tile_max =
				block_reduce_max(local_max, shared_reduction);

			float local_sum = 0.0F;

			for (
				std::size_t index = threadIdx.x;
				index < tile_length;
				index += blockDim.x
			) {
				const float probability =
					expf(shared_scores[index] - tile_max);

				shared_scores[index] = probability;
				local_sum += probability;
			}

			const float tile_sum =
				block_reduce_sum(local_sum, shared_reduction);

			if (threadIdx.x == 0) {
				const float running_max = shared_state[0];
				const float new_max = fmaxf(running_max, tile_max);

				shared_state[2] = running_max == -FLT_MAX
					? 0.0F
					: expf(running_max - new_max);

				shared_state[3] = expf(tile_max - new_max);
				shared_state[0] = new_max;
			}

			__syncthreads();

			const float old_scale = shared_state[2];
			const float tile_scale = shared_state[3];

			for (
				std::size_t value_index = threadIdx.x;
				value_index < dv;
				value_index += blockDim.x
			) {
				float weighted_value = 0.0F;

				for (
					std::size_t local_key = 0;
					local_key < tile_length;
					++local_key
				) {
					const std::size_t key_index =
						key_start + local_key;

					const T *value =
						v + (key_index * nkvhead + kv_head) * dv;

					weighted_value +=
						shared_scores[local_key]
						* to_float<T>(value[value_index]);
				}

				shared_output[value_index] =
					shared_output[value_index] * old_scale
					+ weighted_value * tile_scale;
			}

			if (threadIdx.x == 0) {
				shared_state[1] =
					shared_state[1] * old_scale
					+ tile_sum * tile_scale;
			}

			__syncthreads();
		}

		T *output =
			attn_val + (query_index * nhead + query_head) * dv;

		for (
			std::size_t index = threadIdx.x;
			index < dv;
			index += blockDim.x
		) {
			output[index] = from_float<T>(
				shared_output[index] / shared_state[1]
			);
		}

		__syncthreads();
	}
}

template <typename T>
void launch_fallback(
	T *attn_val,
	const T *q,
	const T *k,
	const T *v,
	float scale,
	std::size_t seqlen,
	std::size_t nhead,
	std::size_t dv,
	std::size_t total_len,
	std::size_t nkvhead,
	std::size_t d,
	cudaStream_t stream
) {
	const std::size_t task_count =
		checked_product(
			seqlen,
			nhead,
			"SelfAttention: task count overflows size_t."
		);

	if (task_count == 0) {
		return;
	}

	CHECK_ARGUMENT(
		d <= std::numeric_limits<std::size_t>::max() - TILE_KV,
		"SelfAttention: shared-memory element count overflows size_t."
	);

	const std::size_t partial_shared_elements = d + TILE_KV;

	CHECK_ARGUMENT(
		dv <= std::numeric_limits<std::size_t>::max()
			- partial_shared_elements
			- WARP_COUNT
			- 4,
		"SelfAttention: shared-memory element count overflows size_t."
	);

	const std::size_t shared_elements =
		partial_shared_elements + dv + WARP_COUNT + 4;

	CHECK_ARGUMENT(
		shared_elements
			<= std::numeric_limits<std::size_t>::max() / sizeof(float),
		"SelfAttention: shared-memory byte count overflows size_t."
	);

	const std::size_t shared_bytes =
		shared_elements * sizeof(float);

	int device = -1;
	CUDA_CHECK(cudaGetDevice(&device));

	cudaDeviceProp properties{};
	CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

	const std::size_t maximum_shared_bytes =
		properties.sharedMemPerBlockOptin > 0
			? static_cast<std::size_t>(properties.sharedMemPerBlockOptin)
			: static_cast<std::size_t>(properties.sharedMemPerBlock);

	CHECK_ARGUMENT(
		shared_bytes <= maximum_shared_bytes,
		"SelfAttention: d and dv require more shared memory than this GPU supports."
	);

	if (
		shared_bytes
		> static_cast<std::size_t>(properties.sharedMemPerBlock)
	) {
		CUDA_CHECK(cudaFuncSetAttribute(
			self_attention_fallback_kernel<T>,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			static_cast<int>(shared_bytes)
		));
	}

	const std::size_t grid_size =
		get_capped_grid_size(
			task_count,
			1,
			CUDA_DEFAULT_MAX_GRID_SIZE
		);

	self_attention_fallback_kernel<T>
		<<<
			static_cast<unsigned int>(grid_size),
			static_cast<unsigned int>(CUDA_BLOCK_SIZE),
			shared_bytes,
			stream
		>>>(
			attn_val,
			q,
			k,
			v,
			scale,
			seqlen,
			nhead,
			dv,
			total_len,
			nkvhead,
			d
		);

	CUDA_CHECK(cudaGetLastError());
}

#if LLAISYS_HAS_CUDNN_SDPA

namespace fe = cudnn_frontend;

inline constexpr std::int64_t Q_UID = 1;
inline constexpr std::int64_t K_UID = 2;
inline constexpr std::int64_t V_UID = 3;
inline constexpr std::int64_t O_UID = 4;

// Safety guardrails for the thread-local cuDNN SDPA caches. Autoregressive
// decoding can generate a distinct graph key for every total_len, so an
// unbounded cache can retain graph objects, cuDNN handles, and workspaces
// indefinitely. These capacities are intentionally easy to tune after
// workload-specific benchmarking.
inline constexpr std::size_t CUDNN_GRAPH_CACHE_CAPACITY_PER_DEVICE = 8;
inline constexpr std::size_t CUDNN_REJECTED_CACHE_CAPACITY_PER_DEVICE = 64;

static_assert(
	CUDNN_GRAPH_CACHE_CAPACITY_PER_DEVICE > 0,
	"SelfAttention: cuDNN graph cache capacity must be positive."
);

static_assert(
	CUDNN_REJECTED_CACHE_CAPACITY_PER_DEVICE > 0,
	"SelfAttention: cuDNN rejected cache capacity must be positive."
);

void check_cudnn(
	cudnnStatus_t status,
	const char *operation
) {
	if (status == CUDNN_STATUS_SUCCESS) {
		return;
	}

	throw std::runtime_error(
		std::string("SelfAttention: ")
		+ operation
		+ " failed: "
		+ cudnnGetErrorString(status)
	);
}

std::uint32_t float_bits(float value) {
	std::uint32_t bits = 0;
	static_assert(sizeof(bits) == sizeof(value));
	std::memcpy(&bits, &value, sizeof(bits));
	return bits;
}

struct CudnnGraphKey {
	int device;
	llaisysDataType_t type;
	std::size_t seqlen;
	std::size_t nhead;
	std::size_t dv;
	std::size_t total_len;
	std::size_t nkvhead;
	std::size_t d;
	std::uint32_t scale_bits;

	bool operator==(const CudnnGraphKey &other) const noexcept {
		return device == other.device
			&& type == other.type
			&& seqlen == other.seqlen
			&& nhead == other.nhead
			&& dv == other.dv
			&& total_len == other.total_len
			&& nkvhead == other.nkvhead
			&& d == other.d
			&& scale_bits == other.scale_bits;
	}
};

struct CudnnGraphKeyHash {
	std::size_t operator()(const CudnnGraphKey &key) const noexcept {
		std::size_t hash = 1469598103934665603ULL;

		auto combine = [&hash](std::size_t value) {
			hash ^= value;
			hash *= 1099511628211ULL;
		};

		combine(static_cast<std::size_t>(key.device));
		combine(static_cast<std::size_t>(key.type));
		combine(key.seqlen);
		combine(key.nhead);
		combine(key.dv);
		combine(key.total_len);
		combine(key.nkvhead);
		combine(key.d);
		combine(key.scale_bits);
		return hash;
	}
};

class CudnnGraphEntry final {
public:
	explicit CudnnGraphEntry(const CudnnGraphKey &key)
		: _device_id(key.device) {
		CUDA_CHECK(cudaSetDevice(_device_id));

		try {
			check_cudnn(cudnnCreate(&_handle), "cudnnCreate");

			const fe::DataType_t io_type =
				key.type == LLAISYS_DTYPE_F16
					? fe::DataType_t::HALF
					: fe::DataType_t::BFLOAT16;

			_graph = std::make_shared<fe::graph::Graph>();
			_graph
				->set_io_data_type(io_type)
				.set_intermediate_data_type(fe::DataType_t::FLOAT)
				.set_compute_data_type(fe::DataType_t::FLOAT);

			const std::int64_t sequence_length =
				static_cast<std::int64_t>(key.seqlen);
			const std::int64_t total_length =
				static_cast<std::int64_t>(key.total_len);
			const std::int64_t query_heads =
				static_cast<std::int64_t>(key.nhead);
			const std::int64_t kv_heads =
				static_cast<std::int64_t>(key.nkvhead);
			const std::int64_t query_dimension =
				static_cast<std::int64_t>(key.d);
			const std::int64_t value_dimension =
				static_cast<std::int64_t>(key.dv);

			// cuDNN descriptors use logical BHSD dimensions. Custom strides map
			// them directly onto LLAISYS's contiguous [S, H, D] memory, so no
			// transpose or temporary layout conversion is needed.
			auto Q = _graph->tensor(
				fe::graph::Tensor_attributes()
					.set_name("Q")
					.set_uid(Q_UID)
					.set_dim({1, query_heads, sequence_length, query_dimension})
					.set_stride({
						sequence_length * query_heads * query_dimension,
						query_dimension,
						query_heads * query_dimension,
						1,
					})
			);

			auto K = _graph->tensor(
				fe::graph::Tensor_attributes()
					.set_name("K")
					.set_uid(K_UID)
					.set_dim({1, kv_heads, total_length, query_dimension})
					.set_stride({
						total_length * kv_heads * query_dimension,
						query_dimension,
						kv_heads * query_dimension,
						1,
					})
			);

			auto V = _graph->tensor(
				fe::graph::Tensor_attributes()
					.set_name("V")
					.set_uid(V_UID)
					.set_dim({1, kv_heads, total_length, value_dimension})
					.set_stride({
						total_length * kv_heads * value_dimension,
						value_dimension,
						kv_heads * value_dimension,
						1,
					})
			);

			float scale = 0.0F;
			const std::uint32_t bits = key.scale_bits;
			std::memcpy(&scale, &bits, sizeof(scale));

			auto options = fe::graph::SDPA_attributes()
				.set_name("llaisys_self_attention")
				.set_generate_stats(false)
				.set_attn_scale(scale)
				.set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT)
				.set_diagonal_band_right_bound(0);

			auto [O, Stats] = _graph->sdpa(Q, K, V, options);
			(void)Stats;

			O->set_output(true)
				.set_uid(O_UID)
				.set_dim({1, query_heads, sequence_length, value_dimension})
				.set_stride({
					sequence_length * query_heads * value_dimension,
					value_dimension,
					query_heads * value_dimension,
					1,
				});

			const auto build_error =
				_graph->build(_handle, {fe::HeurMode_t::A});

			if (!build_error.is_good()) {
				throw std::runtime_error(
					"SelfAttention: cuDNN SDPA graph build failed."
				);
			}

			std::int64_t workspace_size = 0;
			const auto workspace_error =
				_graph->get_workspace_size(workspace_size);

			if (!workspace_error.is_good() || workspace_size < 0) {
				throw std::runtime_error(
					"SelfAttention: cuDNN SDPA workspace query failed."
				);
			}

			_workspace_size = static_cast<std::size_t>(workspace_size);

			if (_workspace_size > 0) {
				CUDA_CHECK(cudaMalloc(&_workspace, _workspace_size));
			}
		} catch (...) {
			// A throwing constructor does not run ~CudnnGraphEntry(). Release
			// partially created CUDA/cuDNN resources before propagating the
			// failure to the bounded rejected-graph cache.
			release_resources_noexcept();
			throw;
		}
	}

	~CudnnGraphEntry() noexcept {
		int previous_device = -1;
		const cudaError_t get_device_status =
			cudaGetDevice(&previous_device);

		// cuDNN handles and CUDA allocations belong to the device that
		// was current when the graph entry was created.
		(void)cudaSetDevice(_device_id);
		release_resources_noexcept();

		if (
			get_device_status == cudaSuccess
			&& previous_device >= 0
			&& previous_device != _device_id
		) {
			(void)cudaSetDevice(previous_device);
		}
	}

	CudnnGraphEntry(const CudnnGraphEntry &) = delete;
	CudnnGraphEntry &operator=(const CudnnGraphEntry &) = delete;

	void execute(
		void *output,
		const void *q,
		const void *k,
		const void *v,
		cudaStream_t stream
	) {
		check_cudnn(
			cudnnSetStream(_handle, stream),
			"cudnnSetStream"
		);

		std::unordered_map<
			fe::graph::Tensor_attributes::uid_t,
			void *
		> variant_pack{
			{Q_UID, const_cast<void *>(q)},
			{K_UID, const_cast<void *>(k)},
			{V_UID, const_cast<void *>(v)},
			{O_UID, output},
		};

		const auto execute_error =
			_graph->execute(
				_handle,
				variant_pack,
				_workspace
			);

		if (!execute_error.is_good()) {
			throw std::runtime_error(
				"SelfAttention: cuDNN SDPA execution failed."
			);
		}
	}

private:
	void release_resources_noexcept() noexcept {
		if (_workspace != nullptr) {
			(void)cudaFree(_workspace);
			_workspace = nullptr;
		}

		if (_handle != nullptr) {
			(void)cudnnDestroy(_handle);
			_handle = nullptr;
		}
	}

	int _device_id{-1};
	cudnnHandle_t _handle{nullptr};
	std::shared_ptr<fe::graph::Graph> _graph;
	void *_workspace{nullptr};
	std::size_t _workspace_size{0};
};

class CudnnGraphCache final {
public:
	CudnnGraphEntry *find(const CudnnGraphKey &key) {
		auto iterator = _index.find(key);

		if (iterator == _index.end()) {
			return nullptr;
		}

		// Move the most recently used graph to the front. std::list::splice
		// preserves iterators, so the unordered_map index remains valid.
		_entries.splice(
			_entries.begin(),
			_entries,
			iterator->second
		);

		return iterator->second->entry.get();
	}

	void insert(
		const CudnnGraphKey &key,
		std::unique_ptr<CudnnGraphEntry> entry
	) {
		auto existing = _index.find(key);

		if (existing != _index.end()) {
			existing->second->entry = std::move(entry);
			_entries.splice(
				_entries.begin(),
				_entries,
				existing->second
			);
			return;
		}

		_entries.push_front(Node{key, std::move(entry)});

		try {
			_index.emplace(
				_entries.front().key,
				_entries.begin()
			);
		} catch (...) {
			// Preserve the previous cache contents if index allocation fails.
			_entries.pop_front();
			throw;
		}

		while (_entries.size() > CUDNN_GRAPH_CACHE_CAPACITY_PER_DEVICE) {
			_index.erase(_entries.back().key);
			_entries.pop_back();
		}
	}

private:
	struct Node {
		CudnnGraphKey key;
		std::unique_ptr<CudnnGraphEntry> entry;
	};

	using EntryList = std::list<Node>;

	EntryList _entries;
	std::unordered_map<
		CudnnGraphKey,
		EntryList::iterator,
		CudnnGraphKeyHash
	> _index;
};

class RejectedCudnnGraphCache final {
public:
	bool contains(const CudnnGraphKey &key) {
		auto iterator = _index.find(key);

		if (iterator == _index.end()) {
			return false;
		}

		_entries.splice(
			_entries.begin(),
			_entries,
			iterator->second
		);
		return true;
	}

	void insert(const CudnnGraphKey &key) {
		auto existing = _index.find(key);

		if (existing != _index.end()) {
			_entries.splice(
				_entries.begin(),
				_entries,
				existing->second
			);
			return;
		}

		_entries.push_front(key);

		try {
			_index.emplace(
				_entries.front(),
				_entries.begin()
			);
		} catch (...) {
			_entries.pop_front();
			throw;
		}

		while (
			_entries.size()
				> CUDNN_REJECTED_CACHE_CAPACITY_PER_DEVICE
		) {
			_index.erase(_entries.back());
			_entries.pop_back();
		}
	}

private:
	using EntryList = std::list<CudnnGraphKey>;

	EntryList _entries;
	std::unordered_map<
		CudnnGraphKey,
		EntryList::iterator,
		CudnnGraphKeyHash
	> _index;
};

struct CudnnDeviceCache {
	CudnnGraphCache graphs;
	RejectedCudnnGraphCache rejected;
};

// The outer map is naturally bounded by the number of CUDA devices used by
// the current host thread. Each device cache independently enforces its LRU
// capacities so activity on one GPU cannot evict graphs belonging to another.
thread_local std::unordered_map<int, CudnnDeviceCache> cudnn_device_caches;

bool cudnn_supports(
	llaisysDataType_t type,
	std::size_t seqlen,
	std::size_t total_len,
	std::size_t d,
	std::size_t dv
) {
	if (
		type != LLAISYS_DTYPE_F16
		&& type != LLAISYS_DTYPE_BF16
	) {
		return false;
	}

	if (
		seqlen == 0
		|| total_len == 0
		|| d % 8 != 0
		|| dv % 8 != 0
	) {
		return false;
	}

#if defined(CUDART_VERSION)
	if (CUDART_VERSION < 12000) {
		return false;
	}
#endif

	int device = -1;
	if (cudaGetDevice(&device) != cudaSuccess) {
		return false;
	}

	cudaDeviceProp properties{};
	if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
		return false;
	}

	if (properties.major < 8) {
		return false;
	}

	const std::size_t maximum_dimension =
		properties.major >= 9 ? 256 : 128;

	return d <= maximum_dimension
		&& dv <= maximum_dimension;
}

bool try_cudnn(
	void *attn_val,
	const void *q,
	const void *k,
	const void *v,
	float scale,
	llaisysDataType_t type,
	std::size_t seqlen,
	std::size_t nhead,
	std::size_t dv,
	std::size_t total_len,
	std::size_t nkvhead,
	std::size_t d,
	cudaStream_t stream
) {
	if (!cudnn_supports(type, seqlen, total_len, d, dv)) {
		return false;
	}

	int device = -1;
	CUDA_CHECK(cudaGetDevice(&device));

	const CudnnGraphKey key{
		device,
		type,
		seqlen,
		nhead,
		dv,
		total_len,
		nkvhead,
		d,
		float_bits(scale),
	};

	auto &device_cache = cudnn_device_caches[device];

	if (device_cache.rejected.contains(key)) {
		return false;
	}

	CudnnGraphEntry *entry = device_cache.graphs.find(key);

	if (entry == nullptr) {
		std::unique_ptr<CudnnGraphEntry> new_entry;

		try {
			new_entry = std::make_unique<CudnnGraphEntry>(key);
		} catch (...) {
			// A shape unsupported by the installed cuDNN version should
			// use the fused CUDA fallback on later calls as well. The
			// rejected cache is itself bounded, so one-off shapes cannot
			// grow thread-local state indefinitely.
			device_cache.rejected.insert(key);
			return false;
		}

		entry = new_entry.get();
		device_cache.graphs.insert(
			key,
			std::move(new_entry)
		);
	}

	// Do not silently fall back after an execution failure. cuDNN may
	// already have enqueued work on the stream, and hiding the error can
	// produce an unsafe second write to the output.
	entry->execute(
		attn_val,
		q,
		k,
		v,
		stream
	);

	return true;
}

#endif // LLAISYS_HAS_CUDNN_SDPA

} // namespace

namespace llaisys::ops::nvidia {

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
	std::size_t d,
	llaisysStream_t stream
) {
	CHECK_ARGUMENT(
		nhead > 0,
		"SelfAttention: query head count must be greater than zero."
	);

	CHECK_ARGUMENT(
		nkvhead > 0,
		"SelfAttention: KV head count must be greater than zero."
	);

	CHECK_ARGUMENT(
		nhead % nkvhead == 0,
		"SelfAttention: query head count must be a multiple of KV head count."
	);

	CHECK_ARGUMENT(
		total_len >= seqlen,
		"SelfAttention: total KV length must not be smaller than query length."
	);

	CHECK_ARGUMENT(
		d > 0,
		"SelfAttention: query/key head dimension must be greater than zero."
	);

	CHECK_ARGUMENT(
		dv > 0,
		"SelfAttention: value head dimension must be greater than zero."
	);

	CHECK_ARGUMENT(
		std::isfinite(scale),
		"SelfAttention: scale must be finite."
	);

	const std::size_t output_elements =
		checked_product(
			checked_product(
				seqlen,
				nhead,
				"SelfAttention: output vector count overflows size_t."
			),
			dv,
			"SelfAttention: output element count overflows size_t."
		);

	const std::size_t query_elements =
		checked_product(
			checked_product(
				seqlen,
				nhead,
				"SelfAttention: query vector count overflows size_t."
			),
			d,
			"SelfAttention: query element count overflows size_t."
		);

	const std::size_t key_elements =
		checked_product(
			checked_product(
				total_len,
				nkvhead,
				"SelfAttention: key vector count overflows size_t."
			),
			d,
			"SelfAttention: key element count overflows size_t."
		);

	const std::size_t value_elements =
		checked_product(
			checked_product(
				total_len,
				nkvhead,
				"SelfAttention: value vector count overflows size_t."
			),
			dv,
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

	if (output_elements == 0) {
		return;
	}

	const cudaStream_t cuda_stream =
		reinterpret_cast<cudaStream_t>(stream);

#if LLAISYS_HAS_CUDNN_SDPA
	if (try_cudnn(
		attn_val,
		q,
		k,
		v,
		scale,
		type,
		seqlen,
		nhead,
		dv,
		total_len,
		nkvhead,
		d,
		cuda_stream
	)) {
		return;
	}
#endif

	return llaisys::device::nvidia::dispatch_cuda_dtype(
		type,
		[&](auto tag) {
			using T = typename decltype(tag)::type;

			return launch_fallback<T>(
				reinterpret_cast<T *>(attn_val),
				reinterpret_cast<const T *>(q),
				reinterpret_cast<const T *>(k),
				reinterpret_cast<const T *>(v),
				scale,
				seqlen,
				nhead,
				dv,
				total_len,
				nkvhead,
				d,
				cuda_stream
			);
		}
	);
}

} // namespace llaisys::ops::nvidia

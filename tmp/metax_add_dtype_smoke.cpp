#include "../src/ops/add/cuda_compat/add_cuda_compat.cuh"

#include <mc_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

namespace cuda_compat =
	llaisys::ops::cuda_compat;

void check_mc(
	mcError_t status,
	const char *expression
) {
	if (status == mcSuccess) {
		return;
	}

	std::fprintf(
		stderr,
		"MACA call failed: %s, error=%s\n",
		expression,
		mcGetErrorString(status)
	);

	std::exit(EXIT_FAILURE);
}

#define MC_CHECK(expression) \
	check_mc((expression), #expression)

template <typename T>
__global__ void initialize_inputs(
	T *a,
	T *b,
	std::size_t numel
) {
	const std::size_t thread_index =
		static_cast<std::size_t>(blockIdx.x)
			* static_cast<std::size_t>(blockDim.x)
		+ static_cast<std::size_t>(threadIdx.x);

	const std::size_t thread_stride =
		static_cast<std::size_t>(blockDim.x)
			* static_cast<std::size_t>(gridDim.x);

	for (
		std::size_t i = thread_index;
		i < numel;
		i += thread_stride
	) {
		// Keep values small and exactly representable in
		// both FP16 and BF16.
		const float a_value =
			static_cast<float>(i % 8)
			* 0.25F;

		const float b_value =
			static_cast<float>(i % 8)
			* 0.50F;

		a[i] =
			cuda_compat::from_float<T>(
				a_value
			);

		b[i] =
			cuda_compat::from_float<T>(
				b_value
			);
	}
}

template <typename T>
__global__ void convert_to_float(
	float *output,
	const T *input,
	std::size_t numel
) {
	const std::size_t thread_index =
		static_cast<std::size_t>(blockIdx.x)
			* static_cast<std::size_t>(blockDim.x)
		+ static_cast<std::size_t>(threadIdx.x);

	const std::size_t thread_stride =
		static_cast<std::size_t>(blockDim.x)
			* static_cast<std::size_t>(gridDim.x);

	for (
		std::size_t i = thread_index;
		i < numel;
		i += thread_stride
	) {
		output[i] =
			cuda_compat::to_float<T>(
				input[i]
			);
	}
}

template <typename T>
bool run_add_case(
	const char *dtype_name,
	bool request_vectorized
) {
	// 1027 is intentional.
	//
	// FP16/BF16 vector size = 8 elements:
	//
	//   1024 elements -> vectorized body
	//      3 elements -> scalar tail
	constexpr std::size_t numel = 1027;

	constexpr std::size_t block_size = 256;

	constexpr std::size_t initialization_grid_size = 4;

	const std::size_t data_bytes =
		numel * sizeof(T);

	const std::size_t output_bytes =
		numel * sizeof(float);

	T *device_a = nullptr;
	T *device_b = nullptr;
	T *device_c = nullptr;

	float *device_output = nullptr;

	MC_CHECK(
		mcMalloc(
			reinterpret_cast<void **>(&device_a),
			data_bytes
		)
	);

	MC_CHECK(
		mcMalloc(
			reinterpret_cast<void **>(&device_b),
			data_bytes
		)
	);

	MC_CHECK(
		mcMalloc(
			reinterpret_cast<void **>(&device_c),
			data_bytes
		)
	);

	MC_CHECK(
		mcMalloc(
			reinterpret_cast<void **>(&device_output),
			output_bytes
		)
	);

	mcStream_t stream = nullptr;

	MC_CHECK(
		mcStreamCreate(
			&stream
		)
	);

	// ============================================================
	// Initialize FP16/BF16 inputs on device
	// ============================================================

	initialize_inputs<T>
		<<<
			initialization_grid_size,
			block_size,
			0,
			stream
		>>>(
			device_a,
			device_b,
			numel
		);

	MC_CHECK(
		mcGetLastError()
	);

	// ============================================================
	// Determine vectorized-path eligibility
	// ============================================================

	const bool vectorization_available =
		cuda_compat::can_use_vectorized_add<T>(
			device_c,
			device_a,
			device_b,
			numel
		);

	if (
		request_vectorized
		&& !vectorization_available
	) {
		std::fprintf(
			stderr,
			"%s: vectorized path was requested "
			"but eligibility check returned false.\n",
			dtype_name
		);

		return false;
	}

	const bool use_vectorized_kernel =
		request_vectorized;

	const std::size_t work_items =
		cuda_compat::get_add_work_items<T>(
			numel,
			use_vectorized_kernel
		);

	const std::size_t grid_size =
		(work_items + block_size - 1)
		/ block_size;

	std::printf(
		"[%s %s]\n",
		dtype_name,
		use_vectorized_kernel
			? "vectorized"
			: "scalar"
	);

	std::printf(
		"    vectorization_available = %s\n",
		vectorization_available
			? "true"
			: "false"
	);

	std::printf(
		"    numel                   = %zu\n",
		numel
	);

	std::printf(
		"    work_items              = %zu\n",
		work_items
	);

	std::printf(
		"    block_size              = %zu\n",
		block_size
	);

	std::printf(
		"    grid_size               = %zu\n",
		grid_size
	);

	// ============================================================
	// Shared CUDA-compatible Add kernel
	// ============================================================

	cuda_compat::launch_add_kernel<T>(
		device_c,
		device_a,
		device_b,
		numel,
		block_size,
		grid_size,
		use_vectorized_kernel,
		stream
	);

	MC_CHECK(
		mcGetLastError()
	);

	// ============================================================
	// Convert result to FP32 for host verification
	// ============================================================

	convert_to_float<T>
		<<<
			initialization_grid_size,
			block_size,
			0,
			stream
		>>>(
			device_output,
			device_c,
			numel
		);

	MC_CHECK(
		mcGetLastError()
	);

	MC_CHECK(
		mcStreamSynchronize(
			stream
		)
	);

	float *host_output =
		new float[numel];

	MC_CHECK(
		mcMemcpy(
			host_output,
			device_output,
			output_bytes,
			mcMemcpyDeviceToHost
		)
	);

	// ============================================================
	// Verify
	// ============================================================

	bool passed = true;

	for (
		std::size_t i = 0;
		i < numel;
		++i
	) {
		const float a_value =
			static_cast<float>(i % 8)
			* 0.25F;

		const float b_value =
			static_cast<float>(i % 8)
			* 0.50F;

		const float expected =
			a_value + b_value;

		const float actual =
			host_output[i];

		if (
			std::fabs(
				actual - expected
			) > 1e-6F
		) {
			std::fprintf(
				stderr,
				"    Mismatch at %zu: "
				"got=%f expected=%f\n",
				i,
				actual,
				expected
			);

			passed = false;
			break;
		}
	}

	delete[] host_output;

	MC_CHECK(
		mcStreamDestroy(
			stream
		)
	);

	MC_CHECK(
		mcFree(
			device_output
		)
	);

	MC_CHECK(
		mcFree(
			device_c
		)
	);

	MC_CHECK(
		mcFree(
			device_b
		)
	);

	MC_CHECK(
		mcFree(
			device_a
		)
	);

	if (passed) {
		std::printf(
			"    PASSED\n"
		);
	}

	return passed;
}

} // namespace

int main() {
	MC_CHECK(
		mcSetDevice(0)
	);

	bool passed = true;

	passed =
		run_add_case<
			cuda_compat::fp16_t
		>(
			"FP16",
			false
		)
		&& passed;

	passed =
		run_add_case<
			cuda_compat::fp16_t
		>(
			"FP16",
			true
		)
		&& passed;

	passed =
		run_add_case<
			cuda_compat::bf16_t
		>(
			"BF16",
			false
		)
		&& passed;

	passed =
		run_add_case<
			cuda_compat::bf16_t
		>(
			"BF16",
			true
		)
		&& passed;

	if (!passed) {
		std::fprintf(
			stderr,
			"MetaX Add dtype smoke test FAILED\n"
		);

		return EXIT_FAILURE;
	}

	std::printf(
		"All shared CUDA-compatible "
		"FP16/BF16 Add tests PASSED on MetaX\n"
	);

	return EXIT_SUCCESS;
}

#include "/data/llaisys-26s/src/ops/add/cuda_compat/add_cuda_compat.cuh"

#include <mc_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

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

} // namespace

int main() {
	namespace cuda_compat =
		llaisys::ops::cuda_compat;

	// 1027 is intentional:
	//
	//   1024 elements -> float4 vectorized body
	//      3 elements -> scalar tail
	constexpr std::size_t numel = 1027;

	constexpr std::size_t bytes =
		numel * sizeof(float);

	std::vector<float> host_a(numel);
	std::vector<float> host_b(numel);
	std::vector<float> host_c(numel, 0.0F);

	for (std::size_t i = 0; i < numel; ++i) {
		host_a[i] =
			static_cast<float>(i) * 0.25F;

		host_b[i] =
			static_cast<float>(i) * 0.50F + 1.0F;
	}

	// ============================================================
	// MetaX Runtime setup
	// ============================================================

	MC_CHECK(
		mcSetDevice(0)
	);

	mcStream_t stream = nullptr;

	MC_CHECK(
		mcStreamCreate(
			&stream
		)
	);

	float *device_a = nullptr;
	float *device_b = nullptr;
	float *device_c = nullptr;

	MC_CHECK(
		mcMalloc(
			reinterpret_cast<void **>(
				&device_a
			),
			bytes
		)
	);

	MC_CHECK(
		mcMalloc(
			reinterpret_cast<void **>(
				&device_b
			),
			bytes
		)
	);

	MC_CHECK(
		mcMalloc(
			reinterpret_cast<void **>(
				&device_c
			),
			bytes
		)
	);

	// ============================================================
	// Host -> Device
	// ============================================================

	MC_CHECK(
		mcMemcpy(
			device_a,
			host_a.data(),
			bytes,
			mcMemcpyHostToDevice
		)
	);

	MC_CHECK(
		mcMemcpy(
			device_b,
			host_b.data(),
			bytes,
			mcMemcpyHostToDevice
		)
	);

	// ============================================================
	// Shared CUDA-compatible vectorization decision
	// ============================================================

	const bool use_vectorized_kernel =
		cuda_compat::can_use_vectorized_add<float>(
			device_c,
			device_a,
			device_b,
			numel
		);

	std::printf(
		"use_vectorized_kernel = %s\n",
		use_vectorized_kernel
			? "true"
			: "false"
	);

	if (!use_vectorized_kernel) {
		std::fprintf(
			stderr,
			"Expected FP32 vectorized Add path, "
			"but vectorization was rejected.\n"
		);

		MC_CHECK(mcFree(device_c));
		MC_CHECK(mcFree(device_b));
		MC_CHECK(mcFree(device_a));
		MC_CHECK(mcStreamDestroy(stream));

		return EXIT_FAILURE;
	}

	// ============================================================
	// Shared logical work calculation
	// ============================================================

	const std::size_t work_items =
		cuda_compat::get_add_work_items<float>(
			numel,
			use_vectorized_kernel
		);

	constexpr std::size_t block_size = 256;

	const std::size_t grid_size =
		(work_items + block_size - 1)
		/ block_size;

	std::printf(
		"numel      = %zu\n",
		numel
	);

	std::printf(
		"work_items = %zu\n",
		work_items
	);

	std::printf(
		"block_size = %zu\n",
		block_size
	);

	std::printf(
		"grid_size  = %zu\n",
		grid_size
	);

	// ============================================================
	// Shared CUDA-compatible Add kernel
	// ============================================================

	cuda_compat::launch_add_kernel<float>(
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

	MC_CHECK(
		mcStreamSynchronize(
			stream
		)
	);

	// ============================================================
	// Device -> Host
	// ============================================================

	MC_CHECK(
		mcMemcpy(
			host_c.data(),
			device_c,
			bytes,
			mcMemcpyDeviceToHost
		)
	);

	// ============================================================
	// Correctness verification
	// ============================================================

	for (std::size_t i = 0; i < numel; ++i) {
		const float expected =
			host_a[i] + host_b[i];

		const float actual =
			host_c[i];

		if (
			std::fabs(actual - expected)
			> 1e-6F
		) {
			std::fprintf(
				stderr,
				"Mismatch at index %zu: "
				"got=%f expected=%f\n",
				i,
				actual,
				expected
			);

			MC_CHECK(mcFree(device_c));
			MC_CHECK(mcFree(device_b));
			MC_CHECK(mcFree(device_a));
			MC_CHECK(mcStreamDestroy(stream));

			return EXIT_FAILURE;
		}
	}

	// ============================================================
	// Cleanup
	// ============================================================

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

	MC_CHECK(
		mcStreamDestroy(
			stream
		)
	);

	std::printf(
		"Shared CUDA-compatible FP32 vectorized Add "
		"+ scalar tail PASSED on MetaX\n"
	);

	return EXIT_SUCCESS;
}
#include "../src/core/context/context.hpp"

#include <iostream>

int main() {
	using llaisys::core::Context;
	using llaisys::core::context;

	std::cout
		<< "===== Context Device Switch Test ====="
		<< std::endl;

	Context &ctx = context();

	// --------------------------------------------------------
	// CPU
	// --------------------------------------------------------

	std::cout
		<< "[1] Activate CPU"
		<< std::endl;

	ctx.setDevice(
		LLAISYS_DEVICE_CPU,
		0
	);

	auto *cpu_runtime_first =
		&ctx.runtime();

	if (
		cpu_runtime_first->deviceType()
			!= LLAISYS_DEVICE_CPU
		|| cpu_runtime_first->deviceId() != 0
	) {
		std::cerr
			<< "CPU runtime activation failed."
			<< std::endl;

		return 1;
	}

	std::cout
		<< "CPU Runtime address: "
		<< cpu_runtime_first
		<< std::endl;


#ifdef ENABLE_NVIDIA_API

	// --------------------------------------------------------
	// NVIDIA
	// --------------------------------------------------------

	std::cout
		<< "\n[2] Activate NVIDIA"
		<< std::endl;

	ctx.setDevice(
		LLAISYS_DEVICE_NVIDIA,
		0
	);

	auto *nvidia_runtime_first =
		&ctx.runtime();

	if (
		nvidia_runtime_first->deviceType()
			!= LLAISYS_DEVICE_NVIDIA
		|| nvidia_runtime_first->deviceId() != 0
	) {
		std::cerr
			<< "NVIDIA runtime activation failed."
			<< std::endl;

		return 1;
	}

	std::cout
		<< "NVIDIA Runtime address: "
		<< nvidia_runtime_first
		<< std::endl;


	// --------------------------------------------------------
	// CPU again
	// --------------------------------------------------------

	std::cout
		<< "\n[3] Switch back to CPU"
		<< std::endl;

	ctx.setDevice(
		LLAISYS_DEVICE_CPU,
		0
	);

	auto *cpu_runtime_second =
		&ctx.runtime();

	std::cout
		<< "CPU Runtime address: "
		<< cpu_runtime_second
		<< std::endl;

	if (
		cpu_runtime_first
			!= cpu_runtime_second
	) {
		std::cerr
			<< "ERROR: CPU Runtime was recreated."
			<< std::endl;

		return 1;
	}

	std::cout
		<< "CPU Runtime reuse verified."
		<< std::endl;


	// --------------------------------------------------------
	// NVIDIA again
	// --------------------------------------------------------

	std::cout
		<< "\n[4] Switch back to NVIDIA"
		<< std::endl;

	ctx.setDevice(
		LLAISYS_DEVICE_NVIDIA,
		0
	);

	auto *nvidia_runtime_second =
		&ctx.runtime();

	std::cout
		<< "NVIDIA Runtime address: "
		<< nvidia_runtime_second
		<< std::endl;

	if (
		nvidia_runtime_first
			!= nvidia_runtime_second
	) {
		std::cerr
			<< "ERROR: NVIDIA Runtime was recreated."
			<< std::endl;

		return 1;
	}

	std::cout
		<< "NVIDIA Runtime reuse verified."
		<< std::endl;


	// --------------------------------------------------------
	// Repeated switching
	// --------------------------------------------------------

	std::cout
		<< "\n[5] Repeated switching"
		<< std::endl;

	for (int iteration = 0; iteration < 10; ++iteration) {
		ctx.setDevice(
			LLAISYS_DEVICE_CPU,
			0
		);

		if (
			&ctx.runtime()
				!= cpu_runtime_first
		) {
			std::cerr
				<< "CPU Runtime changed on iteration "
				<< iteration
				<< std::endl;

			return 1;
		}

		ctx.setDevice(
			LLAISYS_DEVICE_NVIDIA,
			0
		);

		if (
			&ctx.runtime()
				!= nvidia_runtime_first
		) {
			std::cerr
				<< "NVIDIA Runtime changed on iteration "
				<< iteration
				<< std::endl;

			return 1;
		}

		std::cout
			<< "Iteration "
			<< iteration + 1
			<< "/10 passed."
			<< std::endl;
	}

#else

	std::cout
		<< "\nNVIDIA backend not compiled; "
		<< "CPU-only test completed."
		<< std::endl;

#endif


	std::cout
		<< "\nContext Runtime ownership test passed."
		<< std::endl;

	return 0;
}
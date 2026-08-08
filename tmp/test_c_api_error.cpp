#include "llaisys.h"
#include "llaisys/runtime.h"

#include <cassert>
#include <cstring>
#include <iostream>

int main() {
	std::cout << "===== C API Error Boundary Test =====\n";

	std::cout << "[1] Trigger invalid device ID\n";

	// Deliberately use an invalid device ID.
	const int result =
		llaisysSetContextRuntime(
			LLAISYS_DEVICE_CPU,
			999
		);

	std::cout << "Return value: " << result << '\n';

	const char *error =
		llaisysGetLastError();

	if (error != nullptr) {
		std::cout
			<< "Last error: "
			<< error
			<< '\n';
	}

	// The C API should report failure instead of
	// allowing a C++ exception to cross the C ABI.
	assert(result != 0);

	assert(error != nullptr);
	assert(std::strlen(error) > 0);

	std::cout
		<< "Exception successfully contained "
		<< "inside C ABI boundary.\n";

	std::cout
		<< "C API error boundary test passed.\n";

	return 0;
}
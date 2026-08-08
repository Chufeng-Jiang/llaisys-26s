#pragma once

#include <exception>
#include <string>
#include <utility>

namespace llaisys::c_api {

void clear_last_error() noexcept;

void set_last_error(
	const char *message
) noexcept;

void set_last_error(
	const std::string &message
) noexcept;

const char *get_last_error() noexcept;


template <typename Function>
bool guard(Function &&function) noexcept {
	try {
		clear_last_error();

		std::forward<Function>(
			function
		)();

		return true;
	} catch (const std::exception &exception) {
		set_last_error(
			exception.what()
		);

		return false;
	} catch (...) {
		set_last_error(
			"Unknown C++ exception"
		);

		return false;
	}
}


template <
	typename Result,
	typename Function
>
Result guard_result(
	Function &&function,
	Result failure_value
) noexcept {
	try {
		clear_last_error();

		return std::forward<Function>(
			function
		)();
	} catch (const std::exception &exception) {
		set_last_error(
			exception.what()
		);

		return failure_value;
	} catch (...) {
		set_last_error(
			"Unknown C++ exception"
		);

		return failure_value;
	}
}

} // namespace llaisys::c_api
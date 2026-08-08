import ctypes


def load_error(lib):
	lib.llaisysGetLastError.argtypes = []
	lib.llaisysGetLastError.restype = ctypes.c_char_p


def get_last_error(lib) -> str:
	message = lib.llaisysGetLastError()

	if not message:
		return "Unknown LLAISYS error"

	return message.decode(
		"utf-8",
		errors="replace",
	)


def make_status_checker(lib):
	def check_status(
		result,
		func,
		arguments,
	):
		if result == 0:
			return result

		raise RuntimeError(
			get_last_error(lib)
		)

	return check_status


def make_handle_checker(lib):
	def check_handle(
		result,
		func,
		arguments,
	):
		if result:
			return result

		raise RuntimeError(
			get_last_error(lib)
		)

	return check_handle
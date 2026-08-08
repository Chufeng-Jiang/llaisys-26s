from ctypes import (
	POINTER,
	c_int,
	c_size_t,
	c_ssize_t,
	c_uint8,
	c_void_p,
)

from .error import (
	make_handle_checker,
	make_status_checker,
)

from .llaisys_types import (
	llaisysDataType_t,
	llaisysDeviceType_t,
)


# Opaque C tensor handle.
llaisysTensor_t = c_void_p


def load_tensor(lib):
	check_status = make_status_checker(lib)
	check_handle = make_handle_checker(lib)

	# ============================================================
	# tensorCreate
	# ============================================================

	lib.tensorCreate.argtypes = [
		POINTER(c_size_t),      # shape
		c_size_t,              # ndim
		llaisysDataType_t,     # dtype
		llaisysDeviceType_t,   # device_type
		c_int,                 # device_id
	]

	lib.tensorCreate.restype = llaisysTensor_t
	lib.tensorCreate.errcheck = check_handle


	# ============================================================
	# tensorDestroy
	# ============================================================

	lib.tensorDestroy.argtypes = [
		llaisysTensor_t,
	]

	lib.tensorDestroy.restype = None


	# ============================================================
	# tensorGetData
	# ============================================================

	lib.tensorGetData.argtypes = [
		llaisysTensor_t,
	]

	lib.tensorGetData.restype = c_void_p


	# ============================================================
	# tensorGetNdim
	# ============================================================

	lib.tensorGetNdim.argtypes = [
		llaisysTensor_t,
	]

	lib.tensorGetNdim.restype = c_size_t


	# ============================================================
	# tensorGetShape
	# ============================================================

	lib.tensorGetShape.argtypes = [
		llaisysTensor_t,
		POINTER(c_size_t),
	]

	lib.tensorGetShape.restype = None


	# ============================================================
	# tensorGetStrides
	# ============================================================

	lib.tensorGetStrides.argtypes = [
		llaisysTensor_t,
		POINTER(c_ssize_t),
	]

	lib.tensorGetStrides.restype = None


	# ============================================================
	# tensorGetDataType
	# ============================================================

	lib.tensorGetDataType.argtypes = [
		llaisysTensor_t,
	]

	lib.tensorGetDataType.restype = llaisysDataType_t


	# ============================================================
	# tensorGetDeviceType
	# ============================================================

	lib.tensorGetDeviceType.argtypes = [
		llaisysTensor_t,
	]

	lib.tensorGetDeviceType.restype = llaisysDeviceType_t


	# ============================================================
	# tensorGetDeviceId
	# ============================================================

	lib.tensorGetDeviceId.argtypes = [
		llaisysTensor_t,
	]

	lib.tensorGetDeviceId.restype = c_int


	# ============================================================
	# tensorDebug
	# ============================================================

	lib.tensorDebug.argtypes = [
		llaisysTensor_t,
	]

	# If tensorDebug has been changed to return int in C++,
	# use c_int + status checker.
	lib.tensorDebug.restype = c_int
	lib.tensorDebug.errcheck = check_status


	# ============================================================
	# tensorIsContiguous
	# ============================================================

	lib.tensorIsContiguous.argtypes = [
		llaisysTensor_t,
	]

	lib.tensorIsContiguous.restype = c_uint8


	# ============================================================
	# tensorLoad
	# ============================================================

	lib.tensorLoad.argtypes = [
		llaisysTensor_t,
		c_void_p,
	]

	lib.tensorLoad.restype = c_int
	lib.tensorLoad.errcheck = check_status


	# ============================================================
	# tensorView
	# ============================================================

	lib.tensorView.argtypes = [
		llaisysTensor_t,
		POINTER(c_size_t),
		c_size_t,
	]

	lib.tensorView.restype = llaisysTensor_t
	lib.tensorView.errcheck = check_handle


	# ============================================================
	# tensorPermute
	# ============================================================

	lib.tensorPermute.argtypes = [
		llaisysTensor_t,
		POINTER(c_size_t),
	]

	lib.tensorPermute.restype = llaisysTensor_t
	lib.tensorPermute.errcheck = check_handle


	# ============================================================
	# tensorSlice
	# ============================================================

	lib.tensorSlice.argtypes = [
		llaisysTensor_t,
		c_size_t,   # dim
		c_size_t,   # start
		c_size_t,   # end
	]

	lib.tensorSlice.restype = llaisysTensor_t
	lib.tensorSlice.errcheck = check_handle
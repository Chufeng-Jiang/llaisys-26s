from ctypes import POINTER, c_int, c_size_t, c_ssize_t, c_uint8, c_void_p

from .error import make_handle_checker, make_status_checker
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t

# Opaque C tensor handle.
llaisysTensor_t = c_void_p


def load_tensor(lib):
    check_status = make_status_checker(lib)
    check_handle = make_handle_checker(lib)

    # ============================================================
    # tensorCreate
    # ============================================================

    lib.tensorCreate.argtypes = [POINTER(c_size_t), c_size_t, llaisysDataType_t, llaisysDeviceType_t, c_int]

    lib.tensorCreate.restype = llaisysTensor_t
    lib.tensorCreate.errcheck = check_handle

    # ============================================================
    # tensorDestroy
    # ============================================================

    lib.tensorDestroy.argtypes = [llaisysTensor_t]

    lib.tensorDestroy.restype = c_int
    lib.tensorDestroy.errcheck = check_status

    # ============================================================
    # tensorGetData
    # ============================================================

    lib.tensorGetData.argtypes = [llaisysTensor_t, POINTER(c_void_p)]

    lib.tensorGetData.restype = c_int
    lib.tensorGetData.errcheck = check_status

    # ============================================================
    # tensorGetNdim
    # ============================================================

    lib.tensorGetNdim.argtypes = [llaisysTensor_t, POINTER(c_size_t)]

    lib.tensorGetNdim.restype = c_int
    lib.tensorGetNdim.errcheck = check_status

    # ============================================================
    # tensorGetShape
    # ============================================================

    lib.tensorGetShape.argtypes = [llaisysTensor_t, POINTER(c_size_t)]

    lib.tensorGetShape.restype = c_int
    lib.tensorGetShape.errcheck = check_status

    # ============================================================
    # tensorGetStrides
    # ============================================================

    lib.tensorGetStrides.argtypes = [llaisysTensor_t, POINTER(c_ssize_t)]

    lib.tensorGetStrides.restype = c_int
    lib.tensorGetStrides.errcheck = check_status

    # ============================================================
    # tensorGetDataType
    # ============================================================

    lib.tensorGetDataType.argtypes = [llaisysTensor_t, POINTER(llaisysDataType_t)]

    lib.tensorGetDataType.restype = c_int
    lib.tensorGetDataType.errcheck = check_status

    # ============================================================
    # tensorGetDeviceType
    # ============================================================

    lib.tensorGetDeviceType.argtypes = [llaisysTensor_t, POINTER(llaisysDeviceType_t)]

    lib.tensorGetDeviceType.restype = c_int
    lib.tensorGetDeviceType.errcheck = check_status

    # ============================================================
    # tensorGetDeviceId
    # ============================================================

    lib.tensorGetDeviceId.argtypes = [llaisysTensor_t, POINTER(c_int)]

    lib.tensorGetDeviceId.restype = c_int
    lib.tensorGetDeviceId.errcheck = check_status

    # ============================================================
    # tensorDebug
    # ============================================================

    lib.tensorDebug.argtypes = [llaisysTensor_t]

    lib.tensorDebug.restype = c_int
    lib.tensorDebug.errcheck = check_status

    # ============================================================
    # tensorIsContiguous
    # ============================================================

    lib.tensorIsContiguous.argtypes = [llaisysTensor_t, POINTER(c_uint8)]

    lib.tensorIsContiguous.restype = c_int
    lib.tensorIsContiguous.errcheck = check_status

    # ============================================================
    # tensorLoad
    # ============================================================

    lib.tensorLoad.argtypes = [llaisysTensor_t, c_void_p]

    lib.tensorLoad.restype = c_int
    lib.tensorLoad.errcheck = check_status

    # ============================================================
    # tensorView
    # ============================================================

    lib.tensorView.argtypes = [llaisysTensor_t, POINTER(c_size_t), c_size_t]

    lib.tensorView.restype = llaisysTensor_t
    lib.tensorView.errcheck = check_handle

    # ============================================================
    # tensorPermute
    # ============================================================

    lib.tensorPermute.argtypes = [llaisysTensor_t, POINTER(c_size_t)]

    lib.tensorPermute.restype = llaisysTensor_t
    lib.tensorPermute.errcheck = check_handle

    # ============================================================
    # tensorSlice
    # ============================================================

    lib.tensorSlice.argtypes = [llaisysTensor_t, c_size_t, c_size_t, c_size_t]

    lib.tensorSlice.restype = llaisysTensor_t
    lib.tensorSlice.errcheck = check_handle

import os
import sys

project_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

test_dir = os.path.join(
    project_dir,
    "test",
)

sys.path.insert(0, project_dir)
sys.path.insert(0, test_dir)

import ctypes

from llaisys.libllaisys import (
    LIB_LLAISYS,
    llaisysDataType_t,
    llaisysDeviceType_t,
    llaisysTensor_t,
)

import llaisys


def expect_runtime_error(
    name,
    function,
):
    print(f"=== Test {name} ===")

    try:
        function()

    except RuntimeError as error:
        print("Caught RuntimeError:")
        print(error)
        print(f"{name} error propagation passed.\n")
        return

    raise RuntimeError(f"Expected {name} to raise RuntimeError.")


def main():
    print("===== Tensor Getter C API Error Boundary Test =====")

    null_tensor = llaisysTensor_t()

    lib = LIB_LLAISYS

    # ============================================================
    # tensorGetNdim
    # ============================================================

    def test_get_ndim():
        value = ctypes.c_size_t()

        lib.tensorGetNdim(
            null_tensor,
            ctypes.byref(value),
        )

    expect_runtime_error(
        "tensorGetNdim",
        test_get_ndim,
    )

    # ============================================================
    # tensorGetShape
    # ============================================================

    def test_get_shape():
        value = (ctypes.c_size_t * 1)()

        lib.tensorGetShape(
            null_tensor,
            value,
        )

    expect_runtime_error(
        "tensorGetShape",
        test_get_shape,
    )

    # ============================================================
    # tensorGetStrides
    # ============================================================

    def test_get_strides():
        value = (ctypes.c_ssize_t * 1)()

        lib.tensorGetStrides(
            null_tensor,
            value,
        )

    expect_runtime_error(
        "tensorGetStrides",
        test_get_strides,
    )

    # ============================================================
    # tensorGetDataType
    # ============================================================

    def test_get_dtype():
        value = llaisys.libllaisys.llaisysDataType_t()

        lib.tensorGetDataType(
            null_tensor,
            ctypes.byref(value),
        )

    expect_runtime_error(
        "tensorGetDataType",
        test_get_dtype,
    )

    # ============================================================
    # tensorGetDeviceType
    # ============================================================

    def test_get_device_type():
        value = llaisys.libllaisys.llaisysDeviceType_t()

        lib.tensorGetDeviceType(
            null_tensor,
            ctypes.byref(value),
        )

    expect_runtime_error(
        "tensorGetDeviceType",
        test_get_device_type,
    )

    # ============================================================
    # tensorGetDeviceId
    # ============================================================

    def test_get_device_id():
        value = ctypes.c_int()

        lib.tensorGetDeviceId(
            null_tensor,
            ctypes.byref(value),
        )

    expect_runtime_error(
        "tensorGetDeviceId",
        test_get_device_id,
    )

    # ============================================================
    # tensorGetData
    # ============================================================

    def test_get_data():
        value = ctypes.c_void_p()

        lib.tensorGetData(
            null_tensor,
            ctypes.byref(value),
        )

    expect_runtime_error(
        "tensorGetData",
        test_get_data,
    )

    # ============================================================
    # tensorIsContiguous
    # ============================================================

    def test_is_contiguous():
        value = ctypes.c_uint8()

        lib.tensorIsContiguous(
            null_tensor,
            ctypes.byref(value),
        )

    expect_runtime_error(
        "tensorIsContiguous",
        test_is_contiguous,
    )

    print("Tensor getter C API error boundary test passed.")


if __name__ == "__main__":
    main()

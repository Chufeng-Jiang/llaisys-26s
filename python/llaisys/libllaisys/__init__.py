import ctypes
import os
import sys
from pathlib import Path

from .error import load_error
from .llaisys_types import (
    DataType,
    DeviceType,
    MemcpyKind,
    llaisysDataType_t,
    llaisysDeviceType_t,
    llaisysMemcpyKind_t,
    llaisysStream_t,
)
from .ops import load_ops
from .qwen2 import (
    LlaisysQwen2Meta,
    LlaisysQwen2Model,
    LlaisysQwen2Weights,
    llaisysQwen2Model_t,
    load_qwen2,
)
from .runtime import LlaisysRuntimeAPI, load_runtime
from .tensor import llaisysTensor_t, load_tensor


def load_shared_library():
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_error(LIB_LLAISYS)
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
load_qwen2(LIB_LLAISYS)

__all__ = [
    "LIB_LLAISYS",
    "DataType",
    "DeviceType",
    "LlaisysQwen2Meta",
    "LlaisysQwen2Model",
    "LlaisysQwen2Weights",
    "LlaisysRuntimeAPI",
    "MemcpyKind",
    "llaisysDataType_t",
    "llaisysDeviceType_t",
    "llaisysMemcpyKind_t",
    "llaisysQwen2Model_t",
    "llaisysStream_t",
    "llaisysTensor_t",
]

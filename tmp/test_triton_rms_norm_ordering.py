import sys
from pathlib import Path

import torch


# ============================================================
# Project paths
# ============================================================

repo_root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(repo_root / "python"))

sys.path.insert(0, str(repo_root / "test"))


# ============================================================
# Imports
# ============================================================

import llaisys

from llaisys.libllaisys import DeviceType
from llaisys.runtime import RuntimeAPI

from llaisys.triton import execution_context

from llaisys.triton.ops import add as triton_add

from llaisys.triton.ops import rms_norm as triton_rms_norm

from test_utils import (
    check_equal,
    device_name,
    dtype_name,
    llaisys_to_torch_memcpy_kind,
    random_tensor,
    reference_torch_device,
    torch_dtype,
)


# ============================================================
# Copy LLAISYS tensor -> PyTorch
# ============================================================
#
# PyTorch is used ONLY as storage for the synchronized reference.
#
# It does not calculate RMSNorm in this ordering test.
#
# ============================================================


def copy_llaisys_to_torch(tensor):
    shape = tensor.shape()
    strides = tensor.strides()

    right = 0

    for dim in range(len(shape)):
        if strides[dim] < 0:
            raise ValueError("Negative strides are not supported")

        if shape[dim] > 0:
            right += strides[dim] * (shape[dim] - 1)

    result_device_name = device_name(tensor.device_type())

    result_dtype = torch_dtype(dtype_name(tensor.dtype()))

    tmp = torch.zeros(
        (right + 1,), dtype=result_dtype, device=reference_torch_device(result_device_name, tensor.device_id())
    )

    result = torch.as_strided(tmp, shape, strides)

    runtime = RuntimeAPI(tensor.device_type())

    runtime.memcpy_sync(
        result.data_ptr(),
        tensor.data_ptr(),
        (right + 1) * tmp.element_size(),
        llaisys_to_torch_memcpy_kind(tensor.device_type()),
    )

    return result.clone()


# ============================================================
# Synchronized reference chain
# ============================================================
#
# Exact operator sequence:
#
#     Native Add
#         ↓
#     sync
#         ↓
#     Triton RMSNorm
#         ↓
#     sync
#         ↓
#     Triton Add
#         ↓
#     sync
#         ↓
#     Native Add
#         ↓
#     sync
#
# ============================================================


def run_synchronized_chain(runtime, x, residual, weight, bias, tmp_input, tmp_norm, tmp_add, out, eps):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Step 1
        #
        # Native:
        #
        # tmp_input = x + residual
        # ====================================================

        llaisys.Ops.add(tmp_input, x, residual)

        runtime.device_synchronize()

        # ====================================================
        # Step 2
        #
        # Triton:
        #
        # tmp_norm = RMSNorm(tmp_input, weight)
        # ====================================================

        triton_rms_norm(tmp_norm, tmp_input, weight, eps)

        runtime.device_synchronize()

        # ====================================================
        # Step 3
        #
        # Triton:
        #
        # tmp_add = tmp_norm + bias
        # ====================================================

        triton_add(tmp_add, tmp_norm, bias)

        runtime.device_synchronize()

        # ====================================================
        # Step 4
        #
        # Native:
        #
        # out = tmp_add + residual
        # ====================================================

        llaisys.Ops.add(out, tmp_add, residual)

        runtime.device_synchronize()


# ============================================================
# Asynchronous chain
# ============================================================
#
# Same operators.
#
# Same inputs.
#
# Same execution context.
#
# The ONLY difference:
#
#     no device_synchronize()
#
# between operators.
#
# ============================================================


def run_async_chain(x, residual, weight, bias, tmp_input, tmp_norm, tmp_add, out, eps):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Step 1
        # Native Add
        # ====================================================

        llaisys.Ops.add(tmp_input, x, residual)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Step 2
        # Triton RMSNorm
        # ====================================================

        triton_rms_norm(tmp_norm, tmp_input, weight, eps)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Step 3
        # Triton Add
        # ====================================================

        triton_add(tmp_add, tmp_norm, bias)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Step 4
        # Native Add
        # ====================================================

        llaisys.Ops.add(out, tmp_add, residual)


# ============================================================
# One ordering case
# ============================================================


def test_rms_norm_ordering(shape, dtype_name_value):
    ncol = shape[1]

    eps = 1e-5

    # ========================================================
    # Shared read-only inputs
    # ========================================================

    _, x = random_tensor(shape, dtype_name_value, "nvidia")

    _, residual = random_tensor(shape, dtype_name_value, "nvidia")

    _, weight = random_tensor((ncol,), dtype_name_value, "nvidia")

    _, bias = random_tensor(shape, dtype_name_value, "nvidia")

    # ========================================================
    # Synchronized reference intermediates
    # ========================================================

    _, sync_tmp_input = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_tmp_norm = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_tmp_add = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_out = random_tensor(shape, dtype_name_value, "nvidia")

    # ========================================================
    # Async intermediates
    # ========================================================

    _, async_tmp_input = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_tmp_norm = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_tmp_add = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_out = random_tensor(shape, dtype_name_value, "nvidia")

    # ========================================================
    # Runtime
    # ========================================================

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    runtime.set_device(0)

    # ========================================================
    # Synchronized reference
    # ========================================================

    run_synchronized_chain(
        runtime, x, residual, weight, bias, sync_tmp_input, sync_tmp_norm, sync_tmp_add, sync_out, eps
    )

    # ========================================================
    # Snapshot every synchronized intermediate
    # ========================================================

    sync_tmp_input_ref = copy_llaisys_to_torch(sync_tmp_input)

    sync_tmp_norm_ref = copy_llaisys_to_torch(sync_tmp_norm)

    sync_tmp_add_ref = copy_llaisys_to_torch(sync_tmp_add)

    sync_out_ref = copy_llaisys_to_torch(sync_out)

    # ========================================================
    # Async execution
    # ========================================================

    run_async_chain(x, residual, weight, bias, async_tmp_input, async_tmp_norm, async_tmp_add, async_out, eps)

    # ========================================================
    # First synchronization after complete async chain
    # ========================================================

    runtime.device_synchronize()

    # ========================================================
    # STRICT comparisons
    #
    # This is intentionally bitwise exact:
    #
    # synchronized path
    #     vs
    # asynchronous path
    #
    # both use exactly the same kernels.
    # ========================================================

    assert check_equal(async_tmp_input, sync_tmp_input_ref, strict=True), (
        f"Native Add ordering mismatch: shape={shape}, dtype={dtype_name_value}"
    )

    assert check_equal(async_tmp_norm, sync_tmp_norm_ref, strict=True), (
        f"Triton RMSNorm ordering mismatch: shape={shape}, dtype={dtype_name_value}"
    )

    assert check_equal(async_tmp_add, sync_tmp_add_ref, strict=True), (
        f"Triton Add ordering mismatch: shape={shape}, dtype={dtype_name_value}"
    )

    assert check_equal(async_out, sync_out_ref, strict=True), (
        f"Final Native Add ordering mismatch: shape={shape}, dtype={dtype_name_value}"
    )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("Testing RMSNorm mixed Native/Triton same-stream ordering")

    print()

    print("Reference:")

    print("  Native Add -> sync -> Triton RMSNorm -> sync -> Triton Add -> sync -> Native Add -> sync")

    print()

    print("Test:")

    print("  Native Add -> Triton RMSNorm -> Triton Add -> Native Add")

    print("  with NO intermediate synchronization")

    print()

    # ========================================================
    # Representative ordering cases
    #
    # We do NOT need to repeat every numerical-correctness
    # shape here.
    #
    # Numerical correctness already tested:
    #
    #     (1, 4)
    #     (1, 1536)
    #     (16, 1536)
    #     (1, 4096)
    #     (512, 4095)
    #     (512, 4096)
    #
    # Ordering test keeps representative:
    #
    #     tiny
    #     decode
    #     multi-row
    #     irregular large
    # ========================================================

    test_shapes = [(1, 4), (1, 1536), (16, 1536), (512, 4095)]

    test_dtypes = ["f32", "f16", "bf16"]

    rounds = 100

    # ========================================================
    # Stress test
    # ========================================================

    for round_index in range(rounds):
        for shape in test_shapes:
            for dtype_name_value in test_dtypes:
                test_rms_norm_ordering(shape, dtype_name_value)

        if (round_index + 1) % 10 == 0:
            print(f"  completed {round_index + 1}/{rounds} rounds")

    print()

    print("\033[92mRMSNorm synchronized-vs-async ordering test passed!\033[0m")

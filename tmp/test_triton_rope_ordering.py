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

from llaisys.triton.ops import rope as triton_rope

from test_utils import (
    arrange_tensor,
    check_equal,
    device_name,
    dtype_name,
    llaisys_to_torch_memcpy_kind,
    random_tensor,
    reference_torch_device,
    torch_dtype,
)


# ============================================================
# Copy LLAISYS Tensor -> PyTorch
# ============================================================
#
# PyTorch is used only as storage for comparison.
#
# It does NOT calculate the ordering reference.
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
# Synchronized reference
# ============================================================
#
# Exact chain:
#
#     Native Add
#         ↓
#     sync
#         ↓
#     Triton RoPE
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


def run_synchronized_chain(runtime, x, residual, bias, pos_ids, tmp_input, tmp_rope, tmp_add, out, theta):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Step 1
        #
        # Native Add
        #
        # tmp_input = x + residual
        # ====================================================

        llaisys.Ops.add(tmp_input, x, residual)

        runtime.device_synchronize()

        # ====================================================
        # Step 2
        #
        # Triton RoPE
        # ====================================================

        triton_rope(tmp_rope, tmp_input, pos_ids, theta)

        runtime.device_synchronize()

        # ====================================================
        # Step 3
        #
        # Triton Add
        # ====================================================

        triton_add(tmp_add, tmp_rope, bias)

        runtime.device_synchronize()

        # ====================================================
        # Step 4
        #
        # Native Add
        # ====================================================

        llaisys.Ops.add(out, tmp_add, residual)

        runtime.device_synchronize()


# ============================================================
# Asynchronous chain
# ============================================================
#
# Same kernels.
#
# Same inputs.
#
# Same stream.
#
# No synchronization between operators.
# ============================================================


def run_async_chain(x, residual, bias, pos_ids, tmp_input, tmp_rope, tmp_add, out, theta):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Native Add
        # ====================================================

        llaisys.Ops.add(tmp_input, x, residual)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Triton RoPE
        # ====================================================

        triton_rope(tmp_rope, tmp_input, pos_ids, theta)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Triton Add
        # ====================================================

        triton_add(tmp_add, tmp_rope, bias)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Native Add
        # ====================================================

        llaisys.Ops.add(out, tmp_add, residual)


# ============================================================
# One ordering test
# ============================================================


def test_rope_ordering(shape, start_end, dtype_name_value):
    theta = 10000.0

    # ========================================================
    # Shared read-only inputs
    # ========================================================

    _, x = random_tensor(shape, dtype_name_value, "nvidia")

    _, residual = random_tensor(shape, dtype_name_value, "nvidia")

    _, bias = random_tensor(shape, dtype_name_value, "nvidia")

    _, pos_ids = arrange_tensor(start_end[0], start_end[1], "nvidia")

    # ========================================================
    # Synchronized intermediates
    # ========================================================

    _, sync_tmp_input = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_tmp_rope = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_tmp_add = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_out = random_tensor(shape, dtype_name_value, "nvidia")

    # ========================================================
    # Async intermediates
    # ========================================================

    _, async_tmp_input = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_tmp_rope = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_tmp_add = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_out = random_tensor(shape, dtype_name_value, "nvidia")

    # ========================================================
    # Runtime
    # ========================================================

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    runtime.set_device(0)

    # ========================================================
    # Reference execution
    # ========================================================

    run_synchronized_chain(
        runtime, x, residual, bias, pos_ids, sync_tmp_input, sync_tmp_rope, sync_tmp_add, sync_out, theta
    )

    # ========================================================
    # Snapshot synchronized results
    # ========================================================

    sync_tmp_input_ref = copy_llaisys_to_torch(sync_tmp_input)

    sync_tmp_rope_ref = copy_llaisys_to_torch(sync_tmp_rope)

    sync_tmp_add_ref = copy_llaisys_to_torch(sync_tmp_add)

    sync_out_ref = copy_llaisys_to_torch(sync_out)

    # ========================================================
    # Async execution
    # ========================================================

    run_async_chain(x, residual, bias, pos_ids, async_tmp_input, async_tmp_rope, async_tmp_add, async_out, theta)

    # ========================================================
    # First synchronization after entire chain
    # ========================================================

    runtime.device_synchronize()

    # ========================================================
    # Strict comparisons
    #
    # Same kernels + same input.
    #
    # Therefore synchronized and asynchronous paths should be
    # bitwise identical.
    # ========================================================

    assert check_equal(async_tmp_input, sync_tmp_input_ref, strict=True), (
        f"Native Add ordering mismatch: shape={shape}, dtype={dtype_name_value}"
    )

    assert check_equal(async_tmp_rope, sync_tmp_rope_ref, strict=True), (
        f"Triton RoPE ordering mismatch: shape={shape}, range={start_end}, dtype={dtype_name_value}"
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
    print("Testing RoPE mixed Native/Triton same-stream ordering")

    print()

    print("Reference:")

    print("  Native Add -> sync -> Triton RoPE -> sync -> Triton Add -> sync -> Native Add -> sync")

    print()

    print("Test:")

    print("  Native Add -> Triton RoPE -> Triton Add -> Native Add")

    print("  with NO intermediate synchronization")

    print()

    # ========================================================
    # Representative ordering cases
    #
    # Numerical correctness already covers the full large
    # 512-token stress matrix.
    #
    # Ordering only needs representative dependency patterns.
    #
    # Keep d=4096 here, but use seq=64 to avoid making the
    # 100-round ordering stress unnecessarily expensive.
    # ========================================================

    test_cases = [
        ((2, 1, 4), (0, 2)),
        ((1, 12, 128), (512, 513)),
        ((64, 12, 128), (512, 576)),
        ((64, 4, 4096), (512, 576)),
    ]

    test_dtypes = ["f32", "f16", "bf16"]

    rounds = 100

    # ========================================================
    # Stress test
    # ========================================================

    for round_index in range(rounds):
        for shape, start_end in test_cases:
            for dtype_name_value in test_dtypes:
                test_rope_ordering(shape, start_end, dtype_name_value)

        if (round_index + 1) % 10 == 0:
            print(f"  completed {round_index + 1}/{rounds} rounds")

    print()

    print("\033[92mRoPE synchronized-vs-async ordering test passed!\033[0m")

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

from llaisys.triton.ops import argmax as triton_argmax

from test_utils import (
    check_equal,
    device_name,
    dtype_name,
    llaisys_to_torch_memcpy_kind,
    random_tensor,
    reference_torch_device,
    torch_dtype,
    zero_tensor,
)


# ============================================================
# Copy LLAISYS tensor -> PyTorch
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
#
#     Triton Argmax
#         ↓
#     sync
#
#     Triton Add
#         ↓
#     sync
#
#     Native Add
#         ↓
#     sync
#
#
# This tests:
#
#     Native → Triton
#
#     internal Argmax:
#         Stage1 → StageN → StageN
#
#     Triton → Triton
#
#     Triton → Native
#
# ============================================================


def run_synchronized_chain(runtime, a, b, scalar_bias_1, scalar_bias_2, tmp_vals, max_idx, max_val, tmp_scalar, out):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Step 1
        #
        # Native Add
        #
        # tmp_vals = a + b
        # ====================================================

        llaisys.Ops.add(tmp_vals, a, b)

        runtime.device_synchronize()

        # ====================================================
        # Step 2
        #
        # Triton Argmax
        #
        # max_val = max(tmp_vals)
        # max_idx = argmax(tmp_vals)
        #
        # For large tensors this itself contains:
        #
        #     Stage 1
        #       ↓
        #     Stage N
        #       ↓
        #     Stage N ...
        # ====================================================

        triton_argmax(max_idx, max_val, tmp_vals)

        runtime.device_synchronize()

        # ====================================================
        # Step 3
        #
        # Triton Add
        #
        # tmp_scalar = max_val + scalar_bias_1
        # ====================================================

        triton_add(tmp_scalar, max_val, scalar_bias_1)

        runtime.device_synchronize()

        # ====================================================
        # Step 4
        #
        # Native Add
        #
        # out = tmp_scalar + scalar_bias_2
        # ====================================================

        llaisys.Ops.add(out, tmp_scalar, scalar_bias_2)

        runtime.device_synchronize()


# ============================================================
# Async chain
# ============================================================
#
# Same kernels.
#
# Same inputs.
#
# No intermediate synchronization.
# ============================================================


def run_async_chain(a, b, scalar_bias_1, scalar_bias_2, tmp_vals, max_idx, max_val, tmp_scalar, out):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Native Add
        # ====================================================

        llaisys.Ops.add(tmp_vals, a, b)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Triton Argmax
        # ====================================================

        triton_argmax(max_idx, max_val, tmp_vals)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Triton Add
        # ====================================================

        triton_add(tmp_scalar, max_val, scalar_bias_1)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Native Add
        # ====================================================

        llaisys.Ops.add(out, tmp_scalar, scalar_bias_2)


# ============================================================
# One ordering case
# ============================================================


def test_argmax_ordering(numel, dtype_name_value):
    shape = (numel,)

    # ========================================================
    # Shared read-only vector inputs
    # ========================================================

    _, a = random_tensor(shape, dtype_name_value, "nvidia")

    _, b = random_tensor(shape, dtype_name_value, "nvidia")

    # ========================================================
    # Shared scalar biases
    # ========================================================

    _, scalar_bias_1 = random_tensor((1,), dtype_name_value, "nvidia")

    _, scalar_bias_2 = random_tensor((1,), dtype_name_value, "nvidia")

    # ========================================================
    # Synchronized tensors
    # ========================================================

    _, sync_tmp_vals = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_max_idx = zero_tensor((1,), "i64", "nvidia")

    _, sync_max_val = zero_tensor((1,), dtype_name_value, "nvidia")

    _, sync_tmp_scalar = zero_tensor((1,), dtype_name_value, "nvidia")

    _, sync_out = zero_tensor((1,), dtype_name_value, "nvidia")

    # ========================================================
    # Async tensors
    # ========================================================

    _, async_tmp_vals = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_max_idx = zero_tensor((1,), "i64", "nvidia")

    _, async_max_val = zero_tensor((1,), dtype_name_value, "nvidia")

    _, async_tmp_scalar = zero_tensor((1,), dtype_name_value, "nvidia")

    _, async_out = zero_tensor((1,), dtype_name_value, "nvidia")

    # ========================================================
    # Runtime
    # ========================================================

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    runtime.set_device(0)

    # ========================================================
    # Run synchronized reference
    # ========================================================

    run_synchronized_chain(
        runtime,
        a,
        b,
        scalar_bias_1,
        scalar_bias_2,
        sync_tmp_vals,
        sync_max_idx,
        sync_max_val,
        sync_tmp_scalar,
        sync_out,
    )

    # ========================================================
    # Snapshot synchronized results
    # ========================================================

    sync_tmp_vals_ref = copy_llaisys_to_torch(sync_tmp_vals)

    sync_max_idx_ref = copy_llaisys_to_torch(sync_max_idx)

    sync_max_val_ref = copy_llaisys_to_torch(sync_max_val)

    sync_tmp_scalar_ref = copy_llaisys_to_torch(sync_tmp_scalar)

    sync_out_ref = copy_llaisys_to_torch(sync_out)

    # ========================================================
    # Run asynchronous chain
    # ========================================================

    run_async_chain(
        a, b, scalar_bias_1, scalar_bias_2, async_tmp_vals, async_max_idx, async_max_val, async_tmp_scalar, async_out
    )

    # ========================================================
    # Only synchronization after complete async chain
    # ========================================================

    runtime.device_synchronize()

    # ========================================================
    # Stage-by-stage strict equality
    # ========================================================

    assert check_equal(async_tmp_vals, sync_tmp_vals_ref, strict=True), (
        f"Native Add ordering mismatch: numel={numel}, dtype={dtype_name_value}"
    )

    assert check_equal(async_max_idx, sync_max_idx_ref, strict=True), (
        f"Triton Argmax index ordering mismatch: numel={numel}, dtype={dtype_name_value}"
    )

    assert check_equal(async_max_val, sync_max_val_ref, strict=True), (
        f"Triton Argmax value ordering mismatch: numel={numel}, dtype={dtype_name_value}"
    )

    assert check_equal(async_tmp_scalar, sync_tmp_scalar_ref, strict=True), (
        f"Triton Add ordering mismatch: numel={numel}, dtype={dtype_name_value}"
    )

    assert check_equal(async_out, sync_out_ref, strict=True), (
        f"Final Native Add ordering mismatch: numel={numel}, dtype={dtype_name_value}"
    )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("Testing Argmax mixed Native/Triton same-stream ordering")

    print()

    print("Reference:")

    print("  Native Add -> sync -> Triton Argmax -> sync -> Triton Add -> sync -> Native Add -> sync")

    print()

    print("Test:")

    print("  Native Add -> Triton Argmax -> Triton Add -> Native Add")

    print("  with NO intermediate synchronization")

    print()

    # ========================================================
    # Representative reduction depths
    #
    # 4:
    #     trivial single-stage
    #
    # 1024:
    #     exact Stage-1 block
    #
    # 1025:
    #     first multi-block boundary
    #
    # 4097:
    #     several Stage-1 blocks
    #
    # 151936:
    #     vocabulary-like workload
    #
    # 2097152:
    #
    #     Stage1:
    #         2,097,152 / 1024 = 2048 partials
    #
    #     StageN:
    #         2048 → 2
    #
    #     StageN:
    #         2 → 1
    #
    # This explicitly tests a 3-kernel Argmax chain.
    # ========================================================

    test_sizes = [4, 1024, 1025, 4097, 151936, 2097152]

    test_dtypes = ["f32", "f16", "bf16"]

    # ========================================================
    # Keep the 2M-element stress case manageable.
    #
    # Correctness has already exhaustively tested the semantic
    # cases. Here we are testing ordering, not rediscovering
    # numerical semantics.
    # ========================================================

    rounds = 20

    for round_index in range(rounds):
        for numel in test_sizes:
            for dtype_name_value in test_dtypes:
                test_argmax_ordering(numel, dtype_name_value)

        if (round_index + 1) % 5 == 0:
            print(f"  completed {round_index + 1}/{rounds} rounds")

    print()

    print("\033[92mArgmax synchronized-vs-async ordering test passed!\033[0m")

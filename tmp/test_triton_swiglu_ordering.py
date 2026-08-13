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

from llaisys.triton.ops import swiglu as triton_swiglu

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
# Copy one LLAISYS Tensor to a PyTorch Tensor
# ============================================================
#
# This mirrors the copy logic used by check_equal().
#
# We need this because our ordering reference is NOT PyTorch
# computation.
#
# Instead:
#
#     synchronized LLAISYS execution
#
# is the reference for:
#
#     asynchronous LLAISYS execution
#
# ============================================================


def copy_llaisys_to_torch(tensor):
    shape = tensor.shape()
    strides = tensor.strides()

    # --------------------------------------------------------
    # Compute backing storage size needed for the strided view.
    # --------------------------------------------------------

    right = 0

    for dim in range(len(shape)):
        if strides[dim] < 0:
            raise ValueError("Negative strides are not supported")

        if shape[dim] > 0:
            right += strides[dim] * (shape[dim] - 1)

    result_device_name = device_name(tensor.device_type())

    result_dtype = torch_dtype(dtype_name(tensor.dtype()))

    # --------------------------------------------------------
    # Allocate PyTorch backing storage
    # --------------------------------------------------------

    tmp = torch.zeros(
        (right + 1,), dtype=result_dtype, device=reference_torch_device(result_device_name, tensor.device_id())
    )

    # --------------------------------------------------------
    # Recreate the same shape / strides
    # --------------------------------------------------------

    result = torch.as_strided(tmp, shape, strides)

    # --------------------------------------------------------
    # Copy LLAISYS memory -> PyTorch
    # --------------------------------------------------------

    runtime = RuntimeAPI(tensor.device_type())

    runtime.memcpy_sync(
        result.data_ptr(),
        tensor.data_ptr(),
        (right + 1) * tmp.element_size(),
        llaisys_to_torch_memcpy_kind(tensor.device_type()),
    )

    return result.clone()


# ============================================================
# Synchronized reference execution
# ============================================================
#
# IMPORTANT:
#
# This uses EXACTLY the same implementations as the async test:
#
#     Native Add
#     Triton SwiGLU
#     Triton Add
#     Native Add
#
# The only difference is:
#
#     device_synchronize()
#
# after every operator.
#
# ============================================================


def run_synchronized_chain(runtime, gate, up, bias, tmp_gate, tmp_swiglu, tmp_add, out):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ----------------------------------------------------
        # Step 1
        #
        # Native Add
        #
        # tmp_gate = gate + bias
        # ----------------------------------------------------

        llaisys.Ops.add(tmp_gate, gate, bias)

        runtime.device_synchronize()

        # ----------------------------------------------------
        # Step 2
        #
        # Triton SwiGLU
        # ----------------------------------------------------

        triton_swiglu(tmp_swiglu, tmp_gate, up)

        runtime.device_synchronize()

        # ----------------------------------------------------
        # Step 3
        #
        # Triton Add
        # ----------------------------------------------------

        triton_add(tmp_add, tmp_swiglu, bias)

        runtime.device_synchronize()

        # ----------------------------------------------------
        # Step 4
        #
        # Native Add
        # ----------------------------------------------------

        llaisys.Ops.add(out, tmp_add, bias)

        runtime.device_synchronize()


# ============================================================
# Asynchronous execution
# ============================================================
#
# Same kernels.
#
# Same execution context.
#
# Same inputs.
#
# But absolutely NO synchronization between operators.
#
# ============================================================


def run_async_chain(gate, up, bias, tmp_gate, tmp_swiglu, tmp_add, out):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ----------------------------------------------------
        # Native Add
        # ----------------------------------------------------

        llaisys.Ops.add(tmp_gate, gate, bias)

        # ----------------------------------------------------
        # NO synchronization
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Triton SwiGLU
        # ----------------------------------------------------

        triton_swiglu(tmp_swiglu, tmp_gate, up)

        # ----------------------------------------------------
        # NO synchronization
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Triton Add
        # ----------------------------------------------------

        triton_add(tmp_add, tmp_swiglu, bias)

        # ----------------------------------------------------
        # NO synchronization
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Native Add
        # ----------------------------------------------------

        llaisys.Ops.add(out, tmp_add, bias)


# ============================================================
# One test case
# ============================================================


def test_swiglu_ordering(shape, dtype_name_value):
    # ========================================================
    # Inputs
    # ========================================================
    #
    # Both reference and async chains use exactly the same
    # input tensors.
    #
    # Inputs are read-only.
    # ========================================================

    _, gate = random_tensor(shape, dtype_name_value, "nvidia")

    _, up = random_tensor(shape, dtype_name_value, "nvidia")

    _, bias = random_tensor(shape, dtype_name_value, "nvidia")

    # ========================================================
    # Synchronized reference outputs
    # ========================================================

    _, sync_tmp_gate = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_tmp_swiglu = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_tmp_add = random_tensor(shape, dtype_name_value, "nvidia")

    _, sync_out = random_tensor(shape, dtype_name_value, "nvidia")

    # ========================================================
    # Async outputs
    # ========================================================

    _, async_tmp_gate = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_tmp_swiglu = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_tmp_add = random_tensor(shape, dtype_name_value, "nvidia")

    _, async_out = random_tensor(shape, dtype_name_value, "nvidia")

    # ========================================================
    # Runtime
    # ========================================================

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    runtime.set_device(0)

    # ========================================================
    # Reference:
    #
    # synchronize after every operator
    # ========================================================

    run_synchronized_chain(runtime, gate, up, bias, sync_tmp_gate, sync_tmp_swiglu, sync_tmp_add, sync_out)

    # ========================================================
    # Snapshot synchronized reference results
    #
    # Convert them to PyTorch ONLY for comparison.
    #
    # PyTorch does not compute the reference math here.
    # ========================================================

    sync_tmp_gate_ref = copy_llaisys_to_torch(sync_tmp_gate)

    sync_tmp_swiglu_ref = copy_llaisys_to_torch(sync_tmp_swiglu)

    sync_tmp_add_ref = copy_llaisys_to_torch(sync_tmp_add)

    sync_out_ref = copy_llaisys_to_torch(sync_out)

    # ========================================================
    # Test:
    #
    # same operators with zero intermediate synchronization
    # ========================================================

    run_async_chain(gate, up, bias, async_tmp_gate, async_tmp_swiglu, async_tmp_add, async_out)

    # ========================================================
    # First synchronization after the entire async chain
    # ========================================================

    runtime.device_synchronize()

    # ========================================================
    # Strict intermediate comparisons
    #
    # Because the synchronized and asynchronous paths use the
    # exact same kernels and exact same input values, the
    # results should be bitwise identical.
    #
    # No tolerance is needed.
    # ========================================================

    assert check_equal(async_tmp_gate, sync_tmp_gate_ref, strict=True), (
        f"Native Add ordering mismatch: shape={shape}, dtype={dtype_name_value}"
    )

    assert check_equal(async_tmp_swiglu, sync_tmp_swiglu_ref, strict=True), (
        f"Triton SwiGLU ordering mismatch: shape={shape}, dtype={dtype_name_value}"
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
    print("Testing SwiGLU mixed Native/Triton same-stream ordering")

    print()

    print("Reference:")

    print("  Native Add -> sync -> Triton SwiGLU -> sync -> Triton Add -> sync -> Native Add -> sync")

    print()

    print("Test:")

    print("  Native Add -> Triton SwiGLU -> Triton Add -> Native Add")

    print("  with NO intermediate synchronization")

    print()

    # ========================================================
    # Cases
    # ========================================================

    test_cases = [
        ((2, 3), "f32"),
        ((33, 65), "f32"),
        ((512, 4096), "f32"),
        ((2, 3), "f16"),
        ((33, 65), "f16"),
        ((512, 4096), "f16"),
        ((2, 3), "bf16"),
        ((33, 65), "bf16"),
        ((512, 4096), "bf16"),
    ]

    # ========================================================
    # Stress rounds
    # ========================================================

    rounds = 100

    for round_index in range(rounds):
        for shape, dtype_name_value in test_cases:
            test_swiglu_ordering(shape, dtype_name_value)

        if (round_index + 1) % 10 == 0:
            print(f"  completed {round_index + 1}/{rounds} rounds")

    print()

    print("\033[92mSwiGLU synchronized-vs-async ordering test passed!\033[0m")

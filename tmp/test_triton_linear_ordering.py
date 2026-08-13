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

from llaisys.triton.ops import linear as triton_linear

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
#
# Chain:
#
#     Native Add
#         ↓
#     sync
#
#     Triton Linear
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
# This verifies:
#
#     Native write
#         ↓
#     Triton GEMM reads new input
#
#     Triton GEMM write
#         ↓
#     Triton Add reads new output
#
#     Triton write
#         ↓
#     Native Add reads new output
# ============================================================


def run_synchronized_chain(runtime, x_a, x_b, weight, bias, post_a, post_b, tmp_x, tmp_linear, tmp_add, out):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Step 1
        #
        # Native Add
        #
        # tmp_x = x_a + x_b
        # ====================================================

        llaisys.Ops.add(tmp_x, x_a, x_b)

        runtime.device_synchronize()

        # ====================================================
        # Step 2
        #
        # Triton Linear
        #
        # tmp_linear =
        #
        #     tmp_x @ weight.T + bias
        #
        # This must observe the Native Add result.
        # ====================================================

        triton_linear(tmp_linear, tmp_x, weight, bias)

        runtime.device_synchronize()

        # ====================================================
        # Step 3
        #
        # Triton Add
        # ====================================================

        triton_add(tmp_add, tmp_linear, post_a)

        runtime.device_synchronize()

        # ====================================================
        # Step 4
        #
        # Native Add
        # ====================================================

        llaisys.Ops.add(out, tmp_add, post_b)

        runtime.device_synchronize()


# ============================================================
# Asynchronous chain
#
# Same operations.
#
# No intermediate synchronization.
# ============================================================


def run_async_chain(x_a, x_b, weight, bias, post_a, post_b, tmp_x, tmp_linear, tmp_add, out):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Native Add
        # ====================================================

        llaisys.Ops.add(tmp_x, x_a, x_b)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Triton Linear
        # ====================================================

        triton_linear(tmp_linear, tmp_x, weight, bias)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Triton Add
        # ====================================================

        triton_add(tmp_add, tmp_linear, post_a)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Native Add
        # ====================================================

        llaisys.Ops.add(out, tmp_add, post_b)


# ============================================================
# One ordering case
# ============================================================


def test_linear_ordering(m, n, k, use_bias, dtype_name_value):
    x_shape = (m, k)

    weight_shape = (n, k)

    output_shape = (m, n)

    # ========================================================
    # Shared X inputs
    #
    # Native Add produces:
    #
    #     tmp_x = x_a + x_b
    # ========================================================

    _, x_a = random_tensor(x_shape, dtype_name_value, "nvidia", scale=0.1)

    _, x_b = random_tensor(x_shape, dtype_name_value, "nvidia", scale=0.1)

    # ========================================================
    # Shared Linear weight
    # ========================================================

    _, weight = random_tensor(weight_shape, dtype_name_value, "nvidia", scale=0.01)

    # ========================================================
    # Optional bias
    # ========================================================

    bias = None

    if use_bias:
        _, bias = random_tensor((n,), dtype_name_value, "nvidia")

    # ========================================================
    # Post-Linear Add inputs
    # ========================================================

    _, post_a = random_tensor(output_shape, dtype_name_value, "nvidia")

    _, post_b = random_tensor(output_shape, dtype_name_value, "nvidia")

    # ========================================================
    # Synchronized intermediates
    # ========================================================

    _, sync_tmp_x = random_tensor(x_shape, dtype_name_value, "nvidia")

    _, sync_tmp_linear = random_tensor(output_shape, dtype_name_value, "nvidia")

    _, sync_tmp_add = random_tensor(output_shape, dtype_name_value, "nvidia")

    _, sync_out = random_tensor(output_shape, dtype_name_value, "nvidia")

    # ========================================================
    # Async intermediates
    # ========================================================

    _, async_tmp_x = random_tensor(x_shape, dtype_name_value, "nvidia")

    _, async_tmp_linear = random_tensor(output_shape, dtype_name_value, "nvidia")

    _, async_tmp_add = random_tensor(output_shape, dtype_name_value, "nvidia")

    _, async_out = random_tensor(output_shape, dtype_name_value, "nvidia")

    # ========================================================
    # Runtime
    # ========================================================

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    runtime.set_device(0)

    # ========================================================
    # Synchronized reference
    # ========================================================

    run_synchronized_chain(
        runtime, x_a, x_b, weight, bias, post_a, post_b, sync_tmp_x, sync_tmp_linear, sync_tmp_add, sync_out
    )

    # ========================================================
    # Snapshot synchronized outputs
    # ========================================================

    sync_tmp_x_ref = copy_llaisys_to_torch(sync_tmp_x)

    sync_tmp_linear_ref = copy_llaisys_to_torch(sync_tmp_linear)

    sync_tmp_add_ref = copy_llaisys_to_torch(sync_tmp_add)

    sync_out_ref = copy_llaisys_to_torch(sync_out)

    # ========================================================
    # Async chain
    # ========================================================

    run_async_chain(x_a, x_b, weight, bias, post_a, post_b, async_tmp_x, async_tmp_linear, async_tmp_add, async_out)

    # ========================================================
    # First synchronization only AFTER complete chain
    # ========================================================

    runtime.device_synchronize()

    # ========================================================
    # IMPORTANT:
    #
    # Here strict equality is appropriate.
    #
    # We are NOT comparing:
    #
    #     cuBLAS vs Triton
    #
    # We are comparing:
    #
    #     same Triton Linear
    #     same inputs
    #     same operation sequence
    #
    # with synchronization being the only difference.
    # ========================================================

    assert check_equal(async_tmp_x, sync_tmp_x_ref, strict=True), (
        f"Native Add -> Linear input ordering mismatch: M={m}, N={n}, K={k}, bias={use_bias}, dtype={dtype_name_value}"
    )

    assert check_equal(async_tmp_linear, sync_tmp_linear_ref, strict=True), (
        f"Triton Linear ordering mismatch: M={m}, N={n}, K={k}, bias={use_bias}, dtype={dtype_name_value}"
    )

    assert check_equal(async_tmp_add, sync_tmp_add_ref, strict=True), (
        f"Triton Add after Linear ordering mismatch: M={m}, N={n}, K={k}, bias={use_bias}, dtype={dtype_name_value}"
    )

    assert check_equal(async_out, sync_out_ref, strict=True), (
        f"Final Native Add ordering mismatch: M={m}, N={n}, K={k}, bias={use_bias}, dtype={dtype_name_value}"
    )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("Testing Linear mixed Native/Triton same-stream ordering")

    print()

    print("Reference:")

    print("  Native Add -> sync -> Triton Linear -> sync -> Triton Add -> sync -> Native Add -> sync")

    print()

    print("Test:")

    print("  Native Add -> Triton Linear -> Triton Add -> Native Add")

    print("  with NO intermediate synchronization")

    print()

    # ========================================================
    # Representative cases
    #
    # Tiny:
    #     basic matrix mapping + bias
    #
    # Tail:
    #     M / N / K all cross tile boundaries
    #
    # Decode:
    #     M = 1
    #
    # Small-batch:
    #     M = 32
    #
    # Full 4096×4096 correctness was already tested above.
    # Ordering does not need to repeatedly allocate/run that
    # expensive case.
    # ========================================================

    test_cases = [(2, 3, 4, True), (17, 37, 33, True), (1, 1536, 1536, False), (32, 1536, 1536, True)]

    test_dtypes = ["f32", "f16", "bf16"]

    # ========================================================
    # Ordering only tests stream dependency.
    #
    # Large 4096 functional workloads have already passed,
    # therefore 20 rounds here are enough for a robust
    # synchronization-vs-async stress test.
    # ========================================================

    rounds = 20

    for round_index in range(rounds):
        for m, n, k, use_bias in test_cases:
            for dtype_name_value in test_dtypes:
                test_linear_ordering(m, n, k, use_bias, dtype_name_value)

        if (round_index + 1) % 5 == 0:
            print(f"  completed {round_index + 1}/{rounds} rounds")

    print()

    print("\033[92mLinear synchronized-vs-async ordering test passed!\033[0m")

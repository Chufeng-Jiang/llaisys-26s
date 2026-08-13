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

from llaisys.triton.ops import embedding as triton_embedding

from test_utils import (
    check_equal,
    device_name,
    dtype_name,
    llaisys_to_torch_memcpy_kind,
    random_int_tensor,
    random_tensor,
    reference_torch_device,
    torch_dtype,
)


# ============================================================
# Copy LLAISYS Tensor -> PyTorch
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
# Chain:
#
#     Native Add
#         ↓
#     sync
#
#     Triton Embedding
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
# ============================================================


def run_synchronized_chain(runtime, weight_a, weight_b, index, bias_a, bias_b, tmp_weight, tmp_embedding, tmp_add, out):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Step 1
        #
        # Native Add
        #
        # tmp_weight = weight_a + weight_b
        #
        # The Triton Embedding that follows must observe this
        # newly written embedding table.
        # ====================================================

        llaisys.Ops.add(tmp_weight, weight_a, weight_b)

        runtime.device_synchronize()

        # ====================================================
        # Step 2
        #
        # Triton Embedding
        #
        # tmp_embedding =
        #     tmp_weight[index, :]
        # ====================================================

        triton_embedding(tmp_embedding, index, tmp_weight)

        runtime.device_synchronize()

        # ====================================================
        # Step 3
        #
        # Triton Add
        # ====================================================

        triton_add(tmp_add, tmp_embedding, bias_a)

        runtime.device_synchronize()

        # ====================================================
        # Step 4
        #
        # Native Add
        # ====================================================

        llaisys.Ops.add(out, tmp_add, bias_b)

        runtime.device_synchronize()


# ============================================================
# Asynchronous chain
# ============================================================
#
# Exact same operations.
#
# No synchronization between operators.
# ============================================================


def run_async_chain(weight_a, weight_b, index, bias_a, bias_b, tmp_weight, tmp_embedding, tmp_add, out):
    with execution_context(DeviceType.NVIDIA, device_id=0):
        # ====================================================
        # Native Add
        # ====================================================

        llaisys.Ops.add(tmp_weight, weight_a, weight_b)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Triton Embedding
        # ====================================================

        triton_embedding(tmp_embedding, index, tmp_weight)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Triton Add
        # ====================================================

        triton_add(tmp_add, tmp_embedding, bias_a)

        # ====================================================
        # NO synchronization
        # ====================================================

        # ====================================================
        # Native Add
        # ====================================================

        llaisys.Ops.add(out, tmp_add, bias_b)


# ============================================================
# One ordering case
# ============================================================


def test_embedding_ordering(index_count, vocabulary_size, embedding_dim, dtype_name_value):
    weight_shape = (vocabulary_size, embedding_dim)

    index_shape = (index_count,)

    output_shape = (index_count, embedding_dim)

    # ========================================================
    # Shared read-only embedding-table inputs
    # ========================================================

    _, weight_a = random_tensor(weight_shape, dtype_name_value, "nvidia")

    _, weight_b = random_tensor(weight_shape, dtype_name_value, "nvidia")

    # ========================================================
    # Shared valid Int64 indices
    # ========================================================

    _, index = random_int_tensor(index_shape, "nvidia", high=vocabulary_size)

    # ========================================================
    # Shared output biases
    # ========================================================

    _, bias_a = random_tensor(output_shape, dtype_name_value, "nvidia")

    _, bias_b = random_tensor(output_shape, dtype_name_value, "nvidia")

    # ========================================================
    # Synchronized intermediates
    # ========================================================

    _, sync_tmp_weight = random_tensor(weight_shape, dtype_name_value, "nvidia")

    _, sync_tmp_embedding = random_tensor(output_shape, dtype_name_value, "nvidia")

    _, sync_tmp_add = random_tensor(output_shape, dtype_name_value, "nvidia")

    _, sync_out = random_tensor(output_shape, dtype_name_value, "nvidia")

    # ========================================================
    # Async intermediates
    # ========================================================

    _, async_tmp_weight = random_tensor(weight_shape, dtype_name_value, "nvidia")

    _, async_tmp_embedding = random_tensor(output_shape, dtype_name_value, "nvidia")

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
        runtime, weight_a, weight_b, index, bias_a, bias_b, sync_tmp_weight, sync_tmp_embedding, sync_tmp_add, sync_out
    )

    # ========================================================
    # Snapshot synchronized results
    # ========================================================

    sync_tmp_weight_ref = copy_llaisys_to_torch(sync_tmp_weight)

    sync_tmp_embedding_ref = copy_llaisys_to_torch(sync_tmp_embedding)

    sync_tmp_add_ref = copy_llaisys_to_torch(sync_tmp_add)

    sync_out_ref = copy_llaisys_to_torch(sync_out)

    # ========================================================
    # Async execution
    # ========================================================

    run_async_chain(
        weight_a, weight_b, index, bias_a, bias_b, async_tmp_weight, async_tmp_embedding, async_tmp_add, async_out
    )

    # ========================================================
    # First synchronization only AFTER the complete chain
    # ========================================================

    runtime.device_synchronize()

    # ========================================================
    # Strict stage-by-stage comparisons
    # ========================================================

    assert check_equal(async_tmp_weight, sync_tmp_weight_ref, strict=True), (
        "Native Add -> Embedding weight ordering mismatch: "
        f"index_count={index_count}, "
        f"vocab={vocabulary_size}, "
        f"dim={embedding_dim}, "
        f"dtype={dtype_name_value}"
    )

    assert check_equal(async_tmp_embedding, sync_tmp_embedding_ref, strict=True), (
        "Triton Embedding ordering mismatch: "
        f"index_count={index_count}, "
        f"vocab={vocabulary_size}, "
        f"dim={embedding_dim}, "
        f"dtype={dtype_name_value}"
    )

    assert check_equal(async_tmp_add, sync_tmp_add_ref, strict=True), (
        "Triton Add after Embedding ordering mismatch: "
        f"index_count={index_count}, "
        f"vocab={vocabulary_size}, "
        f"dim={embedding_dim}, "
        f"dtype={dtype_name_value}"
    )

    assert check_equal(async_out, sync_out_ref, strict=True), (
        "Final Native Add ordering mismatch: "
        f"index_count={index_count}, "
        f"vocab={vocabulary_size}, "
        f"dim={embedding_dim}, "
        f"dtype={dtype_name_value}"
    )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("Testing Embedding mixed Native/Triton same-stream ordering")

    print()

    print("Reference:")

    print("  Native Add -> sync -> Triton Embedding -> sync -> Triton Add -> sync -> Native Add -> sync")

    print()

    print("Test:")

    print("  Native Add -> Triton Embedding -> Triton Add -> Native Add")

    print("  with NO intermediate synchronization")

    print()

    # ========================================================
    # Representative cases
    #
    # 3:
    #     tiny irregular width
    #
    # 127 / 128 / 129:
    #     Triton BLOCK_SIZE boundary
    #
    # 4095:
    #     large tail-mask case
    #
    # 4096:
    #     realistic aligned hidden dimension
    # ========================================================

    test_cases = [(1, 2, 3), (7, 17, 127), (8, 17, 128), (9, 17, 129), (33, 257, 4095), (50, 512, 4096)]

    test_dtypes = ["f32", "f16", "bf16"]

    # ========================================================
    # Embedding is substantially lighter than the Argmax
    # multi-stage stress test, so 100 rounds is reasonable.
    # ========================================================

    rounds = 100

    for round_index in range(rounds):
        for index_count, vocabulary_size, embedding_dim in test_cases:
            for dtype_name_value in test_dtypes:
                test_embedding_ordering(index_count, vocabulary_size, embedding_dim, dtype_name_value)

        if (round_index + 1) % 10 == 0:
            print(f"  completed {round_index + 1}/{rounds} rounds")

    print()

    print("\033[92mEmbedding synchronized-vs-async ordering test passed!\033[0m")

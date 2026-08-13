import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, parent_dir)

sys.path.insert(0, os.path.join(parent_dir, "test"))


import torch

import llaisys

from llaisys.libllaisys import DeviceType
from llaisys.triton import execution_context
from llaisys.triton.ops import add as triton_add
from llaisys.triton.ops import self_attention as triton_self_attention

from test_utils import check_equal, random_tensor


# ============================================================
# PyTorch Self-Attention reference
# ============================================================


def torch_self_attention(attn_val, query, key, value, scale):
    query = query.transpose(-2, -3)

    key = key.transpose(-2, -3)

    value = value.transpose(-2, -3)

    query_length = query.size(-2)

    key_length = key.size(-2)

    # ========================================================
    # Bottom-right causal mask
    # ========================================================

    attn_bias = torch.zeros(query_length, key_length, dtype=query.dtype, device=query.device)

    causal_mask = torch.ones(query_length, key_length, dtype=torch.bool, device=query.device).tril(
        diagonal=(key_length - query_length)
    )

    attn_bias.masked_fill_(causal_mask.logical_not(), float("-inf"))

    # ========================================================
    # GQA
    # ========================================================

    group_size = query.size(-3) // key.size(-3)

    key = key.repeat_interleave(group_size, dim=-3)

    value = value.repeat_interleave(group_size, dim=-3)

    # ========================================================
    # Attention
    # ========================================================

    scores = query @ key.transpose(-2, -1)

    scores *= scale

    scores += attn_bias

    probabilities = torch.softmax(scores, dim=-1)

    result = probabilities @ value

    attn_val.copy_(result.transpose(-2, -3))


# ============================================================
# One ordering test case
# ============================================================


def run_case(qlen, kvlen, nh, nkvh, qk_dim, value_dim, dtype_name, atol, rtol, rounds=20):
    device_name = "nvidia"

    # ========================================================
    # Shared input data
    # ========================================================

    q, q_ = random_tensor((qlen, nh, qk_dim), dtype_name, device_name)

    q_delta, q_delta_ = random_tensor((qlen, nh, qk_dim), dtype_name, device_name, scale=0.01)

    k, k_ = random_tensor((kvlen, nkvh, qk_dim), dtype_name, device_name)

    v, v_ = random_tensor((kvlen, nkvh, value_dim), dtype_name, device_name)

    out_delta_1, out_delta_1_ = random_tensor((qlen, nh, value_dim), dtype_name, device_name, scale=0.01)

    out_delta_2, out_delta_2_ = random_tensor((qlen, nh, value_dim), dtype_name, device_name, scale=0.01)

    scale = 1.0 / (qk_dim**0.5)

    # ========================================================
    # PyTorch expected result
    #
    # Native Add
    #     ↓
    # Triton Self-Attention
    #     ↓
    # Triton Add
    #     ↓
    # Native Add
    # ========================================================

    torch_q = q + q_delta

    torch_attention = torch.empty((qlen, nh, value_dim), dtype=q.dtype, device=q.device)

    torch_self_attention(torch_attention, torch_q, k, v, scale)

    torch_tmp = torch_attention + out_delta_1

    torch_expected = torch_tmp + out_delta_2

    # ========================================================
    # Allocate synchronized-reference tensors
    # ========================================================

    _, q_sync = random_tensor((qlen, nh, qk_dim), dtype_name, device_name)

    _, attn_sync = random_tensor((qlen, nh, value_dim), dtype_name, device_name)

    _, tmp_sync = random_tensor((qlen, nh, value_dim), dtype_name, device_name)

    _, out_sync = random_tensor((qlen, nh, value_dim), dtype_name, device_name)

    # ========================================================
    # Allocate asynchronous-test tensors
    # ========================================================

    _, q_async = random_tensor((qlen, nh, qk_dim), dtype_name, device_name)

    _, attn_async = random_tensor((qlen, nh, value_dim), dtype_name, device_name)

    _, tmp_async = random_tensor((qlen, nh, value_dim), dtype_name, device_name)

    _, out_async = random_tensor((qlen, nh, value_dim), dtype_name, device_name)

    # ========================================================
    # Runtime
    # ========================================================

    runtime = llaisys.RuntimeAPI(DeviceType.NVIDIA)

    runtime.set_device(0)

    # ========================================================
    # Warmup
    # ========================================================

    with execution_context(DeviceType.NVIDIA, device_id=0):
        llaisys.Ops.add(q_async, q_, q_delta_)

        triton_self_attention(attn_async, q_async, k_, v_, scale)

        triton_add(tmp_async, attn_async, out_delta_1_)

        llaisys.Ops.add(out_async, tmp_async, out_delta_2_)

    runtime.device_synchronize()

    # ========================================================
    # Repeated ordering verification
    # ========================================================

    for round_index in range(rounds):
        # ====================================================
        # Reference
        #
        # Explicit synchronization after every operation.
        # ====================================================

        llaisys.Ops.add(q_sync, q_, q_delta_)

        runtime.device_synchronize()

        triton_self_attention(attn_sync, q_sync, k_, v_, scale)

        runtime.device_synchronize()

        triton_add(tmp_sync, attn_sync, out_delta_1_)

        runtime.device_synchronize()

        llaisys.Ops.add(out_sync, tmp_sync, out_delta_2_)

        runtime.device_synchronize()

        # ====================================================
        # Async
        #
        # NO synchronization between operators.
        #
        # All Native and Triton kernels must follow the same
        # LLAISYS Runtime CUDA stream.
        # ====================================================

        with execution_context(DeviceType.NVIDIA, device_id=0):
            llaisys.Ops.add(q_async, q_, q_delta_)

            triton_self_attention(attn_async, q_async, k_, v_, scale)

            triton_add(tmp_async, attn_async, out_delta_1_)

            llaisys.Ops.add(out_async, tmp_async, out_delta_2_)

        # ====================================================
        # One synchronization at end only.
        # ====================================================

        runtime.device_synchronize()

        # ====================================================
        # Verify synchronized chain
        # ====================================================

        assert check_equal(out_sync, torch_expected, atol=atol, rtol=rtol), (
            "Synchronized Self-Attention chain failed: "
            f"qlen={qlen}, "
            f"kvlen={kvlen}, "
            f"nh={nh}, "
            f"nkvh={nkvh}, "
            f"qk_dim={qk_dim}, "
            f"value_dim={value_dim}, "
            f"dtype={dtype_name}, "
            f"round={round_index}"
        )

        # ====================================================
        # Verify asynchronous chain
        #
        # This is the important test:
        #
        # Native Add
        #     ↓
        # Triton Attention
        #     ↓
        # Triton Add
        #     ↓
        # Native Add
        #
        # must preserve dependency ordering without explicit
        # intermediate synchronization.
        # ====================================================

        assert check_equal(out_async, torch_expected, atol=atol, rtol=rtol), (
            "Asynchronous Self-Attention chain failed: "
            f"qlen={qlen}, "
            f"kvlen={kvlen}, "
            f"nh={nh}, "
            f"nkvh={nkvh}, "
            f"qk_dim={qk_dim}, "
            f"value_dim={value_dim}, "
            f"dtype={dtype_name}, "
            f"round={round_index}"
        )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("Testing Self-Attention mixed Native/Triton same-stream ordering")

    print()

    print("Reference:")

    print("  Native Add -> sync -> Triton Self-Attention -> sync -> Triton Add -> sync -> Native Add -> sync")

    print()

    print("Test:")

    print("  Native Add -> Triton Self-Attention -> Triton Add -> Native Add")

    print("  with NO intermediate synchronization")

    print()

    # ========================================================
    # Representative stream-ordering workloads
    #
    # We do NOT need to repeat every formal correctness shape.
    #
    # Ordering tests dependency semantics, not exhaustive
    # mathematical correctness.
    # ========================================================

    test_shapes = [
        # Tiny MHA.
        (2, 2, 1, 1, 4, 4),
        # GQA + prefix KV.
        (5, 11, 4, 2, 8, 8),
        # Decode-like Qwen workload.
        (1, 513, 12, 2, 128, 128),
        # Irregular Q/K and V dimensions.
        (17, 65, 4, 2, 31, 37),
    ]

    test_dtype_prec = [("f32", 1e-5, 1e-5), ("f16", 1e-3, 1e-3), ("bf16", 1e-2, 1e-2)]

    total_cases = len(test_shapes) * len(test_dtype_prec)

    completed_cases = 0

    for shape in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            run_case(*shape, dtype_name=dtype_name, atol=atol, rtol=rtol, rounds=20)

            completed_cases += 1

            print(f"  completed {completed_cases}/{total_cases} cases")

    print()

    print("\033[92mSelf-Attention synchronized-vs-async ordering test passed!\033[0m")

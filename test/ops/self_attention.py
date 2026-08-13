import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, parent_dir)


import torch

import llaisys

from llaisys.triton.ops import self_attention as triton_self_attention

from test_utils import benchmark, check_equal, random_tensor


# ============================================================
# PyTorch reference
# ============================================================


def torch_self_attention(attn_val, query, key, value, scale):
    # ========================================================
    # LLAISYS:
    #
    #     [S, H, D]
    #
    # PyTorch reference:
    #
    #     [H, S, D]
    # ========================================================

    query = query.transpose(-2, -3)

    key = key.transpose(-2, -3)

    value = value.transpose(-2, -3)

    query_length = query.size(-2)

    key_length = key.size(-2)

    # ========================================================
    # Bottom-right causal mask
    #
    # diagonal:
    #
    #     key_length - query_length
    #
    # This matches prefix-KV semantics.
    # ========================================================

    attn_bias = torch.zeros(query_length, key_length, dtype=query.dtype, device=query.device)

    temp_mask = torch.ones(query_length, key_length, dtype=torch.bool, device=query.device).tril(
        diagonal=(key_length - query_length)
    )

    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    # ========================================================
    # GQA expansion
    # ========================================================

    group_size = query.size(-3) // key.size(-3)

    key = key.repeat_interleave(group_size, -3)

    value = value.repeat_interleave(group_size, -3)

    # ========================================================
    # QK^T
    # ========================================================

    attn_weight = query @ key.transpose(-2, -1)

    attn_weight *= scale

    # ========================================================
    # Causal mask + Softmax
    # ========================================================

    attn_weight += attn_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)

    # ========================================================
    # Softmax(QK^T) @ V
    # ========================================================

    result = attn_weight @ value

    attn_val.copy_(result.transpose(-2, -3))


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_self_attention(attn_val, q, k, v, scale, backend):
    if backend == "native":
        llaisys.Ops.self_attention(attn_val, q, k, v, scale)

        return

    if backend == "triton":
        triton_self_attention(attn_val, q, k, v, scale)

        return

    raise ValueError(f"Unsupported Self-Attention backend: {backend}")


# ============================================================
# One correctness case
# ============================================================


def test_op_self_attention(
    qlen,
    kvlen,
    nh,
    nkvh,
    qk_dim,
    value_dim,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    backend="native",
    profile=False,
):
    print(
        f"   qlen={qlen} "
        f"kvlen={kvlen} "
        f"nh={nh} "
        f"nkvh={nkvh} "
        f"qk_dim={qk_dim} "
        f"value_dim={value_dim} "
        f"dtype <{dtype_name}> "
        f"backend <{backend}>"
    )

    # ========================================================
    # Q
    # ========================================================

    q, q_ = random_tensor((qlen, nh, qk_dim), dtype_name, device_name)

    # ========================================================
    # K
    # ========================================================

    k, k_ = random_tensor((kvlen, nkvh, qk_dim), dtype_name, device_name)

    # ========================================================
    # V
    # ========================================================

    v, v_ = random_tensor((kvlen, nkvh, value_dim), dtype_name, device_name)

    # ========================================================
    # Scale
    # ========================================================

    scale = 1.0 / (qk_dim**0.5)

    # ========================================================
    # Output
    # ========================================================

    attn_val, attn_val_ = random_tensor((qlen, nh, value_dim), dtype_name, device_name)

    # ========================================================
    # PyTorch reference
    # ========================================================

    torch_self_attention(attn_val, q, k, v, scale)

    # ========================================================
    # LLAISYS
    # ========================================================

    run_llaisys_self_attention(attn_val_, q_, k_, v_, scale, backend)

    # ========================================================
    # Correctness
    #
    # Keep existing tolerance contract.
    #
    # Do NOT widen tolerance yet if Triton fails.
    # ========================================================

    assert check_equal(attn_val_, attn_val, atol=atol, rtol=rtol), (
        f"Self-Attention mismatch: "
        f"qlen={qlen}, "
        f"kvlen={kvlen}, "
        f"nh={nh}, "
        f"nkvh={nkvh}, "
        f"qk_dim={qk_dim}, "
        f"value_dim={value_dim}, "
        f"dtype={dtype_name}, "
        f"backend={backend}"
    )

    # ========================================================
    # Optional diagnostic profile
    # ========================================================

    if profile:
        benchmark(
            lambda: torch_self_attention(attn_val, q, k, v, scale),
            lambda: run_llaisys_self_attention(attn_val_, q_, k_, v_, scale, backend),
            device_name,
        )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)

    parser.add_argument("--backend", default="native", choices=["native", "triton"], type=str)

    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()

    if args.backend == "triton" and args.device != "nvidia":
        raise ValueError("Triton Self-Attention currently supports NVIDIA only")

    # ========================================================
    # Correctness shapes
    #
    # tuple:
    #
    #     qlen
    #     kvlen
    #     nh
    #     nkvh
    #     qk_dim
    #     value_dim
    # ========================================================

    test_shapes = [
        # ====================================================
        # Original tiny test
        # ====================================================
        (2, 2, 1, 1, 4, 4),
        # ====================================================
        # Original GQA + prefix-KV test
        #
        # group_size = 2
        # prefix     = 6
        # ====================================================
        (5, 11, 4, 2, 8, 8),
        # ====================================================
        # Decode-like
        #
        # one new query token
        # long existing KV cache
        #
        # Qwen-like:
        #
        # 12 Q heads
        # 2 KV heads
        # head dim 128
        # ====================================================
        (1, 513, 12, 2, 128, 128),
        # ====================================================
        # Irregular dimensions
        #
        # Q/K dimension crosses BLOCK_D=32 boundary:
        #
        #     31
        #
        # V dimension tests BLOCK_V tail:
        #
        #     37
        #
        # Also tests:
        #
        #     qk_dim != value_dim
        # ====================================================
        (17, 65, 4, 2, 31, 37),
        # ====================================================
        # Prefill-like + prefix KV
        # ====================================================
        (64, 128, 12, 2, 128, 128),
    ]

    # ========================================================
    # Existing numerical contract
    # ========================================================

    test_dtype_prec = [("f32", 1e-5, 1e-5), ("f16", 1e-3, 1e-3), ("bf16", 1e-2, 1e-2)]

    print(f"Testing Ops.self_attention on {args.device} with {args.backend} backend")

    for shape in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_self_attention(
                *shape,
                dtype_name=dtype_name,
                atol=atol,
                rtol=rtol,
                device_name=args.device,
                backend=args.backend,
                profile=args.profile,
            )

    print()

    print("\033[92mTest passed!\033[0m")

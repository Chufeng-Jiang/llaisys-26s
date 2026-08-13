import argparse
import os
import sys

import torch


# ============================================================
# Project paths
# ============================================================

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, parent_dir)


# ============================================================
# Imports
# ============================================================

import llaisys

from llaisys.triton.ops import rope as triton_rope

from test_utils import arrange_tensor, benchmark_llaisys, check_equal, random_tensor


# ============================================================
# PyTorch reference
# ============================================================


def torch_rope(y: torch.Tensor, x: torch.Tensor, pos_ids: torch.Tensor, theta: float):
    assert y.dim() == 3

    seq_len, n_heads, head_dim = y.shape

    assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

    # ========================================================
    # Split dimension into paired halves
    # ========================================================

    half_dim = head_dim // 2

    x_a = x[..., :half_dim]

    x_b = x[..., half_dim:]

    # ========================================================
    # Position IDs in FP32
    # ========================================================

    positions = pos_ids.to(torch.float32).unsqueeze(1)

    # ========================================================
    # Match LLAISYS RoPE angle semantics
    #
    # exponent:
    #
    #     2 * i / d
    #
    # denominator:
    #
    #     theta ** exponent
    #
    # angle:
    #
    #     position / denominator
    # ========================================================

    pair_index = torch.arange(0, half_dim, dtype=torch.float32, device=y.device)

    exponent = 2.0 * pair_index / head_dim

    denominator = theta**exponent

    freqs = positions / denominator

    sine = freqs.sin().unsqueeze(1)

    cosine = freqs.cos().unsqueeze(1)

    # ========================================================
    # Rotation
    # ========================================================

    y[..., :half_dim] = x_a * cosine - x_b * sine

    y[..., half_dim:] = x_b * cosine + x_a * sine


# ============================================================
# One correctness case
# ============================================================


def test_op_rope(
    shape, start_end, dtype_name="f32", atol=1e-5, rtol=1e-5, device_name="cpu", backend="native", profile=False
):
    print(f"   shape {shape} range {start_end} dtype <{dtype_name}> device <{device_name}> backend <{backend}>")

    # ========================================================
    # Input
    # ========================================================

    x_ref, x = random_tensor(shape, dtype_name, device_name)

    # ========================================================
    # Position IDs
    # ========================================================

    pos_ref, pos = arrange_tensor(start_end[0], start_end[1], device_name)

    # ========================================================
    # Theta
    # ========================================================

    theta = 10000.0

    # ========================================================
    # Output
    # ========================================================

    y_ref, y = random_tensor(shape, dtype_name, device_name)

    # ========================================================
    # PyTorch correctness reference
    # ========================================================

    torch_rope(y_ref, x_ref, pos_ref, theta)

    # ========================================================
    # LLAISYS
    # ========================================================

    if backend == "native":
        llaisys.Ops.rope(y, x, pos, theta)

    elif backend == "triton":
        if device_name != "nvidia":
            raise ValueError("The Triton RoPE backend currently supports NVIDIA only")

        triton_rope(y, x, pos, theta)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # ========================================================
    # Correctness
    # ========================================================

    assert check_equal(y, y_ref, atol=atol, rtol=rtol)

    # ========================================================
    # Diagnostic profiling only
    #
    # Final paper metrics will later use the unified benchmark
    # harness.
    # ========================================================

    if profile:
        if backend == "native":
            benchmark_llaisys(
                lambda: llaisys.Ops.rope(y, x, pos, theta),
                device_name,
                label=(f"RoPE native shape={shape} range={start_end} dtype={dtype_name}"),
            )

        else:
            benchmark_llaisys(
                lambda: triton_rope(y, x, pos, theta),
                device_name,
                label=(f"RoPE Triton shape={shape} range={start_end} dtype={dtype_name}"),
            )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)

    parser.add_argument("--backend", default="native", choices=["native", "triton"], type=str)

    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()

    # ========================================================
    # RoPE diagnostic / formal cases
    # ========================================================
    #
    # Case 1
    #
    #     tiny sanity check
    #
    # Case 2
    #
    #     decode-like:
    #     seq=1
    #     heads=12
    #     d=128
    #     nonzero position
    #
    # Case 3
    #
    #     prefill-like:
    #     seq=512
    #     heads=12
    #     d=128
    #
    # Case 4
    #
    #     intermediate head dimension
    #
    # Case 5
    #
    #     large-head-dimension stress case
    #
    # This keeps the same numerical stress matrix already used
    # by the Native / MetaX RoPE work.
    # ========================================================

    test_shapes = [
        ((2, 1, 4), (0, 2)),
        ((1, 12, 128), (512, 513)),
        ((512, 12, 128), (512, 1024)),
        ((512, 12, 256), (512, 1024)),
        ((512, 4, 4096), (512, 1024)),
    ]

    # ========================================================
    # Existing evidence-based RoPE tolerances
    # ========================================================
    #
    # FP32:
    #
    #     atol = 2e-4
    #     rtol = 1e-4
    #
    # This tolerance was already established by the existing
    # RoPE numerical-portability stress work.
    #
    # Do NOT silently increase it for Triton.
    # ========================================================

    test_dtype_prec = [("f32", 2e-4, 1e-4), ("f16", 1e-3, 1e-3), ("bf16", 1e-2, 1e-2)]

    print(f"Testing Ops.rope on {args.device} with {args.backend} backend")

    # ========================================================
    # Run
    # ========================================================

    for shape, start_end in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_rope(shape, start_end, dtype_name, atol, rtol, args.device, args.backend, args.profile)

    print("\033[92mTest passed!\033[0m\n")

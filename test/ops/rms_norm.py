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

from llaisys.triton.ops import rms_norm as triton_rms_norm

from test_utils import benchmark_llaisys, check_equal, random_tensor


# ============================================================
# PyTorch reference
# ============================================================
#
# LLAISYS numerical contract:
#
#     input  -> FP32
#     weight -> FP32
#
#     FP32 square
#     FP32 reduction
#     FP32 normalization
#     FP32 scaling
#
#     final result -> output dtype
#
# ============================================================


def torch_rms_norm(out, x, weight, eps):
    x_f32 = x.float()

    weight_f32 = weight.float()

    square = x_f32 * x_f32

    mean_square = torch.mean(square, dim=-1, keepdim=True)

    inverse_rms = 1.0 / torch.sqrt(mean_square + eps)

    result = x_f32 * weight_f32 * inverse_rms

    out.copy_(result.to(out.dtype))


# ============================================================
# One correctness case
# ============================================================


def test_op_rms_norm(shape, dtype_name="f32", atol=1e-5, rtol=1e-5, device_name="cpu", backend="native", profile=False):
    print(f"   shape {shape} dtype <{dtype_name}> device <{device_name}> backend <{backend}>")

    # ========================================================
    # Input
    # ========================================================

    x_ref, x = random_tensor(shape, dtype_name, device_name)

    # ========================================================
    # Weight
    # ========================================================

    weight_ref, weight = random_tensor((shape[1],), dtype_name, device_name)

    # ========================================================
    # Epsilon
    # ========================================================

    eps = 1e-5

    # ========================================================
    # Output
    # ========================================================

    out_ref, out = random_tensor(shape, dtype_name, device_name)

    # ========================================================
    # PyTorch reference
    # ========================================================

    torch_rms_norm(out_ref, x_ref, weight_ref, eps)

    # ========================================================
    # LLAISYS
    # ========================================================

    if backend == "native":
        llaisys.Ops.rms_norm(out, x, weight, eps)

    elif backend == "triton":
        if device_name != "nvidia":
            raise ValueError("The Triton RMSNorm backend currently supports NVIDIA only")

        triton_rms_norm(out, x, weight, eps)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # ========================================================
    # Correctness
    # ========================================================

    assert check_equal(out, out_ref, atol=atol, rtol=rtol)

    # ========================================================
    # Optional profiling
    #
    # Diagnostic only.
    # Final metrics will later use the unified benchmark suite.
    # ========================================================

    if profile:
        if backend == "native":
            benchmark_llaisys(
                lambda: llaisys.Ops.rms_norm(out, x, weight, eps),
                device_name,
                label=(f"RMSNorm native shape={shape} dtype={dtype_name}"),
            )

        else:
            benchmark_llaisys(
                lambda: triton_rms_norm(out, x, weight, eps),
                device_name,
                label=(f"RMSNorm Triton shape={shape} dtype={dtype_name}"),
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

    print(f"Testing Ops.rms_norm on {args.device} with {args.backend} backend")

    # ========================================================
    # Shapes
    # ========================================================
    #
    # (1, 4)
    #     tiny sanity test
    #
    # (1, 1536)
    #     decode-shaped model width
    #
    # (16, 1536)
    #     small multi-row workload
    #
    # (1, 4096)
    #     wide decode row
    #
    # (512, 4095)
    #     irregular width / masked tail
    #
    # (512, 4096)
    #     large regular workload
    # ========================================================

    test_shapes = [(1, 4), (1, 1536), (16, 1536), (1, 4096), (512, 4095), (512, 4096)]

    # ========================================================
    # DType tolerances
    # ========================================================

    test_dtype_prec = [("f32", 1e-5, 1e-5), ("f16", 1e-3, 1e-3), ("bf16", 1e-2, 1e-2)]

    # ========================================================
    # Run
    # ========================================================

    for shape in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_rms_norm(shape, dtype_name, atol, rtol, args.device, args.backend, args.profile)

    print("\033[92mTest passed!\033[0m\n")

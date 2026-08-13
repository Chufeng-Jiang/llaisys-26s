import argparse
import os
import sys


# ============================================================
# Project paths
# ============================================================

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

python_dir = os.path.join(repo_root, "python")

test_dir = os.path.join(repo_root, "test")

sys.path.insert(0, python_dir)

sys.path.insert(0, test_dir)


# ============================================================
# Imports
# ============================================================

import torch

import llaisys

from llaisys.triton.ops import swiglu as triton_swiglu

from test_utils import check_equal, random_tensor


# ============================================================
# PyTorch reference
# ============================================================
#
# IMPORTANT:
#
# Match LLAISYS numerical semantics:
#
#     gate -> FP32
#     up   -> FP32
#
#     complete SwiGLU expression in FP32
#
#     final result -> output dtype
#
# Do NOT cast exp() back to FP16/BF16 early.
# ============================================================


def torch_swiglu(out, gate, up):
    gate_f32 = gate.float()
    up_f32 = up.float()

    result = up_f32 * gate_f32 / (1.0 + torch.exp(-gate_f32))

    out.copy_(result.to(out.dtype))


# ============================================================
# One correctness case
# ============================================================


def test_op_swiglu(shape, dtype_name, atol, rtol, device_name, backend):
    print(f"   shape {shape} dtype <{dtype_name}> device <{device_name}> backend <{backend}>")

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    gate_ref, gate = random_tensor(shape, dtype_name, device_name)

    up_ref, up = random_tensor(shape, dtype_name, device_name)

    out_ref, out = random_tensor(shape, dtype_name, device_name)

    # --------------------------------------------------------
    # PyTorch reference
    # --------------------------------------------------------

    torch_swiglu(out_ref, gate_ref, up_ref)

    # --------------------------------------------------------
    # LLAISYS implementation
    # --------------------------------------------------------

    if backend == "native":
        llaisys.Ops.swiglu(out, gate, up)

    elif backend == "triton":
        if device_name != "nvidia":
            raise ValueError("The Triton SwiGLU backend currently supports NVIDIA only")

        triton_swiglu(out, gate, up)

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    assert check_equal(out, out_ref, atol=atol, rtol=rtol)


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "metax"], type=str)

    parser.add_argument("--backend", default="native", choices=["native", "triton"], type=str)

    args = parser.parse_args()

    print(f"Testing Ops.swiglu on {args.device} with {args.backend} backend")

    # ========================================================
    # Shapes
    # ========================================================
    #
    # Include:
    #
    #     tiny
    #     irregular / masked tail
    #     large
    #
    # (33, 65) is especially important because numel=2145
    # is not divisible by BLOCK_SIZE=256.
    # ========================================================

    test_shapes = [(2, 3), (33, 65), (512, 4096)]

    # ========================================================
    # DTypes / tolerances
    # ========================================================

    test_dtypes = [("f32", 1e-5, 1e-5), ("f16", 1e-3, 1e-3), ("bf16", 1e-2, 1e-2)]

    # ========================================================
    # Run
    # ========================================================

    for shape in test_shapes:
        for dtype_name, atol, rtol in test_dtypes:
            test_op_swiglu(shape, dtype_name, atol, rtol, args.device, args.backend)

    print("\033[92mTest passed!\033[0m")

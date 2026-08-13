import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, parent_dir)


import torch

import llaisys

from llaisys.triton.ops import linear as triton_linear

from test_utils import benchmark, check_equal, random_tensor


# ============================================================
# PyTorch reference
# ============================================================


def torch_linear(out, x, w, bias):
    torch.nn.functional.linear(x, w, bias, out=out)


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_linear(out, x, w, bias, backend):
    if backend == "native":
        llaisys.Ops.linear(out, x, w, bias)

        return

    if backend == "triton":
        triton_linear(out, x, w, bias)

        return

    raise ValueError(f"Unsupported Linear backend: {backend}")


# ============================================================
# One correctness case
# ============================================================


def test_op_linear(
    out_shape,
    x_shape,
    w_shape,
    use_bias=True,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    backend="native",
    profile=False,
):
    print(f"   out {out_shape}, x {x_shape}, w {w_shape}, bias {use_bias}, dtype <{dtype_name}> backend <{backend}>")

    # ========================================================
    # Input
    # ========================================================

    x, x_ = random_tensor(x_shape, dtype_name, device_name, scale=0.1)

    # ========================================================
    # Weight
    # ========================================================

    w, w_ = random_tensor(w_shape, dtype_name, device_name, scale=0.01)

    # ========================================================
    # Bias
    # ========================================================

    bias = None
    bias_ = None

    if use_bias:
        bias, bias_ = random_tensor((w_shape[0],), dtype_name, device_name)

    # ========================================================
    # Output
    # ========================================================

    out, out_ = random_tensor(out_shape, dtype_name, device_name)

    # ========================================================
    # PyTorch reference
    # ========================================================

    torch_linear(out, x, w, bias)

    # ========================================================
    # LLAISYS
    # ========================================================

    run_llaisys_linear(out_, x_, w_, bias_, backend)

    # ========================================================
    # Correctness
    #
    # DO NOT use strict equality.
    #
    # GEMM reduction ordering may differ between:
    #
    #     PyTorch / cuBLAS
    #     Native LLAISYS / cuBLAS
    #     Triton
    #
    # Keep the existing numerical contract unchanged.
    # ========================================================

    assert check_equal(out_, out, atol=atol, rtol=rtol), (
        f"Linear mismatch: "
        f"out={out_shape}, "
        f"x={x_shape}, "
        f"w={w_shape}, "
        f"bias={use_bias}, "
        f"dtype={dtype_name}, "
        f"device={device_name}, "
        f"backend={backend}"
    )

    # ========================================================
    # Optional diagnostic profile
    # ========================================================

    if profile:
        benchmark(
            lambda: torch_linear(out, x, w, bias), lambda: run_llaisys_linear(out_, x_, w_, bias_, backend), device_name
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

    if args.backend == "triton" and args.device != "nvidia":
        raise ValueError("Triton Linear currently supports NVIDIA only")

    # ========================================================
    # Correctness shapes
    #
    # tuple:
    #
    #     out
    #     x
    #     weight
    #     use_bias
    # ========================================================

    test_shapes = [
        # ====================================================
        # Tiny
        # ====================================================
        ((2, 3), (2, 4), (3, 4), True),
        ((2, 3), (2, 4), (3, 4), False),
        # ====================================================
        # BLOCK_K tail:
        #
        # K = 31
        # ====================================================
        ((3, 37), (3, 31), (37, 31), True),
        # ====================================================
        # Exact BLOCK_K:
        #
        # K = 32
        # ====================================================
        ((3, 37), (3, 32), (37, 32), False),
        # ====================================================
        # BLOCK_K + tail:
        #
        # K = 33
        #
        # Also exercises:
        #
        #     M tail
        #     N tail
        # ====================================================
        ((17, 37), (17, 33), (37, 33), True),
        # ====================================================
        # K == 0
        #
        # With bias:
        #
        #     out = broadcast(bias)
        # ====================================================
        ((2, 3), (2, 0), (3, 0), True),
        # ====================================================
        # K == 0
        #
        # Without bias:
        #
        #     out = 0
        # ====================================================
        ((2, 3), (2, 0), (3, 0), False),
        # ====================================================
        # Decode-like
        #
        # M = 1
        # ====================================================
        ((1, 4096), (1, 4096), (4096, 4096), False),
        # ====================================================
        # Small-batch decode
        # ====================================================
        ((32, 4096), (32, 4096), (4096, 4096), True),
        # ====================================================
        # Prefill-like
        # ====================================================
        ((512, 4096), (512, 4096), (4096, 4096), True),
    ]

    test_dtype_prec = [("f32", 1e-5, 1e-5), ("f16", 1e-3, 1e-3), ("bf16", 1e-2, 1e-2)]

    print(f"Testing Ops.linear on {args.device} with {args.backend} backend")

    for shapes in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_linear(
                *shapes,
                dtype_name=dtype_name,
                atol=atol,
                rtol=rtol,
                device_name=args.device,
                backend=args.backend,
                profile=args.profile,
            )

    print()

    print("\033[92mTest passed!\033[0m")

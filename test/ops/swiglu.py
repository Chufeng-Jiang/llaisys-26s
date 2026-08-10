import sys
import os

parent_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
sys.path.insert(0, parent_dir)

import llaisys
import torch

from test_utils import (
    random_tensor,
    check_equal,
    benchmark_llaisys,
)


def torch_swiglu(out, gate, up):
    # Match the LLAISYS CUDA-compatible implementation:
    #
    #   1. convert gate/up to FP32
    #   2. evaluate SwiGLU in FP32
    #   3. cast only the final result back to output dtype
    #
    # Formula:
    #
    #   out = up * gate / (1 + exp(-gate))
    #
    gate_f32 = gate.float()
    up_f32 = up.float()

    result = (
        up_f32
        * gate_f32
        / (
            1.0
            + torch.exp(
                -gate_f32
            )
        )
    )

    out.copy_(
        result.to(
            out.dtype
        )
    )


def test_op_swiglu(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(
        f"   shape {shape} "
        f"dtype <{dtype_name}>"
    )

    gate, gate_ = random_tensor(
        shape,
        dtype_name,
        device_name,
    )

    up, up_ = random_tensor(
        shape,
        dtype_name,
        device_name,
    )

    out, out_ = random_tensor(
        shape,
        dtype_name,
        device_name,
    )

    # ========================================================
    # Correctness reference
    # ========================================================

    torch_swiglu(
        out,
        gate,
        up,
    )

    llaisys.Ops.swiglu(
        out_,
        gate_,
        up_,
    )

    assert check_equal(
        out_,
        out,
        atol=atol,
        rtol=rtol,
    )

    # ========================================================
    # LLAISYS profiling
    # ========================================================

    if profile:
        benchmark_llaisys(
            lambda: llaisys.Ops.swiglu(
                out_,
                gate_,
                up_,
            ),
            device_name,
            label=(
                f"SwiGLU "
                f"shape={shape} "
                f"dtype={dtype_name}"
            ),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cpu",
        choices=[
            "cpu",
            "nvidia",
            "metax",
        ],
        type=str,
    )

    parser.add_argument(
        "--profile",
        action="store_true",
    )

    args = parser.parse_args()

    test_shapes = [
        (2, 3),
        (512, 4096),
    ]

    test_dtype_prec = [
        # dtype, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(
        f"Testing Ops.swiglu "
        f"on {args.device}"
    )

    for shape in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_swiglu(
                shape,
                dtype_name,
                atol,
                rtol,
                args.device,
                args.profile,
            )

    print(
        "\033[92m"
        "Test passed!"
        "\033[0m\n"
    )
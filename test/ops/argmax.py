"""


from calendar import c
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark, zero_tensor


def torch_argmax(max_idx, max_val, vals):
    torch.max(vals, keepdim=True, dim=-1, out=(max_val, max_idx))


def test_op_argmax(
    shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")
    vals, vals_ = random_tensor(shape, dtype_name, device_name)
    max_idx, max_idx_ = zero_tensor((1,), "i64", device_name)
    max_val, max_val_ = zero_tensor((1,), dtype_name, device_name)

    torch_argmax(max_idx, max_val, vals)
    llaisys.Ops.argmax(max_idx_, max_val_, vals_)

    assert check_equal(max_val_, max_val, strict=True) or check_equal(
        max_idx_, max_idx, strict=True
    )

    if profile:
        benchmark(
            lambda: torch_argmax(max_idx, max_val, vals),
            lambda: llaisys.Ops.argmax(max_idx_, max_val_, vals_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    testShapes = [(4,), (4096,)]
    testDtype = ["f32", "f16", "bf16"]
    print(f"Testing Ops.argmax on {args.device}")
    for shape in testShapes:
        for dtype_name in testDtype:
            test_op_argmax(shape, dtype_name, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
    
    """
    
    
import os
import sys

parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, parent_dir)

import llaisys
import torch

from test_utils import benchmark, check_equal, random_tensor, zero_tensor


def torch_argmax(max_idx, max_val, vals):
    torch.max(
        vals,
        keepdim=True,
        dim=-1,
        out=(max_val, max_idx),
    )


def test_op_argmax(
    shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(f"   shape {shape} dtype <{dtype_name}>")

    vals, vals_ = random_tensor(
        shape,
        dtype_name,
        device_name,
    )

    max_idx, max_idx_ = zero_tensor(
        (1,),
        "i64",
        device_name,
    )

    max_val, max_val_ = zero_tensor(
        (1,),
        dtype_name,
        device_name,
    )

    torch_argmax(
        max_idx,
        max_val,
        vals,
    )

    llaisys.Ops.argmax(
        max_idx_,
        max_val_,
        vals_,
    )

    # Both the maximum value and its index must be correct.
    assert check_equal(
        max_val_,
        max_val,
        strict=True,
    ), (
        f"Argmax value mismatch: "
        f"shape={shape}, dtype={dtype_name}, device={device_name}"
    )

    assert check_equal(
        max_idx_,
        max_idx,
        strict=True,
    ), (
        f"Argmax index mismatch: "
        f"shape={shape}, dtype={dtype_name}, device={device_name}"
    )

    if profile:
        benchmark(
            lambda: torch_argmax(
                max_idx,
                max_val,
                vals,
            ),
            lambda: llaisys.Ops.argmax(
                max_idx_,
                max_val_,
                vals_,
            ),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "nvidia"],
        type=str,
    )

    parser.add_argument(
        "--profile",
        action="store_true",
    )

    args = parser.parse_args()

    test_shapes = [
        # Basic cases.
        (1,),
        (4,),

        # Warp boundaries.
        (31,),
        (32,),
        (33,),

        # Typical CUDA block boundaries.
        (255,),
        (256,),
        (257,),

        # CUDA single-block/multi-block path boundary.
        (4095,),
        (4096,),
        (4097,),

        # CPU OpenMP threshold boundary.
        (32767,),
        (32768,),
        (32769,),

        # Large grid-stride-loop case.
        (512 * 4096,),
    ]

    test_dtypes = [
        "f32",
        "f16",
        "bf16",
    ]

    print(f"Testing Ops.argmax on {args.device}")

    for shape in test_shapes:
        for dtype_name in test_dtypes:
            test_op_argmax(
                shape,
                dtype_name,
                args.device,
                args.profile,
            )

    print("\033[92mTest passed!\033[0m\n")
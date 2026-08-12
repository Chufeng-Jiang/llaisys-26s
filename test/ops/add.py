import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
from llaisys.triton.ops import add as triton_add
from test_utils import benchmark_llaisys, check_equal, random_tensor

import llaisys


def torch_add(ans, a, b):
    torch.add(a, b, out=ans)


def llaisys_add(c, a, b, backend_name):
    """
    Dispatch Add to the selected implementation backend.

    backend_name:
        native  -> existing LLAISYS C++/CUDA/MetaX implementation
        triton  -> Triton implementation
    """
    if backend_name == "native":
        llaisys.Ops.add(c, a, b)

    elif backend_name == "triton":
        triton_add(c, a, b)

    else:
        raise ValueError(f"Unsupported backend: {backend_name}")


def test_op_add(shape, dtype_name="f32", atol=1e-5, rtol=1e-5, device_name="cpu", backend_name="native", profile=False):
    print(f"   shape {shape} dtype <{dtype_name}> device <{device_name}> backend <{backend_name}>")

    # PyTorch reference + LLAISYS tensors.
    a, a_ = random_tensor(shape, dtype_name, device_name)

    b, b_ = random_tensor(shape, dtype_name, device_name)

    c, c_ = random_tensor(shape, dtype_name, device_name)

    # Reference result.
    torch_add(c, a, b)

    # LLAISYS implementation.
    llaisys_add(c_, a_, b_, backend_name)

    # Compare Triton/native result with PyTorch.
    assert check_equal(c_, c, atol=atol, rtol=rtol)

    if profile:
        benchmark_llaisys(
            lambda: llaisys_add(c_, a_, b_, backend_name),
            device_name,
            label=(f"Add shape={shape} dtype={dtype_name} backend={backend_name}"),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)

    parser.add_argument("--backend", default="native", choices=["native", "triton"], type=str)

    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()

    # Triton bring-up currently only supports NVIDIA.
    if args.backend == "triton" and args.device != "nvidia":
        raise ValueError("Triton backend currently only supports NVIDIA")

    test_shapes = [(2, 3), (33, 65), (512, 4096)]

    test_dtype_prec = [
        # dtype, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-3, 1e-3),
    ]

    print(f"Testing Ops.add on {args.device} with {args.backend} backend")

    for shape in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_add(shape, dtype_name, atol, rtol, args.device, args.backend, args.profile)

    print("\033[92mTest passed!\033[0m\n")

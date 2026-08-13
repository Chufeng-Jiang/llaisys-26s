import argparse
import os
import sys
from ctypes import c_void_p

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys

from test_utils import benchmark, check_equal, random_tensor, reference_torch_device, zero_tensor


TORCH_DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


def torch_argmax(max_idx, max_val, vals):
    torch.max(vals, keepdim=True, dim=-1, out=(max_val, max_idx))


def run_llaisys_argmax(max_idx, max_val, vals, backend):
    if backend == "native":
        llaisys.Ops.argmax(max_idx, max_val, vals)
        return

    if backend == "triton":
        from llaisys.triton.ops import argmax as triton_argmax

        triton_argmax(max_idx, max_val, vals)
        return

    raise ValueError(f"Unsupported Argmax backend: {backend}")


def test_op_argmax(shape, dtype_name="f32", device_name="cpu", backend="native", profile=False):
    print(f"   random shape {shape} dtype <{dtype_name}> device <{device_name}> backend <{backend}>")

    vals, vals_ = random_tensor(shape, dtype_name, device_name)

    max_idx, max_idx_ = zero_tensor((1,), "i64", device_name)

    max_val, max_val_ = zero_tensor((1,), dtype_name, device_name)

    torch_argmax(max_idx, max_val, vals)

    run_llaisys_argmax(max_idx_, max_val_, vals_, backend)

    assert check_equal(max_val_, max_val, strict=True), (
        f"Argmax value mismatch: shape={shape}, dtype={dtype_name}, device={device_name}, backend={backend}"
    )

    assert check_equal(max_idx_, max_idx, strict=True), (
        f"Argmax index mismatch: shape={shape}, dtype={dtype_name}, device={device_name}, backend={backend}"
    )

    if profile:
        benchmark(
            lambda: torch_argmax(max_idx, max_val, vals),
            lambda: run_llaisys_argmax(max_idx_, max_val_, vals_, backend),
            device_name,
        )


def test_semantic_case(name, values, expected_index, dtype_name, device_name, backend):
    print(f"   semantic {name} dtype <{dtype_name}> backend <{backend}>")

    torch_dtype = TORCH_DTYPES[dtype_name]

    # ========================================================
    # Exact host input
    #
    # Keep one CPU tensor because Tensor.load() takes host data.
    # ========================================================

    vals_host = torch.tensor(values, dtype=torch_dtype, device="cpu").contiguous()

    shape = tuple(vals_host.shape)

    # ========================================================
    # LLAISYS input
    # ========================================================

    _, vals_ = random_tensor(shape, dtype_name, device_name)

    vals_.load(c_void_p(vals_host.data_ptr()))

    # ========================================================
    # PyTorch reference device
    #
    # CPU:
    #     CPU
    #
    # NVIDIA:
    #     CUDA
    #
    # MetaX:
    #     CPU reference
    #
    # This is the important fix.
    # ========================================================

    reference_device = reference_torch_device(device_name)

    vals_reference = vals_host.to(reference_device)

    max_idx = torch.zeros((1,), dtype=torch.int64, device=reference_device)

    max_val = torch.zeros((1,), dtype=torch_dtype, device=reference_device)

    # ========================================================
    # LLAISYS outputs
    # ========================================================

    _, max_idx_ = zero_tensor((1,), "i64", device_name)

    _, max_val_ = zero_tensor((1,), dtype_name, device_name)

    # ========================================================
    # PyTorch reference
    # ========================================================

    torch_argmax(max_idx, max_val, vals_reference)

    actual_reference_index = int(max_idx.item())

    assert actual_reference_index == expected_index, (
        f"Invalid semantic test: "
        f"name={name}, "
        f"dtype={dtype_name}, "
        f"expected={expected_index}, "
        f"PyTorch={actual_reference_index}"
    )

    # ========================================================
    # LLAISYS
    # ========================================================

    run_llaisys_argmax(max_idx_, max_val_, vals_, backend)

    # ========================================================
    # Index must always match exactly.
    # ========================================================

    assert check_equal(max_idx_, max_idx, strict=True), (
        f"Argmax semantic index mismatch: case={name}, dtype={dtype_name}, device={device_name}, backend={backend}"
    )

    # ========================================================
    # NaN != NaN under ordinary strict equality.
    #
    # Therefore NaN cases verify the selected index.
    # Non-NaN cases also verify max value exactly.
    # ========================================================

    reference_is_nan = bool(torch.isnan(max_val).item())

    if reference_is_nan:
        print(f"      NaN reference selected at index {actual_reference_index}")
    else:
        assert check_equal(max_val_, max_val, strict=True), (
            f"Argmax semantic value mismatch: case={name}, dtype={dtype_name}, device={device_name}, backend={backend}"
        )


def run_semantic_tests(device_name, dtype_name, backend):
    cases = [
        ("single_element", [5.0], 0),
        ("normal", [1.0, 4.0, 2.0, 3.0], 1),
        ("maximum_first", [9.0, 1.0, 2.0, 3.0], 0),
        ("maximum_last", [1.0, 2.0, 3.0, 9.0], 3),
        ("duplicate_maximum", [1.0, 9.0, 3.0, 9.0], 1),
        ("all_negative", [-9.0, -3.0, -7.0], 1),
        ("positive_infinity", [1.0, float("inf"), 100.0], 1),
        ("negative_infinity", [float("-inf"), -2.0, -3.0], 1),
        ("wide_duplicate_maximum", [1.0, 2.0, 99.0, 4.0, 5.0, 6.0, 7.0, 99.0], 2),
        ("single_nan", [1.0, float("nan"), 100.0], 1),
        ("multiple_nan", [float("nan"), 1.0, float("nan")], 0),
        ("nan_after_numeric_max", [1000.0, 999.0, float("nan")], 2),
        ("nan_before_numeric_max", [float("nan"), 999.0, 1000.0], 0),
        ("wide_multiple_nan", [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, float("nan"), 1000.0], 2),
        ("cross_block_duplicate_maximum", [99.0 if i in (100, 1100) else float(i % 17) for i in range(1200)], 100),
        ("cross_block_multiple_nan", [float("nan") if i in (123, 1123) else float(i % 23) for i in range(1200)], 123),
    ]

    for name, values, expected_index in cases:
        test_semantic_case(name, values, expected_index, dtype_name, device_name, backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)

    parser.add_argument("--backend", default="native", choices=["native", "triton"], type=str)

    parser.add_argument("--profile", action="store_true")

    parser.add_argument("--skip-random", action="store_true")

    parser.add_argument("--skip-semantic", action="store_true")

    args = parser.parse_args()

    if args.backend == "triton" and args.device != "nvidia":
        raise ValueError("Triton Argmax currently supports NVIDIA only")

    print(f"Testing Ops.argmax on {args.device} with {args.backend} backend")

    correctness_shapes = [
        (1,),
        (4,),
        (31,),
        (32,),
        (33,),
        (63,),
        (64,),
        (65,),
        (255,),
        (256,),
        (257,),
        (1023,),
        (1024,),
        (1025,),
        (2048,),
        (4095,),
        (4096,),
        (4097,),
        (32000,),
        (151936,),
        (512 * 4096,),
    ]

    test_dtypes = ["f32", "f16", "bf16"]

    # ========================================================
    # Random differential correctness
    # ========================================================

    if not args.skip_random:
        print()
        print("=== Random differential correctness ===")

        for shape in correctness_shapes:
            for dtype_name in test_dtypes:
                test_op_argmax(shape, dtype_name, args.device, args.backend, profile=False)

    # ========================================================
    # Deterministic semantics
    # ========================================================

    if not args.skip_semantic:
        print()
        print("=== Deterministic semantic correctness ===")

        for dtype_name in test_dtypes:
            run_semantic_tests(args.device, dtype_name, args.backend)

    # ========================================================
    # Diagnostic performance
    # ========================================================

    if args.profile:
        print()
        print("=== Diagnostic performance benchmark ===")

        profile_shapes = [(4,), (256,), (1024,), (4096,), (32000,), (151936,), (512 * 4096,)]

        for shape in profile_shapes:
            for dtype_name in test_dtypes:
                test_op_argmax(shape, dtype_name, args.device, args.backend, profile=True)

    print()
    print("\033[92mTest passed!\033[0m")

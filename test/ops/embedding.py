import argparse
import os
import sys
from ctypes import c_void_p

import torch


# ============================================================
# Repository paths
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")

# Always prefer the current repository implementation over an
# installed site-packages copy of llaisys.
sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, TEST_DIR)


import llaisys

from llaisys.triton.ops import embedding as triton_embedding

from test_utils import benchmark_llaisys, check_equal, random_int_tensor, random_tensor


# ============================================================
# PyTorch reference
# ============================================================


def torch_embedding(out, idx, embd):
    out[:] = embd[idx]


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_embedding(out, idx, embd, backend):
    if backend == "native":
        llaisys.Ops.embedding(out, idx, embd)
        return

    if backend == "triton":
        triton_embedding(out, idx, embd)
        return

    raise ValueError(f"Unsupported Embedding backend: {backend}")


# ============================================================
# Random valid-index correctness
# ============================================================


def test_op_embedding(
    idx_shape,
    embd_shape,
    dtype_name="f32",
    device_name="cpu",
    backend="native",
    profile=False,
):
    print(
        f"   random "
        f"idx_shape {idx_shape} "
        f"embd_shape {embd_shape} "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    # ========================================================
    # Embedding table
    # ========================================================

    embd, embd_ = random_tensor(
        embd_shape,
        dtype_name,
        device_name,
    )

    # ========================================================
    # Valid Int64 indices
    # ========================================================

    idx, idx_ = random_int_tensor(
        idx_shape,
        device_name,
        high=embd_shape[0],
    )

    # ========================================================
    # Output
    # ========================================================

    out_shape = (
        idx_shape[0],
        embd_shape[1],
    )

    out, out_ = random_tensor(
        out_shape,
        dtype_name,
        device_name,
    )

    # ========================================================
    # PyTorch reference
    # ========================================================

    torch_embedding(
        out,
        idx,
        embd,
    )

    # ========================================================
    # LLAISYS
    # ========================================================

    run_llaisys_embedding(
        out_,
        idx_,
        embd_,
        backend,
    )

    # ========================================================
    # Exact correctness
    # ========================================================

    assert check_equal(
        out_,
        out,
        strict=True,
    ), (
        f"Embedding mismatch: "
        f"idx_shape={idx_shape}, "
        f"embd_shape={embd_shape}, "
        f"dtype={dtype_name}, "
        f"device={device_name}, "
        f"backend={backend}"
    )

    # ========================================================
    # Optional diagnostic profiling
    # ========================================================

    if profile:
        benchmark_llaisys(
            lambda: run_llaisys_embedding(
                out_,
                idx_,
                embd_,
                backend,
            ),
            device_name,
            label=(
                f"Embedding "
                f"idx_shape={idx_shape} "
                f"embd_shape={embd_shape} "
                f"dtype={dtype_name} "
                f"backend={backend}"
            ),
        )


# ============================================================
# Deterministic valid-index semantic case
# ============================================================


def test_semantic_case(
    name,
    indices,
    embd_shape,
    dtype_name,
    device_name,
    backend,
):
    print(
        f"   semantic {name} "
        f"idx_count <{len(indices)}> "
        f"embd_shape {embd_shape} "
        f"dtype <{dtype_name}> "
        f"backend <{backend}>"
    )

    # ========================================================
    # Weight
    # ========================================================

    embd, embd_ = random_tensor(
        embd_shape,
        dtype_name,
        device_name,
    )

    # ========================================================
    # Exact host index tensor
    # ========================================================

    idx_host = torch.tensor(
        indices,
        dtype=torch.int64,
        device="cpu",
    ).contiguous()

    idx_shape = (len(indices),)

    # ========================================================
    # Allocate LLAISYS index tensor
    # ========================================================

    _, idx_ = random_int_tensor(
        idx_shape,
        device_name,
        high=max(embd_shape[0], 1),
    )

    idx_.load(
        c_void_p(idx_host.data_ptr())
    )

    # ========================================================
    # PyTorch reference index
    # ========================================================

    idx_reference = torch.tensor(
        indices,
        dtype=torch.int64,
        device=embd.device,
    )

    # ========================================================
    # Output
    # ========================================================

    out_shape = (
        len(indices),
        embd_shape[1],
    )

    out, out_ = random_tensor(
        out_shape,
        dtype_name,
        device_name,
    )

    # ========================================================
    # PyTorch reference
    # ========================================================

    torch_embedding(
        out,
        idx_reference,
        embd,
    )

    # ========================================================
    # LLAISYS
    # ========================================================

    run_llaisys_embedding(
        out_,
        idx_,
        embd_,
        backend,
    )

    # ========================================================
    # Exact comparison
    # ========================================================

    assert check_equal(
        out_,
        out,
        strict=True,
    ), (
        f"Embedding semantic mismatch: "
        f"case={name}, "
        f"dtype={dtype_name}, "
        f"device={device_name}, "
        f"backend={backend}"
    )


# ============================================================
# Deterministic valid semantic suite
# ============================================================


def run_semantic_tests(
    device_name,
    dtype_name,
    backend,
):
    # D=7 intentionally exercises a non-aligned tail.
    embd_shape = (4, 7)

    cases = [
        (
            "first_row",
            [0],
        ),
        (
            "last_row",
            [3],
        ),
        (
            "duplicate_index",
            [2, 2, 2],
        ),
        (
            "reverse_order",
            [3, 2, 1, 0],
        ),
        (
            "mixed_valid",
            [0, 3, 1, 3, 2, 0],
        ),
    ]

    for name, indices in cases:
        test_semantic_case(
            name,
            indices,
            embd_shape,
            dtype_name,
            device_name,
            backend,
        )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
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
        "--backend",
        default="native",
        choices=[
            "native",
            "triton",
        ],
        type=str,
    )

    parser.add_argument(
        "--profile",
        action="store_true",
    )

    parser.add_argument(
        "--skip-semantic",
        action="store_true",
    )

    args = parser.parse_args()

    if args.backend == "triton" and args.device != "nvidia":
        raise ValueError(
            "Triton Embedding currently supports NVIDIA only"
        )

    print(
        f"Testing Ops.embedding "
        f"on {args.device} "
        f"with {args.backend} backend"
    )

    print(
        f"Using llaisys from: {llaisys.__file__}"
    )

    # ========================================================
    # Correctness matrix
    #
    # D=127 / 128 / 129:
    #     boundary around 128-column tiles
    #
    # D=4095 / 4096:
    #     large irregular/aligned row width
    # ========================================================

    test_shapes = [
        (
            (1,),
            (2, 3),
        ),
        (
            (7,),
            (17, 127),
        ),
        (
            (8,),
            (17, 128),
        ),
        (
            (9,),
            (17, 129),
        ),
        (
            (33,),
            (257, 4095),
        ),
        (
            (50,),
            (512, 4096),
        ),
    ]

    test_dtypes = [
        "f32",
        "f16",
        "bf16",
    ]

    # ========================================================
    # Random valid-index correctness
    # ========================================================

    print()
    print(
        "=== Random valid-index correctness ==="
    )

    for idx_shape, embd_shape in test_shapes:
        for dtype_name in test_dtypes:
            test_op_embedding(
                idx_shape,
                embd_shape,
                dtype_name,
                args.device,
                args.backend,
                profile=False,
            )

    # ========================================================
    # Deterministic valid-index semantics
    # ========================================================

    if not args.skip_semantic:
        print()
        print(
            "=== Deterministic valid-index semantics ==="
        )

        for dtype_name in test_dtypes:
            run_semantic_tests(
                args.device,
                dtype_name,
                args.backend,
            )

    # ========================================================
    # Optional performance diagnostic
    # ========================================================

    if args.profile:
        print()
        print(
            "=== Diagnostic performance benchmark ==="
        )

        profile_shapes = [
            (
                (1,),
                (2, 3),
            ),
            (
                (50,),
                (512, 4096),
            ),
        ]

        for idx_shape, embd_shape in profile_shapes:
            for dtype_name in test_dtypes:
                test_op_embedding(
                    idx_shape,
                    embd_shape,
                    dtype_name,
                    args.device,
                    args.backend,
                    profile=True,
                )

    print()
    print(
        "\033[92mTest passed!\033[0m"
    )
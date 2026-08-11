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

from test_utils import (
    random_int_tensor,
    random_tensor,
    check_equal,
    benchmark_llaisys,
)


def torch_embedding(
    out,
    idx,
    embd,
):
    out[:] = embd[idx]


def test_op_embedding(
    idx_shape,
    embd_shape,
    dtype_name="f32",
    device_name="cpu",
    profile=False,
):
    print(
        f"   idx_shape {idx_shape} " f"embd_shape {embd_shape} " f"dtype <{dtype_name}>"
    )

    # ========================================================
    # Prepare embedding table
    # ========================================================

    embd, embd_ = random_tensor(
        embd_shape,
        dtype_name,
        device_name,
    )

    # ========================================================
    # Prepare indices
    # ========================================================
    #
    # All generated indices are valid:
    #
    #     0 <= idx < vocabulary_size
    #
    # where:
    #
    #     vocabulary_size = embd_shape[0]
    #
    # ========================================================

    idx, idx_ = random_int_tensor(
        idx_shape,
        device_name,
        high=embd_shape[0],
    )

    # ========================================================
    # Prepare output
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
    # Correctness reference
    # ========================================================

    torch_embedding(
        out,
        idx,
        embd,
    )

    llaisys.Ops.embedding(
        out_,
        idx_,
        embd_,
    )

    # Embedding is a direct row gather/copy.
    #
    # There is no floating-point arithmetic in the operator,
    # so the result should be exactly equal for F32/F16/BF16.
    check_equal(
        out_,
        out,
        strict=True,
    )

    # ========================================================
    # LLAISYS profiling
    # ========================================================

    if profile:
        benchmark_llaisys(
            lambda: llaisys.Ops.embedding(
                out_,
                idx_,
                embd_,
            ),
            device_name,
            label=(
                f"Embedding "
                f"idx_shape={idx_shape} "
                f"embd_shape={embd_shape} "
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
        (
            (1,),
            (2, 3),
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

    print(f"Testing Ops.embedding " f"on {args.device}")

    for idx_shape, embd_shape in test_shapes:
        for dtype_name in test_dtypes:
            test_op_embedding(
                idx_shape,
                embd_shape,
                dtype_name,
                args.device,
                args.profile,
            )

    print("\033[92m" "Test passed!" "\033[0m\n")

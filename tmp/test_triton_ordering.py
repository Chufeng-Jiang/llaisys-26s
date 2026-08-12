import sys
from pathlib import Path

# ============================================================
# Project paths
# ============================================================

repo_root = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(repo_root / "python"),
)

sys.path.insert(
    0,
    str(repo_root / "test"),
)

# ============================================================
# Imports
# ============================================================

import llaisys

from llaisys.triton.ops import add as triton_add

from test_utils import (
    random_tensor,
    check_equal,
)


def test_ordering(
    shape,
    dtype_name,
    atol,
    rtol,
):
    a, a_ = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    b, b_ = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, tmp1_ = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, tmp2_ = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, out_ = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    # ========================================================
    # PyTorch reference
    #
    # tmp1 = a + b
    # tmp2 = tmp1 + b
    # out  = tmp2 + b
    #
    # Therefore:
    # out = a + 3 * b
    # ========================================================

    expected = a + b
    expected = expected + b
    expected = expected + b

    # ========================================================
    # LLAISYS execution
    #
    # IMPORTANT:
    # There must be NO synchronization between these three
    # operators. Their dependency must be guaranteed by stream
    # ordering.
    # ========================================================

    # Native CUDA:
    # tmp1 = a + b
    llaisys.Ops.add(
        tmp1_,
        a_,
        b_,
    )

    # Triton:
    # tmp2 = tmp1 + b
    triton_add(
        tmp2_,
        tmp1_,
        b_,
    )

    # Native CUDA:
    # out = tmp2 + b
    llaisys.Ops.add(
        out_,
        tmp2_,
        b_,
    )

    # ========================================================
    # Synchronization happens only after the complete
    # Native -> Triton -> Native dependency chain.
    # ========================================================

    assert check_equal(
        out_,
        expected,
        atol=atol,
        rtol=rtol,
    )


if __name__ == "__main__":
    test_cases = [
        ((2, 3), "f32", 1e-5, 1e-5),
        ((33, 65), "f32", 1e-5, 1e-5),
        ((512, 4096), "f32", 1e-5, 1e-5),
        ((33, 65), "f16", 1e-3, 1e-3),
        ((33, 65), "bf16", 1e-3, 1e-3),
    ]

    rounds = 100

    print(
        "Testing Native -> Triton -> Native "
        "stream ordering"
    )

    for round_idx in range(rounds):
        for shape, dtype_name, atol, rtol in test_cases:
            test_ordering(
                shape,
                dtype_name,
                atol,
                rtol,
            )

        if (round_idx + 1) % 10 == 0:
            print(
                f"  completed "
                f"{round_idx + 1}/{rounds} rounds"
            )

    print(
        "\033[92m"
        "Native -> Triton -> Native ordering test passed!"
        "\033[0m"
    )
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

from llaisys.libllaisys import DeviceType
from llaisys.triton import execution_context
from llaisys.triton.ops import add as triton_add
from test_utils import (
    check_equal,
    random_tensor,
)

import llaisys

# ============================================================
# One ordering test
# ============================================================


def test_execution_context_ordering(
    shape,
    dtype_name,
    atol,
    rtol,
):
    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    a_ref, a = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    b_ref, b = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, tmp1 = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, tmp2 = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, tmp3 = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, out = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    # ========================================================
    # Reference
    #
    # tmp1 = a + b
    # tmp2 = tmp1 + b
    # tmp3 = tmp2 + b
    # out  = tmp3 + b
    #
    # Therefore:
    #
    # out = a + 4b
    # ========================================================

    expected = a_ref + b_ref
    expected = expected + b_ref
    expected = expected + b_ref
    expected = expected + b_ref

    # ========================================================
    # LLAISYS execution
    #
    # IMPORTANT:
    #
    # There must be NO synchronization between any of these
    # operators.
    #
    # Native CUDA operators explicitly use Runtime::_stream.
    #
    # Triton operators should detect that an execution context
    # is already active and directly launch on the same stream.
    # ========================================================

    with execution_context(
        DeviceType.NVIDIA,
        device_id=0,
    ):
        # ----------------------------------------------------
        # Native CUDA
        #
        # tmp1 = a + b
        # ----------------------------------------------------

        llaisys.Ops.add(
            tmp1,
            a,
            b,
        )

        # ----------------------------------------------------
        # Triton
        #
        # tmp2 = tmp1 + b
        # ----------------------------------------------------

        triton_add(
            tmp2,
            tmp1,
            b,
        )

        # ----------------------------------------------------
        # Triton again
        #
        # tmp3 = tmp2 + b
        # ----------------------------------------------------

        triton_add(
            tmp3,
            tmp2,
            b,
        )

        # ----------------------------------------------------
        # Native CUDA again
        #
        # out = tmp3 + b
        # ----------------------------------------------------

        llaisys.Ops.add(
            out,
            tmp3,
            b,
        )

    # ========================================================
    # First synchronization occurs after the complete chain.
    #
    # check_equal performs the final copy/check.
    # ========================================================

    assert check_equal(
        out,
        expected,
        atol=atol,
        rtol=rtol,
    )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    test_cases = [
        (
            (2, 3),
            "f32",
            1e-5,
            1e-5,
        ),
        (
            (33, 65),
            "f32",
            1e-5,
            1e-5,
        ),
        (
            (512, 4096),
            "f32",
            1e-5,
            1e-5,
        ),
        (
            (33, 65),
            "f16",
            1e-3,
            1e-3,
        ),
        (
            (512, 4096),
            "f16",
            1e-3,
            1e-3,
        ),
        (
            (33, 65),
            "bf16",
            1e-3,
            1e-3,
        ),
        (
            (512, 4096),
            "bf16",
            1e-3,
            1e-3,
        ),
    ]

    rounds = 100

    print("Testing Native -> Triton -> Triton -> Native inside execution context")

    for round_idx in range(rounds):
        for (
            shape,
            dtype_name,
            atol,
            rtol,
        ) in test_cases:
            test_execution_context_ordering(
                shape,
                dtype_name,
                atol,
                rtol,
            )

        if (round_idx + 1) % 10 == 0:
            print(f"  completed {round_idx + 1}/{rounds} rounds")

    print("\033[92mExecution-context ordering test passed!\033[0m")

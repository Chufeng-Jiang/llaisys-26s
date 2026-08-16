import argparse
import math
import os
import sys


# ============================================================
# Repository paths
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")

sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, TEST_DIR)


# ============================================================
# Imports
# ============================================================

import torch
import torch.nn.functional as F

import llaisys

from llaisys.triton import execution_context
from llaisys.triton.backends.registry import get_triton_backend
from llaisys.triton.ops import linear as triton_linear

from test_utils import (
    BenchmarkRecorder,
    benchmark,
    build_experiment_output_path,
    check_equal,
    collect_backend_metadata,
    random_tensor,
    zero_tensor,
)


# ============================================================
# Constants
# ============================================================

DTYPE_BYTES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
}

TEST_DTYPE_PREC = [
    ("f32", 1e-5, 1e-5),
    ("f16", 1e-3, 1e-3),
    ("bf16", 1e-2, 1e-2),
]


# ============================================================
# PyTorch eager reference
# ============================================================
#
# Use torch.nn.functional.linear directly.
#
# F.linear has the semantic contract:
#
#     out = x @ weight.T + bias
#
# Important:
#
#     F.linear does NOT expose an out= argument.
#
# Therefore the reference returns the functional result instead
# of pretending to use a preallocated output tensor.
# ============================================================


def torch_linear(x, weight, bias):
    return F.linear(x, weight, bias)


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_linear(out, x, weight, bias, backend):
    if backend == "native":
        llaisys.Ops.linear(out, x, weight, bias)
        return

    if backend == "triton":
        triton_linear(out, x, weight, bias)
        return

    raise ValueError(f"Unsupported Linear backend: {backend}")


# ============================================================
# Configuration
# ============================================================


def _parse_env_config_value(name, default="default"):
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return value


def get_linear_config(out, x, weight, backend):
    m = x.shape()[0]
    k = x.shape()[1]
    n = weight.shape()[0]

    if backend == "native":
        return "backend_policy", {}

    if backend == "triton":
        triton_backend = get_triton_backend(out.device_type())
        config = triton_backend.linear_config(m, n, k)

        effective = {
            "BLOCK_M": config["BLOCK_M"],
            "BLOCK_N": config["BLOCK_N"],
            "BLOCK_K": config["BLOCK_K"],
            "GROUP_M": config["GROUP_M"],
            "num_warps": config["num_warps"],
            "num_stages": config["num_stages"],
            "ZERO_K_BLOCK_SIZE": config["ZERO_K_BLOCK_SIZE"],
        }

        if "ZERO_K_NUM_WARPS" in config:
            effective["ZERO_K_NUM_WARPS"] = config["ZERO_K_NUM_WARPS"]

        return "effective", effective

    raise ValueError(f"Unsupported Linear backend: {backend}")


def get_linear_config_label(out, x, weight, backend):
    config_status, config = get_linear_config(
        out,
        x,
        weight,
        backend,
    )

    if not config:
        return f"config[{config_status}]"

    values = ", ".join(
        f"{key}={value}"
        for key, value in config.items()
    )

    return f"config[{values}]"


def get_linear_output_filename_config(backend):
    if backend == "native":
        return {
            "POLICY": "backend_default",
        }

    if backend == "triton":
        return {
            "BLOCK_M": _parse_env_config_value("LLAISYS_TRITON_BLOCK_M"),
            "BLOCK_N": _parse_env_config_value("LLAISYS_TRITON_BLOCK_N"),
            "BLOCK_K": _parse_env_config_value("LLAISYS_TRITON_BLOCK_K"),
            "GROUP_M": _parse_env_config_value("LLAISYS_TRITON_GROUP_M"),
            "NUM_WARPS": _parse_env_config_value("LLAISYS_TRITON_NUM_WARPS"),
            "NUM_STAGES": _parse_env_config_value("LLAISYS_TRITON_NUM_STAGES"),
        }

    raise ValueError(f"Unsupported Linear backend: {backend}")


# ============================================================
# Derived metrics
# ============================================================
#
# GEMM:
#
#     [M, K] @ [N, K]^T -> [M, N]
#
# Nominal arithmetic:
#
#     2 * M * N * K
#
# plus one bias addition per output element when bias is enabled.
#
# Minimum logical I/O:
#
#     read x
#     read weight
#     read bias, if present
#     write output
#
# This is an operator-level logical traffic model, not measured
# DRAM traffic. Cache reuse and implementation-specific workspace
# traffic are intentionally not modeled here.
# ============================================================


def get_linear_nominal_flops(m, n, k, use_bias):
    flops = 2 * m * n * k

    if use_bias:
        flops += m * n

    return flops


def get_linear_minimum_logical_io_bytes(
    m,
    n,
    k,
    use_bias,
    dtype_name,
):
    element_size = DTYPE_BYTES[dtype_name]

    elements = (
        m * k
        + n * k
        + m * n
    )

    if use_bias:
        elements += n

    return elements * element_size


def get_effective_bandwidth_gbs(traffic_bytes, median_ms):
    return traffic_bytes / median_ms / 1_000_000.0


def get_throughput_tflops(flops, median_ms):
    return flops / median_ms / 1_000_000_000.0


def get_linear_derived_metrics(
    stats,
    m,
    n,
    k,
    use_bias,
    dtype_name,
):
    logical_bytes = get_linear_minimum_logical_io_bytes(
        m,
        n,
        k,
        use_bias,
        dtype_name,
    )

    nominal_flops = get_linear_nominal_flops(
        m,
        n,
        k,
        use_bias,
    )

    llaisys_stats = stats["llaisys"]
    torch_stats = stats.get("torch")

    derived = {
        "minimum_logical_io_traffic_bytes": logical_bytes,
        "nominal_flops": nominal_flops,
        "llaisys_effective_io_bandwidth_gbs": get_effective_bandwidth_gbs(
            logical_bytes,
            llaisys_stats["median_ms"],
        ),
        "llaisys_throughput_tflops": get_throughput_tflops(
            nominal_flops,
            llaisys_stats["median_ms"],
        ),
    }

    if torch_stats is not None:
        derived.update(
            {
                "torch_equivalent_io_bandwidth_gbs": get_effective_bandwidth_gbs(
                    logical_bytes,
                    torch_stats["median_ms"],
                ),
                "torch_throughput_tflops": get_throughput_tflops(
                    nominal_flops,
                    torch_stats["median_ms"],
                ),
            }
        )

    return derived


def print_linear_derived_metrics(
    derived,
    device_name,
    show_bandwidth,
    show_throughput,
):
    if show_bandwidth:
        print(
            f"        LLAISYS {device_name} effective minimum-I/O bandwidth: "
            f"{derived['llaisys_effective_io_bandwidth_gbs']:.2f} GB/s"
        )

        torch_bandwidth = derived.get(
            "torch_equivalent_io_bandwidth_gbs"
        )

        if torch_bandwidth is not None:
            print(
                f"        Torch {device_name} equivalent minimum-I/O bandwidth: "
                f"{torch_bandwidth:.2f} GB/s"
            )

    if show_throughput:
        print(
            f"        LLAISYS {device_name} nominal throughput: "
            f"{derived['llaisys_throughput_tflops']:.3f} TFLOP/s"
        )

        torch_throughput = derived.get(
            "torch_throughput_tflops"
        )

        if torch_throughput is not None:
            print(
                f"        Torch {device_name} nominal throughput: "
                f"{torch_throughput:.3f} TFLOP/s"
            )


# ============================================================
# Benchmark
# ============================================================


def benchmark_linear(
    torch_x,
    torch_weight,
    torch_bias,
    llaisys_out,
    llaisys_x,
    llaisys_weight,
    llaisys_bias,
    backend,
    backend_variant,
    backend_implementation,
    device_name,
    dtype_name,
    suite,
    seed,
    warmup,
    repeat,
    rounds,
    benchmark_order,
    show_config,
    show_bandwidth,
    show_throughput,
    recorder,
    device_metadata,
):
    m = llaisys_x.shape()[0]
    k = llaisys_x.shape()[1]
    n = llaisys_weight.shape()[0]
    use_bias = llaisys_bias is not None

    config_status, config = get_linear_config(
        llaisys_out,
        llaisys_x,
        llaisys_weight,
        backend,
    )

    label = (
        f"Linear M={m} N={n} K={k} "
        f"bias={use_bias} "
        f"dtype={dtype_name} "
        f"backend={backend}"
    )

    if show_config:
        label += (
            " "
            + get_linear_config_label(
                llaisys_out,
                llaisys_x,
                llaisys_weight,
                backend,
            )
        )

    print(f"        {label}:")

    # F.linear is intentionally used as a true functional reference.
    # It returns its own output tensor because F.linear has no out= API.
    torch_fn = lambda: torch_linear(
        torch_x,
        torch_weight,
        torch_bias,
    )

    llaisys_fn = lambda: run_llaisys_linear(
        llaisys_out,
        llaisys_x,
        llaisys_weight,
        llaisys_bias,
        backend,
    )

    if backend == "native":
        stats = benchmark(
            torch_fn,
            llaisys_fn,
            device_name,
            warmup=warmup,
            repeat=repeat,
            rounds=rounds,
            benchmark_order=benchmark_order,
        )
    elif backend == "triton":
        with execution_context(
            llaisys_out.device_type(),
            llaisys_out.device_id(),
        ):
            stats = benchmark(
                torch_fn,
                llaisys_fn,
                device_name,
                warmup=warmup,
                repeat=repeat,
                rounds=rounds,
                benchmark_order=benchmark_order,
            )
    else:
        raise ValueError(
            f"Unsupported Linear backend: {backend}"
        )

    derived = get_linear_derived_metrics(
        stats,
        m,
        n,
        k,
        use_bias,
        dtype_name,
    )

    if show_bandwidth or show_throughput:
        print_linear_derived_metrics(
            derived,
            device_name,
            show_bandwidth,
            show_throughput,
        )

    recorder.record_microbenchmark(
        op="linear",
        backend_name=backend,
        backend_variant=backend_variant,
        backend_implementation=backend_implementation,
        suite=suite,
        device_name=device_name,
        device_id=llaisys_out.device_id(),
        shape=(m, n),
        numel=m * n,
        dtype_name=dtype_name,
        seed=seed,
        config=config,
        config_status=config_status,
        warmup=warmup,
        repeat=repeat,
        rounds=rounds,
        benchmark_order=benchmark_order,
        stats=stats,
        derived=derived,
        workload_metadata={
            "m": m,
            "n": n,
            "k": k,
            "use_bias": use_bias,
            "input_shape": [m, k],
            "weight_shape": [n, k],
            "output_shape": [m, n],
            "torch_reference": "torch.nn.functional.linear",
            "torch_reference_output_policy": "functional_return",
            "input_distribution": "uniform[0,0.1)",
            "weight_distribution": "uniform[0,0.01)",
        },
        device_metadata=device_metadata,
    )


# ============================================================
# One correctness / performance case
# ============================================================


def test_op_linear(
    m,
    n,
    k,
    use_bias=True,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    backend="native",
    profile=False,
    backend_variant="unspecified",
    backend_implementation=None,
    suite="correctness",
    seed=0,
    warmup=10,
    repeat=100,
    rounds=10,
    benchmark_order="alternating",
    show_config=False,
    show_bandwidth=False,
    show_throughput=False,
    recorder=None,
    device_metadata=None,
    case_name=None,
):
    x_shape = (m, k)
    weight_shape = (n, k)
    out_shape = (m, n)

    case_prefix = (
        f"{case_name} "
        if case_name is not None
        else ""
    )

    print(
        f"   {case_prefix}"
        f"M={m} N={n} K={k} "
        f"bias={use_bias} "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    # ========================================================
    # Input / weight
    # ========================================================

    torch_x, llaisys_x = random_tensor(
        x_shape,
        dtype_name,
        device_name,
        scale=0.1,
    )

    torch_weight, llaisys_weight = random_tensor(
        weight_shape,
        dtype_name,
        device_name,
        scale=0.01,
    )

    # ========================================================
    # Optional bias
    # ========================================================

    torch_bias = None
    llaisys_bias = None

    if use_bias:
        torch_bias, llaisys_bias = random_tensor(
            (n,),
            dtype_name,
            device_name,
            scale=0.1,
            bias=-0.05,
        )

    # ========================================================
    # PyTorch functional reference
    # ========================================================

    torch_out = torch_linear(
        torch_x,
        torch_weight,
        torch_bias,
    )

    # ========================================================
    # LLAISYS output
    # ========================================================

    _, llaisys_out = zero_tensor(
        out_shape,
        dtype_name,
        device_name,
    )

    run_llaisys_linear(
        llaisys_out,
        llaisys_x,
        llaisys_weight,
        llaisys_bias,
        backend,
    )

    # ========================================================
    # Correctness
    #
    # GEMM reduction order can differ between PyTorch/native
    # libraries and Triton, so Linear should not use strict
    # bitwise equality.
    # ========================================================

    assert check_equal(
        llaisys_out,
        torch_out,
        atol=atol,
        rtol=rtol,
    ), (
        "Linear mismatch: "
        f"case={case_name}, "
        f"M={m}, N={n}, K={k}, "
        f"bias={use_bias}, "
        f"dtype={dtype_name}, "
        f"device={device_name}, "
        f"backend={backend}"
    )

    if not profile:
        return

    if recorder is None:
        recorder = BenchmarkRecorder()

    benchmark_linear(
        torch_x,
        torch_weight,
        torch_bias,
        llaisys_out,
        llaisys_x,
        llaisys_weight,
        llaisys_bias,
        backend,
        backend_variant,
        backend_implementation,
        device_name,
        dtype_name,
        suite,
        seed,
        warmup,
        repeat,
        rounds,
        benchmark_order,
        show_config,
        show_bandwidth,
        show_throughput,
        recorder,
        device_metadata or {},
    )


# ============================================================
# Correctness suites
# ============================================================


def get_fixed_correctness_cases():
    return [
        # ----------------------------------------------------
        # Tiny / degenerate feature counts
        # ----------------------------------------------------
        ("scalar_no_bias", 1, 1, 1, False),
        ("scalar_bias", 1, 1, 1, True),
        ("tiny_no_bias", 2, 3, 4, False),
        ("tiny_bias", 2, 3, 4, True),

        # ----------------------------------------------------
        # K tile boundary around the current portable baseline
        # BLOCK_K = 32.
        # ----------------------------------------------------
        ("k31", 3, 37, 31, True),
        ("k32", 3, 37, 32, False),
        ("k33", 17, 37, 33, True),

        # ----------------------------------------------------
        # M boundary around BLOCK_M = 16.
        # ----------------------------------------------------
        ("m15", 15, 37, 33, True),
        ("m16", 16, 37, 33, False),
        ("m17", 17, 37, 33, True),

        # ----------------------------------------------------
        # N boundary around BLOCK_N = 32.
        # ----------------------------------------------------
        ("n31", 17, 31, 33, True),
        ("n32", 17, 32, 33, False),
        ("n33", 17, 33, 33, True),

        # ----------------------------------------------------
        # All three dimensions around tile boundaries.
        # ----------------------------------------------------
        ("tile_minus_one", 15, 31, 31, True),
        ("tile_exact", 16, 32, 32, True),
        ("tile_plus_one", 17, 33, 33, True),

        # ----------------------------------------------------
        # Highly rectangular matrices.
        # ----------------------------------------------------
        ("row_vector", 1, 257, 65, True),
        ("column_like_output", 65, 1, 257, True),
        ("rectangular", 7, 129, 63, False),

        # ----------------------------------------------------
        # K == 0 semantics.
        #
        # with bias:
        #     output = broadcast(bias)
        #
        # without bias:
        #     output = zero
        # ----------------------------------------------------
        ("zero_k_tiny_bias", 2, 3, 0, True),
        ("zero_k_tiny_no_bias", 2, 3, 0, False),
        ("zero_k_tail_bias", 17, 37, 0, True),
        ("zero_k_tail_no_bias", 17, 37, 0, False),

        # ----------------------------------------------------
        # GROUP_M boundary for the current portable baseline:
        #
        # BLOCK_M * GROUP_M = 16 * 8 = 128 rows.
        # ----------------------------------------------------
        ("group_m_minus_one", 127, 33, 31, True),
        ("group_m_exact", 128, 33, 32, False),
        ("group_m_plus_one", 129, 33, 33, True),

        # ----------------------------------------------------
        # Representative inference workloads.
        # ----------------------------------------------------
        ("decode", 1, 4096, 4096, False),
        ("small_batch_decode", 32, 4096, 4096, True),
        ("prefill", 512, 4096, 4096, True),
    ]


def get_triton_dynamic_boundary_cases(device_name):
    triton_backend = get_triton_backend(
        {
            "nvidia": llaisys.DeviceType.NVIDIA,
            "metax": llaisys.DeviceType.METAX,
            "amd": getattr(llaisys.DeviceType, "AMD", None),
        }[device_name]
    )

    # Use a non-degenerate shape only to query the policy.
    config = triton_backend.linear_config(
        17,
        37,
        33,
    )

    block_m = int(config["BLOCK_M"])
    block_n = int(config["BLOCK_N"])
    block_k = int(config["BLOCK_K"])
    group_m = int(config["GROUP_M"])
    zero_k_block = int(config["ZERO_K_BLOCK_SIZE"])

    cases = []

    # --------------------------------------------------------
    # M boundary
    # --------------------------------------------------------

    for delta, label in (
        (-1, "minus_one"),
        (0, "exact"),
        (1, "plus_one"),
    ):
        m = block_m + delta

        if m > 0:
            cases.append(
                (
                    f"dynamic_block_m_{label}",
                    m,
                    block_n + 5,
                    block_k + 1,
                    True,
                )
            )

    # --------------------------------------------------------
    # N boundary
    # --------------------------------------------------------

    for delta, label in (
        (-1, "minus_one"),
        (0, "exact"),
        (1, "plus_one"),
    ):
        n = block_n + delta

        if n > 0:
            cases.append(
                (
                    f"dynamic_block_n_{label}",
                    block_m + 1,
                    n,
                    block_k + 1,
                    True,
                )
            )

    # --------------------------------------------------------
    # K boundary
    # --------------------------------------------------------

    for delta, label in (
        (-1, "minus_one"),
        (0, "exact"),
        (1, "plus_one"),
    ):
        k = block_k + delta

        if k >= 0:
            cases.append(
                (
                    f"dynamic_block_k_{label}",
                    block_m + 1,
                    block_n + 5,
                    k,
                    True,
                )
            )

    # --------------------------------------------------------
    # GROUP_M row boundary
    # --------------------------------------------------------

    group_rows = block_m * group_m

    for delta, label in (
        (-1, "minus_one"),
        (0, "exact"),
        (1, "plus_one"),
    ):
        m = group_rows + delta

        if m > 0:
            cases.append(
                (
                    f"dynamic_group_m_{label}",
                    m,
                    block_n + 1,
                    block_k + 1,
                    True,
                )
            )

    # --------------------------------------------------------
    # ZERO_K_BLOCK_SIZE boundary.
    #
    # Use M=1 so output numel == N.
    # Test both bias/no-bias paths around the 1D zero-K kernel
    # block boundary.
    # --------------------------------------------------------

    for delta, label in (
        (-1, "minus_one"),
        (0, "exact"),
        (1, "plus_one"),
    ):
        n = zero_k_block + delta

        if n <= 0:
            continue

        cases.append(
            (
                f"dynamic_zero_k_{label}_bias",
                1,
                n,
                0,
                True,
            )
        )

        cases.append(
            (
                f"dynamic_zero_k_{label}_no_bias",
                1,
                n,
                0,
                False,
            )
        )

    # Remove exact duplicates while preserving order.
    result = []
    seen = set()

    for case in cases:
        key = case[1:]

        if key in seen:
            continue

        seen.add(key)
        result.append(case)

    return result


def get_profile_cases(profile_suite):
    sweep = [
        ("square_256", 256, 256, 256, True),
        ("square_512", 512, 512, 512, True),
        ("square_1024", 1024, 1024, 1024, True),
    ]

    llm = [
        ("decode", 1, 4096, 4096, False),
        ("small_batch_decode", 32, 4096, 4096, True),
        ("prefill", 512, 4096, 4096, True),
    ]

    if profile_suite == "sweep":
        return [
            ("sweep", *case)
            for case in sweep
        ]

    if profile_suite == "llm":
        return [
            ("llm", *case)
            for case in llm
        ]

    return (
        [
            ("sweep", *case)
            for case in sweep
        ]
        + [
            ("llm", *case)
            for case in llm
        ]
    )


# ============================================================
# Profiler helpers
# ============================================================


def _torch_profiler_synchronize(device_name):
    if device_name in ("nvidia", "amd"):
        torch.cuda.synchronize()


def _begin_profiler_range(label, device_name):
    if device_name not in ("nvidia", "amd"):
        return False

    if not torch.cuda.is_available():
        return False

    try:
        torch.cuda.nvtx.range_push(label)
        return True
    except Exception:
        return False


def _end_profiler_range(range_pushed):
    if not range_pushed:
        return

    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def run_linear_profiler_case(
    *,
    m,
    n,
    k,
    use_bias,
    dtype_name,
    atol,
    rtol,
    device_name,
    backend,
    backend_variant,
    profiler_target,
    profiler_warmup,
    profiler_launches,
    profiler_check,
    show_config,
):
    print()
    print("=== Profiler single case ===")
    print(
        f"   target <{profiler_target}> "
        f"M={m} N={n} K={k} "
        f"bias={use_bias} "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    torch_x, llaisys_x = random_tensor(
        (m, k),
        dtype_name,
        device_name,
        scale=0.1,
    )

    torch_weight, llaisys_weight = random_tensor(
        (n, k),
        dtype_name,
        device_name,
        scale=0.01,
    )

    torch_bias = None
    llaisys_bias = None

    if use_bias:
        torch_bias, llaisys_bias = random_tensor(
            (n,),
            dtype_name,
            device_name,
            scale=0.1,
            bias=-0.05,
        )

    _, llaisys_out = zero_tensor(
        (m, n),
        dtype_name,
        device_name,
    )

    if profiler_target == "torch":
        if device_name == "metax":
            raise ValueError(
                "Torch profiler target is unavailable for MetaX because "
                "the current MetaX reference tensor is hosted on CPU."
            )

        target_fn = lambda: torch_linear(
            torch_x,
            torch_weight,
            torch_bias,
        )

        synchronize = lambda: _torch_profiler_synchronize(
            device_name
        )

        config_status = "reference"
        config = {}

        target_label = (
            f"LLAISYS_PROFILE:linear:torch:{device_name}:"
            f"M={m}:N={n}:K={k}:bias={use_bias}:"
            f"dtype={dtype_name}"
        )

        for _ in range(profiler_warmup):
            target_fn()

        synchronize()

        range_pushed = _begin_profiler_range(
            target_label,
            device_name,
        )

        try:
            for _ in range(profiler_launches):
                target_fn()

            synchronize()
        finally:
            _end_profiler_range(range_pushed)

        if profiler_check:
            torch_out = torch_linear(
                torch_x,
                torch_weight,
                torch_bias,
            )

            run_llaisys_linear(
                llaisys_out,
                llaisys_x,
                llaisys_weight,
                llaisys_bias,
                backend,
            )

            assert check_equal(
                llaisys_out,
                torch_out,
                atol=atol,
                rtol=rtol,
            ), (
                "Linear profiler correctness mismatch: "
                f"M={m}, N={n}, K={k}, "
                f"dtype={dtype_name}, "
                f"device={device_name}, "
                f"backend={backend}"
            )
    else:
        config_status, config = get_linear_config(
            llaisys_out,
            llaisys_x,
            llaisys_weight,
            backend,
        )

        if show_config:
            print(
                "        "
                + get_linear_config_label(
                    llaisys_out,
                    llaisys_x,
                    llaisys_weight,
                    backend,
                )
            )

        target_fn = lambda: run_llaisys_linear(
            llaisys_out,
            llaisys_x,
            llaisys_weight,
            llaisys_bias,
            backend,
        )

        api = llaisys.RuntimeAPI(
            llaisys_out.device_type()
        )

        synchronize = api.device_synchronize

        config_tag = ",".join(
            f"{key}={value}"
            for key, value in config.items()
        )

        target_label = (
            f"LLAISYS_PROFILE:linear:{backend}:"
            f"{backend_variant}:{device_name}:"
            f"M={m}:N={n}:K={k}:bias={use_bias}:"
            f"dtype={dtype_name}:{config_tag}"
        )

        def execute_target():
            for _ in range(profiler_warmup):
                target_fn()

            synchronize()

            range_pushed = _begin_profiler_range(
                target_label,
                device_name,
            )

            try:
                for _ in range(profiler_launches):
                    target_fn()

                synchronize()
            finally:
                _end_profiler_range(
                    range_pushed
                )

        if backend == "triton":
            with execution_context(
                llaisys_out.device_type(),
                llaisys_out.device_id(),
            ):
                execute_target()
        else:
            execute_target()

        if profiler_check:
            torch_out = torch_linear(
                torch_x,
                torch_weight,
                torch_bias,
            )

            _torch_profiler_synchronize(
                device_name
            )

            assert check_equal(
                llaisys_out,
                torch_out,
                atol=atol,
                rtol=rtol,
            ), (
                "Linear profiler correctness mismatch: "
                f"M={m}, N={n}, K={k}, "
                f"dtype={dtype_name}, "
                f"device={device_name}, "
                f"backend={backend}"
            )

    print(
        f"Profiler target range: {target_label}"
    )

    print(
        f"Profiler launches: warmup={profiler_warmup}, "
        f"target={profiler_launches}"
    )

    if profiler_check:
        print("Profiler post-check: passed")

    return {
        "target": profiler_target,
        "m": m,
        "n": n,
        "k": k,
        "use_bias": use_bias,
        "dtype": dtype_name,
        "config_status": config_status,
        "config": config,
        "warmup": profiler_warmup,
        "launches": profiler_launches,
        "range": target_label,
    }


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
            "amd",
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
        "--backend-variant",
        default="unspecified",
        type=str,
        help=(
            "Experiment variant label, for example baseline, tuned, "
            "autotuned, or vendor-specific."
        ),
    )

    parser.add_argument(
        "--backend-implementation",
        default=None,
        type=str,
        help=(
            "Optional implementation override. Native normally maps to "
            "cpu/cuda/maca/hip and Triton maps to triton."
        ),
    )

    execution_mode = parser.add_mutually_exclusive_group()

    execution_mode.add_argument(
        "--profile",
        action="store_true",
        help="Run the paper-oriented Linear microbenchmark suite.",
    )

    execution_mode.add_argument(
        "--profiler-mode",
        action="store_true",
        help=(
            "Run one controlled Linear workload for "
            "ncu/nsys/mcProfiler/rocprof."
        ),
    )

    parser.add_argument(
        "--case-m",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--case-n",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--case-k",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--case-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--case-dtype",
        default="f16",
        choices=[
            "f32",
            "f16",
            "bf16",
        ],
        type=str,
    )

    parser.add_argument(
        "--profiler-target",
        default="llaisys",
        choices=[
            "llaisys",
            "torch",
        ],
        type=str,
    )

    parser.add_argument(
        "--profiler-warmup",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--profiler-launches",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--profiler-check",
        action="store_true",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
    )

    parser.add_argument(
        "--show-bandwidth",
        action="store_true",
    )

    parser.add_argument(
        "--show-throughput",
        action="store_true",
    )

    parser.add_argument(
        "--skip-correctness",
        action="store_true",
    )

    parser.add_argument(
        "--skip-dynamic-boundaries",
        action="store_true",
        help=(
            "Skip Triton configuration-derived tile and zero-K "
            "boundary correctness cases."
        ),
    )

    parser.add_argument(
        "--profile-suite",
        default="all",
        choices=[
            "sweep",
            "llm",
            "all",
        ],
        type=str,
    )

    parser.add_argument(
        "--seed",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--warmup",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--repeat",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--rounds",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--benchmark-order",
        default="alternating",
        choices=[
            "llaisys_then_torch",
            "torch_then_llaisys",
            "alternating",
        ],
        type=str,
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        type=str,
    )

    parser.add_argument(
        "--no-record",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--run-id",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--run-note",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    # ========================================================
    # Validation
    # ========================================================

    if (
        args.backend == "triton"
        and args.device == "cpu"
    ):
        raise ValueError(
            "Triton Linear requires a GPU device"
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup must be non-negative"
        )

    if args.repeat <= 0:
        raise ValueError(
            "--repeat must be greater than zero"
        )

    if args.rounds <= 0:
        raise ValueError(
            "--rounds must be greater than zero"
        )

    if args.profiler_warmup < 0:
        raise ValueError(
            "--profiler-warmup must be non-negative"
        )

    if args.profiler_launches <= 0:
        raise ValueError(
            "--profiler-launches must be greater than zero"
        )

    if args.profiler_mode:
        if (
            args.case_m is None
            or args.case_n is None
            or args.case_k is None
        ):
            raise ValueError(
                "--case-m, --case-n, and --case-k are required "
                "with --profiler-mode"
            )

        if args.case_m <= 0:
            raise ValueError(
                "--case-m must be greater than zero"
            )

        if args.case_n <= 0:
            raise ValueError(
                "--case-n must be greater than zero"
            )

        if args.case_k < 0:
            raise ValueError(
                "--case-k must not be negative"
            )

    torch.manual_seed(
        args.seed
    )

    if (
        args.device in ("nvidia", "amd")
        and torch.cuda.is_available()
    ):
        torch.cuda.manual_seed_all(
            args.seed
        )

    backend_metadata = collect_backend_metadata(
        args.backend,
        args.device,
        variant=args.backend_variant,
        implementation=args.backend_implementation,
    )

    filename_config = get_linear_output_filename_config(
        args.backend
    )

    if args.output is not None:
        output_path = args.output
    elif (
        args.profile
        and not args.no_record
    ):
        output_path = build_experiment_output_path(
            args.output_dir,
            op="linear",
            device_name=args.device,
            backend=backend_metadata,
            config=filename_config,
        )
    else:
        output_path = None

    run_metadata = {
        "profile_suite": args.profile_suite,
        "benchmark_order": args.benchmark_order,
        "note": args.run_note,
        "reference": {
            "torch": "torch.nn.functional.linear",
            "execution": "eager",
            "output_policy": "functional_return",
        },
        "input_distribution": {
            "x": "uniform[0,0.1)",
            "weight": "uniform[0,0.01)",
            "bias": "uniform[-0.05,0.05)",
        },
        "profiler_mode": args.profiler_mode,
        "profiler_case": {
            "m": args.case_m,
            "n": args.case_n,
            "k": args.case_k,
            "bias": args.case_bias,
            "dtype": args.case_dtype,
            "target": args.profiler_target,
            "warmup": args.profiler_warmup,
            "launches": args.profiler_launches,
        },
        "output": {
            "automatic": args.output is None,
            "directory": args.output_dir,
            "filename_config": filename_config,
        },
    }

    recorder = BenchmarkRecorder(
        output_path=output_path,
        repo_root=REPO_ROOT,
        run_id=args.run_id,
        run_metadata=run_metadata,
    )

    print(
        f"Testing Ops.linear on {args.device} "
        f"with {args.backend} backend"
    )

    print(
        f"Backend identity: "
        f"name={backend_metadata['name']}, "
        f"implementation={backend_metadata['implementation']}, "
        f"variant={backend_metadata['variant']}"
    )

    print(
        f"Random seed: {args.seed}"
    )

    print(
        f"Benchmark protocol: "
        f"warmup={args.warmup}, "
        f"repeat={args.repeat}, "
        f"rounds={args.rounds}, "
        f"order={args.benchmark_order}"
    )

    print(
        f"Using llaisys from: {llaisys.__file__}"
    )

    if output_path is not None:
        print(
            f"Recording JSONL: {output_path}"
        )

        print(
            f"Run ID: {recorder.run_id}"
        )

    # ========================================================
    # Profiler mode
    # ========================================================

    if args.profiler_mode:
        tolerance = {
            "f32": (1e-5, 1e-5),
            "f16": (1e-3, 1e-3),
            "bf16": (1e-2, 1e-2),
        }[args.case_dtype]

        run_linear_profiler_case(
            m=args.case_m,
            n=args.case_n,
            k=args.case_k,
            use_bias=args.case_bias,
            dtype_name=args.case_dtype,
            atol=tolerance[0],
            rtol=tolerance[1],
            device_name=args.device,
            backend=args.backend,
            backend_variant=args.backend_variant,
            profiler_target=args.profiler_target,
            profiler_warmup=args.profiler_warmup,
            profiler_launches=args.profiler_launches,
            profiler_check=args.profiler_check,
            show_config=args.show_config,
        )

        print()
        print(
            "\033[92mProfiler run completed!\033[0m"
        )

        raise SystemExit(0)

    # ========================================================
    # Fixed correctness suite
    # ========================================================

    if not args.skip_correctness:
        print()
        print(
            "=== Correctness: fixed semantic / tile / workload cases ==="
        )

        for (
            case_name,
            m,
            n,
            k,
            use_bias,
        ) in get_fixed_correctness_cases():
            for (
                dtype_name,
                atol,
                rtol,
            ) in TEST_DTYPE_PREC:
                test_op_linear(
                    m,
                    n,
                    k,
                    use_bias,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    device_name=args.device,
                    backend=args.backend,
                    profile=False,
                    case_name=case_name,
                )

    # ========================================================
    # Dynamic Triton boundaries
    # ========================================================

    if (
        args.backend == "triton"
        and not args.skip_correctness
        and not args.skip_dynamic_boundaries
    ):
        print()
        print(
            "=== Correctness: effective Triton configuration boundaries ==="
        )

        dynamic_cases = get_triton_dynamic_boundary_cases(
            args.device
        )

        for (
            case_name,
            m,
            n,
            k,
            use_bias,
        ) in dynamic_cases:
            for (
                dtype_name,
                atol,
                rtol,
            ) in TEST_DTYPE_PREC:
                test_op_linear(
                    m,
                    n,
                    k,
                    use_bias,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    device_name=args.device,
                    backend=args.backend,
                    profile=False,
                    case_name=case_name,
                )

    # ========================================================
    # Performance
    # ========================================================

    if args.profile:
        print()
        print(
            "=== Performance ==="
        )

        for (
            suite,
            case_name,
            m,
            n,
            k,
            use_bias,
        ) in get_profile_cases(
            args.profile_suite
        ):
            for (
                dtype_name,
                atol,
                rtol,
            ) in TEST_DTYPE_PREC:
                test_op_linear(
                    m,
                    n,
                    k,
                    use_bias,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    device_name=args.device,
                    backend=args.backend,
                    profile=True,
                    backend_variant=args.backend_variant,
                    backend_implementation=args.backend_implementation,
                    suite=suite,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    rounds=args.rounds,
                    benchmark_order=args.benchmark_order,
                    show_config=args.show_config,
                    show_bandwidth=args.show_bandwidth,
                    show_throughput=args.show_throughput,
                    recorder=recorder,
                    device_metadata={},
                    case_name=case_name,
                )

    print()
    print(
        "\033[92mTest passed!\033[0m"
    )
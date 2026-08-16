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

# Always prefer the current repository implementation over an
# installed site-packages copy of llaisys.
sys.path.insert(0, PYTHON_DIR)
sys.path.insert(0, TEST_DIR)


# ============================================================
# Imports
# ============================================================

import torch

import llaisys

from llaisys.triton import execution_context
from llaisys.triton.backends.registry import get_triton_backend
from llaisys.triton.ops import rms_norm as triton_rms_norm

from test_utils import (
    BenchmarkRecorder,
    benchmark,
    build_experiment_output_path,
    check_equal,
    collect_backend_metadata,
    llaisys_device,
    llaisys_dtype,
    random_tensor,
    reference_torch_device,
    torch_dtype,
    torch_to_llaisys_memcpy_kind,
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

DEFAULT_EPS = 1e-5


# ============================================================
# PyTorch reference / performance baseline
# ============================================================
#
# Use PyTorch's RMSNorm operator directly instead of decomposing
# RMSNorm into multiple eager tensor operations.
#
# This gives a substantially fairer framework-level baseline for
# Native CUDA / Triton comparisons. The timed Torch path measures
# torch.nn.functional.rms_norm itself and does not add an explicit
# out.copy_() inside the benchmark function.
#
# Note: torch.nn.functional.rms_norm returns a new output tensor, while
# LLAISYS writes into a preallocated output tensor. This asymmetry is
# recorded explicitly in the experiment metadata.
# ============================================================


def torch_rms_norm(x, weight, eps):
    return torch.nn.functional.rms_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        eps=eps,
    )


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_rms_norm(out, x, weight, eps, backend):
    if backend == "native":
        llaisys.Ops.rms_norm(out, x, weight, eps)
        return

    if backend == "triton":
        triton_rms_norm(out, x, weight, eps)
        return

    raise ValueError(f"Unsupported RMSNorm backend: {backend}")


# ============================================================
# Effective configuration
# ============================================================
#
# RMSNorm differs from Add/SwiGLU:
#
#     configuration is resolved from the row width (ncol),
#     not from the total number of tensor elements.
#
# The portable Triton baseline currently resolves BLOCK_SIZE as
# the next power of two covering ncol and selects num_warps from
# the resulting block size. Always query the backend so JSONL
# records contain the effective per-case configuration.
# ============================================================


def get_rms_norm_config(tensor, backend):
    shape = tensor.shape()

    if len(shape) != 2:
        raise ValueError("RMSNorm configuration requires a 2D tensor")

    ncol = shape[1]

    if backend == "native":
        block_size = os.getenv("LLAISYS_BLOCK_SIZE")

        if block_size is None:
            return "default_unresolved", {"BLOCK_SIZE": None}

        try:
            block_size = int(block_size)
        except ValueError:
            pass

        return "requested_override", {"BLOCK_SIZE": block_size}

    if backend == "triton":
        triton_backend = get_triton_backend(tensor.device_type())
        config = triton_backend.rms_norm_config(ncol)

        return "effective", {
            "BLOCK_SIZE": config["BLOCK_SIZE"],
            "num_warps": config["num_warps"],
        }

    raise ValueError(f"Unsupported RMSNorm backend: {backend}")


def get_rms_norm_config_label(tensor, backend):
    status, config = get_rms_norm_config(tensor, backend)

    if backend == "native" and status == "default_unresolved":
        return "config[BLOCK_SIZE=default]"

    values = ", ".join(f"{key}={value}" for key, value in config.items())
    return f"config[{values}]"


def _parse_env_config_value(name, default="default"):
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return value


def get_rms_norm_output_filename_config(backend):
    if backend == "native":
        return {
            "BLOCK_SIZE": _parse_env_config_value("LLAISYS_BLOCK_SIZE"),
        }

    if backend == "triton":
        return {
            "BLOCK_SIZE": _parse_env_config_value("LLAISYS_TRITON_BLOCK_SIZE"),
            "NUM_WARPS": _parse_env_config_value("LLAISYS_TRITON_NUM_WARPS"),
        }

    raise ValueError(f"Unsupported RMSNorm backend: {backend}")


# ============================================================
# Derived performance metrics
# ============================================================
#
# Minimum logical unique-tensor I/O for X[M, D], W[D], OUT[M, D]:
#
#     read X       -> M * D * element_size
#     read W       -> D * element_size
#     write OUT    -> M * D * element_size
#
# Therefore:
#
#     nominal_io_traffic_bytes
#       = (2 * M * D + D) * element_size
#
# This is intentionally a logical/effective metric. A real GPU can
# reload weight values, keep them in cache, or generate additional
# traffic. Torch bandwidth is also reported as an equivalent logical
# bandwidth rather than a measured DRAM-counter bandwidth.
# ============================================================


def get_rms_norm_nominal_io_traffic(rows, ncol, dtype_name):
    element_size = DTYPE_BYTES[dtype_name]
    input_bytes = rows * ncol * element_size
    weight_bytes = ncol * element_size
    output_bytes = rows * ncol * element_size

    return {
        "input_bytes": input_bytes,
        "weight_bytes": weight_bytes,
        "output_bytes": output_bytes,
        "total_bytes": input_bytes + weight_bytes + output_bytes,
    }


def get_effective_bandwidth_gbs(traffic_bytes, median_ms):
    return traffic_bytes / median_ms / 1_000_000.0


def get_element_throughput_gelem_s(numel, median_ms):
    return numel / median_ms / 1_000_000.0


def get_rms_norm_derived_metrics(stats, rows, ncol, dtype_name):
    traffic = get_rms_norm_nominal_io_traffic(rows, ncol, dtype_name)
    numel = rows * ncol
    llaisys_stats = stats["llaisys"]
    torch_stats = stats.get("torch")

    derived = {
        "nominal_input_read_bytes": traffic["input_bytes"],
        "nominal_weight_read_bytes": traffic["weight_bytes"],
        "nominal_output_write_bytes": traffic["output_bytes"],
        "nominal_io_traffic_bytes": traffic["total_bytes"],
        "llaisys_effective_io_bandwidth_gbs": get_effective_bandwidth_gbs(
            traffic["total_bytes"],
            llaisys_stats["median_ms"],
        ),
        "llaisys_element_throughput_gelem_s": get_element_throughput_gelem_s(
            numel,
            llaisys_stats["median_ms"],
        ),
        "torch_equivalent_io_bandwidth_gbs": None,
        "torch_element_throughput_gelem_s": None,
    }

    if torch_stats is not None:
        derived["torch_equivalent_io_bandwidth_gbs"] = get_effective_bandwidth_gbs(
            traffic["total_bytes"],
            torch_stats["median_ms"],
        )
        derived["torch_element_throughput_gelem_s"] = get_element_throughput_gelem_s(
            numel,
            torch_stats["median_ms"],
        )

    return derived


def print_rms_norm_derived_metrics(
    derived,
    device_name,
    show_bandwidth,
    show_throughput,
):
    if show_bandwidth:
        print(
            f"        LLAISYS {device_name} effective I/O bandwidth: "
            f"{derived['llaisys_effective_io_bandwidth_gbs']:.2f} GB/s"
        )

        torch_bandwidth = derived.get("torch_equivalent_io_bandwidth_gbs")
        if torch_bandwidth is not None:
            print(
                f"        Torch {device_name} equivalent I/O bandwidth: "
                f"{torch_bandwidth:.2f} GB/s"
            )

    if show_throughput:
        print(
            f"        LLAISYS {device_name} element throughput: "
            f"{derived['llaisys_element_throughput_gelem_s']:.3f} GElem/s"
        )

        torch_throughput = derived.get("torch_element_throughput_gelem_s")
        if torch_throughput is not None:
            print(
                f"        Torch {device_name} element throughput: "
                f"{torch_throughput:.3f} GElem/s"
            )


# ============================================================
# Deterministic tensor construction
# ============================================================


def tensor_pair_from_values(values, dtype_name, device_name, device_id=0):
    torch_tensor = torch.tensor(
        values,
        dtype=torch_dtype(dtype_name),
        device=reference_torch_device(device_name, device_id),
    ).contiguous()

    llaisys_tensor = llaisys.Tensor(
        torch_tensor.shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()

    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_llaisys_memcpy_kind(device_name),
    )

    return torch_tensor, llaisys_tensor


# ============================================================
# Deterministic semantic correctness
# ============================================================


def test_rms_norm_semantic_case(
    name,
    x_values,
    weight_values,
    eps,
    dtype_name,
    atol,
    rtol,
    device_name,
    backend,
):
    rows = len(x_values)
    ncol = len(weight_values)
    shape = (rows, ncol)

    print(
        f"   semantic {name} "
        f"shape {shape} "
        f"eps <{eps:g}> "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    x_ref, x = tensor_pair_from_values(
        x_values,
        dtype_name,
        device_name,
    )
    weight_ref, weight = tensor_pair_from_values(
        weight_values,
        dtype_name,
        device_name,
    )
    out_ref, out = zero_tensor(
        shape,
        dtype_name,
        device_name,
    )

    out_ref.copy_(torch_rms_norm(x_ref, weight_ref, eps))
    run_llaisys_rms_norm(out, x, weight, eps, backend)

    assert check_equal(out, out_ref, atol=atol, rtol=rtol), (
        f"RMSNorm semantic mismatch: "
        f"case={name}, shape={shape}, eps={eps}, "
        f"dtype={dtype_name}, device={device_name}, backend={backend}"
    )


def run_rms_norm_semantic_tests(
    dtype_name,
    atol,
    rtol,
    device_name,
    backend,
    eps,
):
    cases = [
        (
            "zero_input_unit_weight",
            [[0.0, 0.0, 0.0, 0.0]],
            [1.0, 1.0, 1.0, 1.0],
        ),
        (
            "constant_positive_unit_weight",
            [[2.0, 2.0, 2.0, 2.0]],
            [1.0, 1.0, 1.0, 1.0],
        ),
        (
            "signed_input_weight_scaling",
            [[-3.0, 4.0, -5.0, 12.0]],
            [1.0, -1.0, 0.5, 0.0],
        ),
        (
            "row_independence",
            [
                [1.0, -1.0, 1.0, -1.0],
                [2.0, -4.0, 6.0, -8.0],
                [0.5, 1.5, -2.5, 3.5],
            ],
            [1.0, 0.5, -1.0, 2.0],
        ),
        (
            "near_zero_eps_dominated",
            [
                [1e-4, -1e-4, 2e-4, -2e-4],
                [3e-4, -3e-4, 4e-4, -4e-4],
            ],
            [1.0, -0.5, 2.0, 0.25],
        ),
    ]

    for name, x_values, weight_values in cases:
        test_rms_norm_semantic_case(
            name,
            x_values,
            weight_values,
            eps,
            dtype_name,
            atol,
            rtol,
            device_name,
            backend,
        )


# ============================================================
# Profiler helpers
# ============================================================


def parse_case_shape(value):
    text = value.strip().lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]

    if not parts:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")

    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid RMSNorm shape {value!r}; "
            "expected forms such as 4096, 1,4096, or 2048x4096"
        ) from exc

    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(
            "all RMSNorm shape dimensions must be greater than zero"
        )

    # RMSNorm is a 2D public operator in this test harness.
    # A single number is convenient for decode-style profiler cases.
    if len(shape) == 1:
        return (1, shape[0])

    if len(shape) != 2:
        raise argparse.ArgumentTypeError(
            "RMSNorm requires a 2D shape; use forms such as "
            "4096, 1,4096, or 2048,4096"
        )

    return shape


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


def run_rms_norm_profiler_case(
    *,
    shape,
    dtype_name,
    atol,
    rtol,
    eps,
    device_name,
    backend,
    backend_variant,
    profiler_target,
    profiler_warmup,
    profiler_launches,
    profiler_check,
    show_config,
):
    rows, ncol = shape
    numel = rows * ncol

    print()
    print("=== Profiler single case ===")
    print(
        f"   target <{profiler_target}> "
        f"shape {shape} "
        f"rows {rows} "
        f"ncol {ncol} "
        f"numel {numel} "
        f"eps <{eps:g}> "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    x_ref, x = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=4.0,
        bias=-2.0,
    )
    weight_ref, weight = random_tensor(
        (ncol,),
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )
    out_ref, out = zero_tensor(
        shape,
        dtype_name,
        device_name,
    )

    if profiler_target == "torch":
        if device_name == "metax":
            raise ValueError(
                "Torch profiler target is unavailable for MetaX because the "
                "current MetaX reference tensors are hosted on CPU."
            )

        target_fn = lambda: torch_rms_norm(
            x_ref,
            weight_ref,
            eps,
        )
        synchronize = lambda: _torch_profiler_synchronize(device_name)
        config_status = "reference"
        config = {}

        target_label = (
            f"LLAISYS_PROFILE:rms_norm:torch:{device_name}:"
            f"shape={'x'.join(str(dim) for dim in shape)}:"
            f"dtype={dtype_name}:eps={eps:g}"
        )

        for _ in range(profiler_warmup):
            target_fn()
        synchronize()

        range_pushed = _begin_profiler_range(target_label, device_name)
        try:
            for _ in range(profiler_launches):
                target_fn()
            synchronize()
        finally:
            _end_profiler_range(range_pushed)

        if profiler_check:
            out_ref.copy_(
                torch_rms_norm(
                    x_ref,
                    weight_ref,
                    eps,
                )
            )
            _torch_profiler_synchronize(device_name)

            run_llaisys_rms_norm(
                out,
                x,
                weight,
                eps,
                backend,
            )

            assert check_equal(out, out_ref, atol=atol, rtol=rtol), (
                f"RMSNorm profiler correctness mismatch: "
                f"shape={shape}, dtype={dtype_name}, eps={eps}, "
                f"device={device_name}, backend={backend}"
            )

        print(
            "Profiler note: Torch target uses torch.nn.functional.rms_norm. "
            "Use the profiler trace to confirm the backend kernel decomposition "
            "rather than assuming a specific launch count."
        )

    else:
        config_status, config = get_rms_norm_config(out, backend)

        if show_config:
            print(f"        {get_rms_norm_config_label(out, backend)}")

        target_fn = lambda: run_llaisys_rms_norm(
            out,
            x,
            weight,
            eps,
            backend,
        )
        api = llaisys.RuntimeAPI(out.device_type())
        synchronize = api.device_synchronize

        config_tag = ",".join(f"{key}={value}" for key, value in config.items())
        target_label = (
            f"LLAISYS_PROFILE:rms_norm:{backend}:{backend_variant}:{device_name}:"
            f"shape={'x'.join(str(dim) for dim in shape)}:"
            f"dtype={dtype_name}:eps={eps:g}:{config_tag}"
        )

        def execute_target():
            for _ in range(profiler_warmup):
                target_fn()
            synchronize()

            range_pushed = _begin_profiler_range(target_label, device_name)
            try:
                for _ in range(profiler_launches):
                    target_fn()
                synchronize()
            finally:
                _end_profiler_range(range_pushed)

        if backend == "triton":
            with execution_context(
                out.device_type(),
                out.device_id(),
            ):
                execute_target()
        else:
            execute_target()

        if profiler_check:
            out_ref.copy_(
                torch_rms_norm(
                    x_ref,
                    weight_ref,
                    eps,
                )
            )
            _torch_profiler_synchronize(device_name)

            assert check_equal(out, out_ref, atol=atol, rtol=rtol), (
                f"RMSNorm profiler correctness mismatch: "
                f"shape={shape}, dtype={dtype_name}, eps={eps}, "
                f"device={device_name}, backend={backend}"
            )

    print(f"Profiler target range: {target_label}")
    print(
        f"Profiler launches: warmup={profiler_warmup}, "
        f"target={profiler_launches}"
    )

    if profiler_target == "llaisys":
        if backend == "triton":
            print(
                "NCU hint: the Triton target kernel is expected to be "
                f"rms_norm_kernel; use --kernel-name rms_norm_kernel "
                f"--launch-skip {profiler_warmup} "
                f"--launch-count {profiler_launches} after confirming the name."
            )
        else:
            print(
                "NCU hint: first run a discovery profile to confirm the Native "
                "RMSNorm kernel name, then use a precise --kernel-name filter."
            )

    if profiler_check:
        print("Profiler post-check: passed")

    return {
        "target": profiler_target,
        "shape": shape,
        "rows": rows,
        "ncol": ncol,
        "numel": numel,
        "dtype": dtype_name,
        "eps": eps,
        "config_status": config_status,
        "config": config,
        "warmup": profiler_warmup,
        "launches": profiler_launches,
        "range": target_label,
    }


# ============================================================
# Benchmark
# ============================================================


def benchmark_rms_norm(
    torch_out,
    torch_x,
    torch_weight,
    llaisys_out,
    llaisys_x,
    llaisys_weight,
    eps,
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
    shape = llaisys_out.shape()
    rows, ncol = shape
    numel = rows * ncol
    config_status, config = get_rms_norm_config(llaisys_out, backend)

    label = (
        f"RMSNorm shape={shape} "
        f"rows={rows} "
        f"ncol={ncol} "
        f"numel={numel} "
        f"eps={eps:g} "
        f"dtype={dtype_name} "
        f"backend={backend}"
    )

    if show_config:
        label += f" {get_rms_norm_config_label(llaisys_out, backend)}"

    print(f"        {label}:")

    torch_fn = lambda: torch_rms_norm(
        torch_x,
        torch_weight,
        eps,
    )
    llaisys_fn = lambda: run_llaisys_rms_norm(
        llaisys_out,
        llaisys_x,
        llaisys_weight,
        eps,
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
        raise ValueError(f"Unsupported RMSNorm backend: {backend}")

    derived = get_rms_norm_derived_metrics(
        stats,
        rows,
        ncol,
        dtype_name,
    )

    if show_bandwidth or show_throughput:
        print_rms_norm_derived_metrics(
            derived,
            device_name,
            show_bandwidth,
            show_throughput,
        )

    recorder.record_microbenchmark(
        op="rms_norm",
        backend_name=backend,
        backend_variant=backend_variant,
        backend_implementation=backend_implementation,
        suite=suite,
        device_name=device_name,
        device_id=llaisys_out.device_id(),
        shape=shape,
        numel=numel,
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
            "rows": rows,
            "normalized_width": ncol,
            "eps": eps,
            "torch_reference": "torch_nn_functional_rms_norm",
            "torch_output_preallocated": False,
            "llaisys_output_preallocated": True,
            "input_range": [-2.0, 2.0],
            "weight_range": [-1.0, 1.0],
        },
        device_metadata=device_metadata,
    )


# ============================================================
# One random correctness/performance case
# ============================================================


def test_op_rms_norm(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    eps=DEFAULT_EPS,
    device_name="cpu",
    backend="native",
    backend_variant="unspecified",
    backend_implementation=None,
    profile=False,
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
):
    if len(shape) != 2:
        raise ValueError(f"RMSNorm test shape must be 2D, got {shape}")

    if recorder is None:
        recorder = BenchmarkRecorder()

    rows, ncol = shape
    numel = rows * ncol

    print(
        f"   shape {shape} "
        f"rows {rows} "
        f"ncol {ncol} "
        f"numel {numel} "
        f"eps <{eps:g}> "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    x_ref, x = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=4.0,
        bias=-2.0,
    )

    weight_ref, weight = random_tensor(
        (ncol,),
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    out_ref, out = zero_tensor(
        shape,
        dtype_name,
        device_name,
    )

    out_ref.copy_(
        torch_rms_norm(
            x_ref,
            weight_ref,
            eps,
        )
    )

    run_llaisys_rms_norm(
        out,
        x,
        weight,
        eps,
        backend,
    )

    assert check_equal(out, out_ref, atol=atol, rtol=rtol), (
        f"RMSNorm mismatch: "
        f"shape={shape}, rows={rows}, ncol={ncol}, numel={numel}, "
        f"eps={eps}, dtype={dtype_name}, "
        f"device={device_name}, backend={backend}"
    )

    if profile:
        benchmark_rms_norm(
            out_ref,
            x_ref,
            weight_ref,
            out,
            x,
            weight,
            eps,
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
        )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="nvidia",
        choices=["cpu", "nvidia", "metax", "amd"],
        type=str,
    )

    parser.add_argument(
        "--backend",
        default="native",
        choices=["native", "triton"],
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
        help="Run the paper-oriented RMSNorm microbenchmark suite.",
    )

    execution_mode.add_argument(
        "--profiler-mode",
        action="store_true",
        help=(
            "Run one controlled RMSNorm workload for external profilers. "
            "This mode does not run the normal benchmark loop or write "
            "microbenchmark JSONL."
        ),
    )

    parser.add_argument(
        "--case-shape",
        default=None,
        type=parse_case_shape,
        help=(
            "Single profiler shape. 4096 means (1, 4096); "
            "32,4096 and 32x4096 are also accepted. "
            "Required with --profiler-mode."
        ),
    )

    parser.add_argument(
        "--case-dtype",
        default="f16",
        choices=["f32", "f16", "bf16"],
        type=str,
    )

    parser.add_argument(
        "--eps",
        default=DEFAULT_EPS,
        type=float,
        help=f"RMSNorm epsilon. Default: {DEFAULT_EPS:g}.",
    )

    parser.add_argument(
        "--profiler-target",
        default="llaisys",
        choices=["llaisys", "torch"],
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
        help="Show the backend-resolved effective RMSNorm configuration.",
    )

    parser.add_argument(
        "--show-bandwidth",
        action="store_true",
        help="Show logical/equivalent RMSNorm I/O bandwidth.",
    )

    parser.add_argument(
        "--show-throughput",
        action="store_true",
        help="Show processed-element throughput in GElem/s.",
    )

    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip semantic and random correctness suites.",
    )

    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip only deterministic RMSNorm semantic cases.",
    )

    parser.add_argument(
        "--profile-suite",
        default="all",
        choices=["sweep", "llm", "all"],
        help="Performance workload suite: synthetic sweep, LLM shapes, or both.",
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

    parser.add_argument("--device-model", default=None, type=str)
    parser.add_argument("--device-architecture", default=None, type=str)
    parser.add_argument("--device-total-memory-gb", default=None, type=float)
    parser.add_argument("--device-partition", default=None, type=str)
    parser.add_argument("--device-partition-kind", default=None, type=str)
    parser.add_argument("--device-partition-mode", default=None, type=str)
    parser.add_argument("--device-partition-instance", default=None, type=str)
    parser.add_argument("--device-compute-partition", default=None, type=str)
    parser.add_argument("--device-memory-partition", default=None, type=str)
    parser.add_argument("--device-memory-limit-gb", default=None, type=float)
    parser.add_argument("--device-compute-fraction", default=None, type=float)
    parser.add_argument("--device-power-limit-w", default=None, type=float)
    parser.add_argument("--accelerator-runtime-version", default=None, type=str)
    parser.add_argument("--accelerator-driver-version", default=None, type=str)
    parser.add_argument("--accelerator-compiler-version", default=None, type=str)

    args = parser.parse_args()

    if args.backend == "triton" and args.device == "cpu":
        raise ValueError("Triton RMSNorm requires a GPU device")

    if args.eps <= 0.0:
        raise ValueError("--eps must be greater than zero")

    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    if args.repeat <= 0:
        raise ValueError("--repeat must be greater than zero")

    if args.rounds <= 0:
        raise ValueError("--rounds must be greater than zero")

    if args.profiler_warmup < 0:
        raise ValueError("--profiler-warmup must be non-negative")

    if args.profiler_launches <= 0:
        raise ValueError("--profiler-launches must be greater than zero")

    if args.profiler_mode and args.case_shape is None:
        raise ValueError("--case-shape is required with --profiler-mode")

    if (
        args.device_compute_fraction is not None
        and not 0.0 <= args.device_compute_fraction <= 1.0
    ):
        raise ValueError("--device-compute-fraction must be within [0, 1]")

    torch.manual_seed(args.seed)

    if args.device in ("nvidia", "amd") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ========================================================
    # DTypes / tolerances
    # ========================================================

    test_dtype_prec = [
        ("f32", 1e-5, 1e-5),
        ("f16", 2e-3, 2e-3),
        ("bf16", 2e-2, 2e-2),
    ]

    # ========================================================
    # Correctness workload suite
    #
    # RMSNorm is reduction-sensitive, so the most important
    # boundaries are normalized widths around powers of two and
    # around the current Triton warp-policy transition.
    #
    # 29 shapes x 3 dtypes = 87 random correctness cases.
    # Together with 5 semantic cases x 3 dtypes, the default
    # correctness run exercises 102 cases.
    # ========================================================

    correctness_shapes = [
        # Tiny.
        (1, 1),
        (1, 2),
        (2, 3),

        # 32 boundary.
        (1, 31),
        (1, 32),
        (1, 33),

        # 128 boundary.
        (1, 127),
        (1, 128),
        (1, 129),

        # 256 boundary.
        (1, 255),
        (1, 256),
        (1, 257),

        # 512 boundary.
        (1, 511),
        (1, 512),
        (1, 513),

        # 1024 boundary.
        (1, 1023),
        (1, 1024),
        (1, 1025),

        # Model-width neighborhood.
        (1, 1535),
        (1, 1536),
        (1, 1537),

        # Current portable Triton policy changes num_warps once the
        # next-power-of-two block exceeds 2048.
        (1, 2047),
        (1, 2048),
        (1, 2049),

        # Common wide hidden-size neighborhood.
        (1, 4095),
        (1, 4096),
        (1, 4097),

        # Multi-row representative cases.
        (16, 1536),
        (512, 4096),
    ]

    # ========================================================
    # Synthetic performance sweep
    #
    # Two dimensions matter:
    #
    #   1. reduction width / per-row work
    #   2. number of independent rows
    #
    # Width sweep:
    #   32, 128, 512, 2048, 2049, 8192
    #
    # Row-count sweep at width 4096:
    #   8, 64, 256, 1024
    # ========================================================

    sweep_shapes = [
        (1, 32),
        (1, 128),
        (1, 512),
        (1, 2048),
        (1, 2049),
        (1, 8192),
        (8, 4096),
        (64, 4096),
        (256, 4096),
        (1024, 4096),
    ]

    # ========================================================
    # LLM-representative RMSNorm shapes
    #
    # 1536 covers a smaller hidden width.
    # 4096 keeps direct comparability with Add/SwiGLU.
    # ========================================================

    llm_shapes = [
        (1, 1536),
        (32, 1536),
        (512, 1536),
        (1, 4096),
        (32, 4096),
        (512, 4096),
        (2048, 4096),
    ]

    backend_metadata = collect_backend_metadata(
        args.backend,
        args.device,
        variant=args.backend_variant,
        implementation=args.backend_implementation,
    )

    filename_config = get_rms_norm_output_filename_config(args.backend)

    if args.output is not None:
        output_path = args.output
    elif args.profile and not args.profiler_mode and not args.no_record:
        output_path = build_experiment_output_path(
            args.output_dir,
            op="rms_norm",
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
            "torch": "torch_nn_functional_rms_norm",
            "torch_output_preallocated": False,
            "llaisys_output_preallocated": True,
        },
        "input_distribution": {
            "x": "uniform[-2,2)",
            "weight": "uniform[-1,1)",
        },
        "eps": args.eps,
        "profiler_mode": args.profiler_mode,
        "profiler_case": {
            "shape": list(args.case_shape) if args.case_shape is not None else None,
            "dtype": args.case_dtype,
            "eps": args.eps,
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

    software_overrides = {}
    accelerator_stack_overrides = {}

    if args.accelerator_runtime_version is not None:
        accelerator_stack_overrides["runtime_version"] = (
            args.accelerator_runtime_version
        )

    if args.accelerator_driver_version is not None:
        accelerator_stack_overrides["driver_version"] = (
            args.accelerator_driver_version
        )

    if args.accelerator_compiler_version is not None:
        accelerator_stack_overrides["compiler"] = {
            "version": args.accelerator_compiler_version,
        }

    if accelerator_stack_overrides:
        software_overrides["accelerator_stack"] = accelerator_stack_overrides

    recorder = BenchmarkRecorder(
        output_path=output_path,
        repo_root=REPO_ROOT,
        run_id=args.run_id,
        run_metadata=run_metadata,
        software_overrides=software_overrides,
    )

    device_metadata = {}

    if args.device_model is not None:
        device_metadata["model"] = args.device_model

    if args.device_architecture is not None:
        device_metadata["architecture"] = args.device_architecture

    if args.device_total_memory_gb is not None:
        device_metadata["total_memory_bytes"] = int(
            args.device_total_memory_gb * 1_000_000_000
        )

    partition_metadata = {}

    if args.device_partition_kind is not None:
        partition_metadata["kind"] = args.device_partition_kind

    if args.device_partition_mode is not None:
        partition_metadata["mode"] = args.device_partition_mode

    if args.device_partition_instance is not None:
        partition_metadata["instance"] = args.device_partition_instance

    if args.device_compute_partition is not None:
        partition_metadata["compute_partition"] = args.device_compute_partition

    if args.device_memory_partition is not None:
        partition_metadata["memory_partition"] = args.device_memory_partition

    if args.device_partition is not None:
        partition_metadata["description"] = args.device_partition

    if partition_metadata:
        device_metadata["partition"] = partition_metadata

    resource_limits = {}

    if args.device_memory_limit_gb is not None:
        resource_limits["memory_limit_bytes"] = int(
            args.device_memory_limit_gb * 1_000_000_000
        )

    if args.device_compute_fraction is not None:
        resource_limits["compute_fraction"] = args.device_compute_fraction

    if args.device_power_limit_w is not None:
        resource_limits["power_limit_w"] = args.device_power_limit_w

    if resource_limits:
        device_metadata["resource_limits"] = resource_limits

    if args.profiler_mode:
        dtype_tolerance = {
            dtype_name: (atol, rtol)
            for dtype_name, atol, rtol in test_dtype_prec
        }
        atol, rtol = dtype_tolerance[args.case_dtype]

        print(
            f"Profiling Ops.rms_norm "
            f"on {args.device} "
            f"with {args.backend} backend"
        )
        print(
            "Backend identity: "
            f"name={backend_metadata['name']}, "
            f"implementation={backend_metadata['implementation']}, "
            f"variant={backend_metadata['variant']}"
        )
        print(f"Random seed: {args.seed}")
        print(
            "Profiler protocol: "
            f"target={args.profiler_target}, "
            f"warmup={args.profiler_warmup}, "
            f"launches={args.profiler_launches}, "
            f"post_check={args.profiler_check}"
        )

        run_rms_norm_profiler_case(
            shape=args.case_shape,
            dtype_name=args.case_dtype,
            atol=atol,
            rtol=rtol,
            eps=args.eps,
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
        print("\033[92mProfiler case completed!\033[0m")
        raise SystemExit(0)

    print(
        f"Testing Ops.rms_norm "
        f"on {args.device} "
        f"with {args.backend} backend"
    )
    print(
        "Backend identity: "
        f"name={backend_metadata['name']}, "
        f"implementation={backend_metadata['implementation']}, "
        f"variant={backend_metadata['variant']}"
    )
    print(f"Random seed: {args.seed}")
    print(
        f"Benchmark protocol: "
        f"warmup={args.warmup}, "
        f"repeat={args.repeat}, "
        f"rounds={args.rounds}, "
        f"order={args.benchmark_order}"
    )
    print(f"RMSNorm epsilon: {args.eps:g}")
    print(f"Using llaisys from: {llaisys.__file__}")

    if recorder.enabled:
        print(f"Recording JSONL: {recorder.output_path}")
        print(f"Run ID: {recorder.run_id}")

    if not args.skip_correctness:
        if not args.skip_semantic:
            print()
            print("=== Correctness: deterministic semantics ===")

            for dtype_name, atol, rtol in test_dtype_prec:
                run_rms_norm_semantic_tests(
                    dtype_name,
                    atol,
                    rtol,
                    args.device,
                    args.backend,
                    args.eps,
                )

        print()
        print("=== Correctness: shape/range coverage ===")

        for shape in correctness_shapes:
            for dtype_name, atol, rtol in test_dtype_prec:
                test_op_rms_norm(
                    shape,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    eps=args.eps,
                    device_name=args.device,
                    backend=args.backend,
                    backend_variant=args.backend_variant,
                    backend_implementation=args.backend_implementation,
                    profile=False,
                    seed=args.seed,
                    recorder=recorder,
                    device_metadata=device_metadata,
                )

    if args.profile:
        print()
        print("=== Performance ===")

        profile_cases = []

        if args.profile_suite in ("sweep", "all"):
            profile_cases.extend(("sweep", shape) for shape in sweep_shapes)

        if args.profile_suite in ("llm", "all"):
            profile_cases.extend(("llm", shape) for shape in llm_shapes)

        for suite, shape in profile_cases:
            for dtype_name, atol, rtol in test_dtype_prec:
                test_op_rms_norm(
                    shape,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    eps=args.eps,
                    device_name=args.device,
                    backend=args.backend,
                    backend_variant=args.backend_variant,
                    backend_implementation=args.backend_implementation,
                    profile=True,
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
                    device_metadata=device_metadata,
                )

    print()
    print("\033[92mTest passed!\033[0m")
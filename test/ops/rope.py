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

import llaisys

from llaisys.triton import execution_context
from llaisys.triton.backends.registry import get_triton_backend
from llaisys.triton.ops import rope as triton_rope

from test_utils import (
    BenchmarkRecorder,
    arrange_tensor,
    benchmark,
    build_experiment_output_path,
    check_equal,
    collect_backend_metadata,
    random_tensor,
)


# ============================================================
# Constants
# ============================================================

DTYPE_BYTES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
}

POSITION_ID_BYTES = 8
DEFAULT_THETA = 10000.0


# ============================================================
# PyTorch reference / performance baseline
# ============================================================
#
# PyTorch does not expose a torch.nn.functional.rope operator that
# directly matches the LLAISYS half-split RoPE contract.
#
# Keep the established vectorized PyTorch reference:
#
#     first half  = x_a * cos - x_b * sin
#     second half = x_b * cos + x_a * sin
#
# The output tensor is preallocated, matching the LLAISYS API.
#
# IMPORTANT:
# This is a vectorized eager PyTorch expression and can expand into
# multiple GPU kernels. It is therefore a correctness reference and a
# framework-level eager baseline, not a claim that Torch provides one
# fused RoPE kernel.
# ============================================================


def torch_rope(y: torch.Tensor, x: torch.Tensor, pos_ids: torch.Tensor, theta: float):
    if y.dim() != 3 or x.dim() != 3:
        raise ValueError("RoPE Torch reference requires 3D input/output tensors")

    if y.shape != x.shape:
        raise ValueError("RoPE Torch reference output shape must match input shape")

    seq_len, _, head_dim = x.shape

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("RoPE head dimension must be positive and even")

    if pos_ids.numel() != seq_len:
        raise ValueError("RoPE position-id count must match sequence length")

    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("RoPE theta must be finite and greater than zero")

    half_dim = head_dim // 2

    x_a = x[..., :half_dim]
    x_b = x[..., half_dim:]

    positions = pos_ids.to(torch.float32).reshape(seq_len, 1)

    pair_index = torch.arange(
        half_dim,
        dtype=torch.float32,
        device=x.device,
    )

    exponent = 2.0 * pair_index / float(head_dim)
    denominator = theta**exponent
    freqs = positions / denominator

    sine = torch.sin(freqs).unsqueeze(1)
    cosine = torch.cos(freqs).unsqueeze(1)

    y[..., :half_dim] = x_a * cosine - x_b * sine
    y[..., half_dim:] = x_b * cosine + x_a * sine


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_rope(out, x, pos_ids, theta, backend):
    if backend == "native":
        llaisys.Ops.rope(out, x, pos_ids, theta)
        return

    if backend == "triton":
        triton_rope(out, x, pos_ids, theta)
        return

    raise ValueError(f"Unsupported RoPE backend: {backend}")


# ============================================================
# Effective configuration
# ============================================================
#
# Triton RoPE currently resolves configuration from head_dim.
# The portable backend policy uses:
#
#     BLOCK_SIZE = 128
#     num_warps  = 4
#
# via backend.rope_config(head_dim).
#
# Native RoPE launch policy is backend-specific. We record generic
# block overrides and the known MetaX direct/cached runtime control.
# ============================================================


def _parse_env_config_value(name, default="default"):
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return value


def get_rope_config(tensor, backend, device_name):
    shape = tensor.shape()

    if len(shape) != 3:
        raise ValueError("RoPE configuration requires a 3D tensor")

    _, head_count, head_dim = shape

    if backend == "native":
        config = {
            "BLOCK_SIZE": _parse_env_config_value("LLAISYS_BLOCK_SIZE"),
        }

        if device_name == "metax":
            config["ROPE_CACHE"] = os.getenv(
                "LLAISYS_METAX_ROPE_CACHE",
                "auto",
            )

        return "requested_or_backend_policy", config

    if backend == "triton":
        triton_backend = get_triton_backend(tensor.device_type())
        config = triton_backend.rope_config(head_dim)

        return "effective", {
            "BLOCK_SIZE": config["BLOCK_SIZE"],
            "num_warps": config["num_warps"],
            "head_count": head_count,
            "head_dim": head_dim,
        }

    raise ValueError(f"Unsupported RoPE backend: {backend}")


def get_rope_config_label(tensor, backend, device_name):
    _, config = get_rope_config(tensor, backend, device_name)
    values = ", ".join(f"{key}={value}" for key, value in config.items())
    return f"config[{values}]"


def get_rope_output_filename_config(backend, device_name):
    if backend == "native":
        config = {
            "BLOCK_SIZE": _parse_env_config_value("LLAISYS_BLOCK_SIZE"),
        }

        if device_name == "metax":
            config["ROPE_CACHE"] = os.getenv(
                "LLAISYS_METAX_ROPE_CACHE",
                "auto",
            )

        return config

    if backend == "triton":
        return {
            "BLOCK_SIZE": _parse_env_config_value(
                "LLAISYS_TRITON_BLOCK_SIZE"
            ),
            "NUM_WARPS": _parse_env_config_value(
                "LLAISYS_TRITON_NUM_WARPS"
            ),
        }

    raise ValueError(f"Unsupported RoPE backend: {backend}")


# ============================================================
# Derived performance metrics
# ============================================================
#
# Minimum logical I/O for X[S,H,D], POS[S], OUT[S,H,D]:
#
#     read X      -> S * H * D * element_size
#     read POS    -> S * sizeof(int64)
#     write OUT   -> S * H * D * element_size
#
# The trigonometric calculations are compute work and are not represented
# by the byte model below. These bandwidth values are logical/effective,
# not hardware DRAM-counter measurements.
#
# The eager Torch reference can materialize frequency/trigonometric
# intermediates, so its value is explicitly called equivalent I/O
# bandwidth.
# ============================================================


def get_rope_nominal_io_traffic_bytes(shape, dtype_name):
    seq_len, head_count, head_dim = shape
    numel = seq_len * head_count * head_dim

    input_bytes = numel * DTYPE_BYTES[dtype_name]
    position_bytes = seq_len * POSITION_ID_BYTES
    output_bytes = numel * DTYPE_BYTES[dtype_name]

    return {
        "input_bytes": input_bytes,
        "position_bytes": position_bytes,
        "output_bytes": output_bytes,
        "total_bytes": input_bytes + position_bytes + output_bytes,
    }


def get_effective_bandwidth_gbs(traffic_bytes, median_ms):
    return traffic_bytes / median_ms / 1_000_000.0


def get_element_throughput_gelem_s(numel, median_ms):
    return numel / median_ms / 1_000_000.0


def get_rope_derived_metrics(stats, shape, dtype_name):
    traffic = get_rope_nominal_io_traffic_bytes(shape, dtype_name)
    numel = math.prod(shape)
    llaisys_stats = stats["llaisys"]
    torch_stats = stats.get("torch")

    derived = {
        "nominal_input_read_bytes": traffic["input_bytes"],
        "nominal_position_read_bytes": traffic["position_bytes"],
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


def print_rope_derived_metrics(
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
# Case validation
# ============================================================


def validate_rope_case(shape, start_end, theta):
    if len(shape) != 3:
        raise ValueError(f"RoPE test shape must be 3D, got {shape}")

    seq_len, head_count, head_dim = shape

    if seq_len <= 0:
        raise ValueError("RoPE sequence length must be greater than zero")

    if head_count <= 0:
        raise ValueError("RoPE head count must be greater than zero")

    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("RoPE head dimension must be positive and even")

    if len(start_end) != 2:
        raise ValueError("RoPE position range must be a (start, end) pair")

    start, end = start_end

    if end - start != seq_len:
        raise ValueError(
            "RoPE position range length must match sequence length: "
            f"shape={shape}, range={start_end}"
        )

    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("RoPE theta must be finite and greater than zero")


# ============================================================
# One random correctness / performance case
# ============================================================


def test_op_rope(
    shape,
    start_end,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    theta=DEFAULT_THETA,
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
    validate_rope_case(shape, start_end, theta)

    if recorder is None:
        recorder = BenchmarkRecorder()

    seq_len, head_count, head_dim = shape
    half_dim = head_dim // 2
    numel = math.prod(shape)

    print(
        f"   shape {shape} "
        f"seq_len {seq_len} "
        f"heads {head_count} "
        f"head_dim {head_dim} "
        f"half_dim {half_dim} "
        f"range {start_end} "
        f"theta <{theta:g}> "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    x_ref, x = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    pos_ref, pos = arrange_tensor(
        start_end[0],
        start_end[1],
        device_name,
    )

    y_ref, y = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    torch_rope(
        y_ref,
        x_ref,
        pos_ref,
        theta,
    )

    run_llaisys_rope(
        y,
        x,
        pos,
        theta,
        backend,
    )

    assert check_equal(y, y_ref, atol=atol, rtol=rtol), (
        f"RoPE mismatch: "
        f"shape={shape}, range={start_end}, theta={theta}, "
        f"dtype={dtype_name}, device={device_name}, backend={backend}"
    )

    if not profile:
        return

    config_status, config = get_rope_config(
        y,
        backend,
        device_name,
    )

    label = (
        f"RoPE shape={shape} "
        f"seq_len={seq_len} "
        f"heads={head_count} "
        f"head_dim={head_dim} "
        f"range={start_end} "
        f"theta={theta:g} "
        f"dtype={dtype_name} "
        f"backend={backend}"
    )

    if show_config:
        label += f" {get_rope_config_label(y, backend, device_name)}"

    print(f"        {label}:")

    torch_fn = lambda: torch_rope(
        y_ref,
        x_ref,
        pos_ref,
        theta,
    )

    llaisys_fn = lambda: run_llaisys_rope(
        y,
        x,
        pos,
        theta,
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
            y.device_type(),
            y.device_id(),
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
        raise ValueError(f"Unsupported RoPE backend: {backend}")

    derived = get_rope_derived_metrics(
        stats,
        shape,
        dtype_name,
    )

    if show_bandwidth or show_throughput:
        print_rope_derived_metrics(
            derived,
            device_name,
            show_bandwidth,
            show_throughput,
        )

    recorder.record_microbenchmark(
        op="rope",
        backend_name=backend,
        backend_variant=backend_variant,
        backend_implementation=backend_implementation,
        suite=suite,
        device_name=device_name,
        device_id=y.device_id(),
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
            "sequence_length": seq_len,
            "head_count": head_count,
            "head_dim": head_dim,
            "half_dim": half_dim,
            "position_start": start_end[0],
            "position_end": start_end[1],
            "position_dtype": "i64",
            "theta": theta,
            "torch_reference": "vectorized_eager_half_split_rope",
            "torch_output_preallocated": True,
            "llaisys_output_preallocated": True,
            "input_range": [-1.0, 1.0],
        },
        device_metadata=device_metadata,
    )


# ============================================================
# Deterministic semantic / contract-sensitive cases
# ============================================================


def test_rope_zero_position_identity(
    dtype_name,
    atol,
    rtol,
    device_name,
    backend,
    theta,
):
    shape = (1, 3, 128)
    start_end = (0, 1)

    print(
        f"   semantic zero_position_identity "
        f"shape {shape} range {start_end} "
        f"theta <{theta:g}> dtype <{dtype_name}> "
        f"device <{device_name}> backend <{backend}>"
    )

    x_ref, x = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    pos_ref, pos = arrange_tensor(0, 1, device_name)

    y_ref, y = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    torch_rope(y_ref, x_ref, pos_ref, theta)
    run_llaisys_rope(y, x, pos, theta, backend)

    assert check_equal(y, y_ref, atol=atol, rtol=rtol), (
        f"RoPE zero-position reference mismatch: "
        f"dtype={dtype_name}, device={device_name}, backend={backend}"
    )

    assert check_equal(y, x_ref, atol=atol, rtol=rtol), (
        f"RoPE position 0 must be identity: "
        f"dtype={dtype_name}, device={device_name}, backend={backend}"
    )


def test_rope_inplace(
    dtype_name,
    atol,
    rtol,
    device_name,
    backend,
    theta,
):
    shape = (3, 4, 128)
    start_end = (17, 20)

    print(
        f"   semantic exact_inplace "
        f"shape {shape} range {start_end} "
        f"theta <{theta:g}> dtype <{dtype_name}> "
        f"device <{device_name}> backend <{backend}>"
    )

    x_ref, x = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    pos_ref, pos = arrange_tensor(
        start_end[0],
        start_end[1],
        device_name,
    )

    y_ref = torch.empty_like(x_ref)
    torch_rope(
        y_ref,
        x_ref,
        pos_ref,
        theta,
    )

    # Exact alias: out == input.
    run_llaisys_rope(
        x,
        x,
        pos,
        theta,
        backend,
    )

    assert check_equal(x, y_ref, atol=atol, rtol=rtol), (
        f"RoPE exact in-place mismatch: "
        f"dtype={dtype_name}, device={device_name}, backend={backend}"
    )


def run_rope_semantic_tests(
    dtype_name,
    atol,
    rtol,
    device_name,
    backend,
    default_theta,
):
    test_rope_zero_position_identity(
        dtype_name,
        atol,
        rtol,
        device_name,
        backend,
        default_theta,
    )

    test_rope_inplace(
        dtype_name,
        atol,
        rtol,
        device_name,
        backend,
        default_theta,
    )

    # Theta coverage. Keep these small so the default correctness run
    # exercises the frequency formula without becoming expensive.
    theta_cases = [
        ("theta_one", 1.0),
        ("theta_default", 10000.0),
        ("theta_large", 1_000_000.0),
    ]

    for name, theta in theta_cases:
        print(
            f"   semantic {name} "
            f"shape (3, 2, 64) range (7, 10) "
            f"theta <{theta:g}> dtype <{dtype_name}> "
            f"device <{device_name}> backend <{backend}>"
        )

        test_op_rope(
            (3, 2, 64),
            (7, 10),
            dtype_name=dtype_name,
            atol=atol,
            rtol=rtol,
            theta=theta,
            device_name=device_name,
            backend=backend,
            profile=False,
        )


# ============================================================
# Profiler helpers
# ============================================================


def parse_case_shape(value):
    text = value.strip().lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]

    if not parts:
        raise argparse.ArgumentTypeError(
            "RoPE shape must contain at least one dimension"
        )

    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid RoPE shape {value!r}"
        ) from exc

    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(
            "all RoPE shape dimensions must be greater than zero"
        )

    # Convenience:
    #   128       -> (1, 1, 128)
    #   12,128    -> (1, 12, 128)
    #   512,12,128 -> unchanged
    if len(shape) == 1:
        shape = (1, 1, shape[0])
    elif len(shape) == 2:
        shape = (1, shape[0], shape[1])
    elif len(shape) != 3:
        raise argparse.ArgumentTypeError(
            "RoPE shape must be HEAD_DIM, HEADS,HEAD_DIM, "
            "or SEQ,HEADS,HEAD_DIM"
        )

    if shape[2] % 2 != 0:
        raise argparse.ArgumentTypeError(
            "RoPE head dimension must be even"
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


def run_rope_profiler_case(
    *,
    shape,
    position_start,
    theta,
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
    seq_len, head_count, head_dim = shape
    start_end = (
        position_start,
        position_start + seq_len,
    )
    numel = math.prod(shape)

    validate_rope_case(shape, start_end, theta)

    print()
    print("=== Profiler single case ===")
    print(
        f"   target <{profiler_target}> "
        f"shape {shape} "
        f"range {start_end} "
        f"theta <{theta:g}> "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    x_ref, x = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    pos_ref, pos = arrange_tensor(
        start_end[0],
        start_end[1],
        device_name,
    )

    y_ref, y = random_tensor(
        shape,
        dtype_name,
        device_name,
        scale=2.0,
        bias=-1.0,
    )

    if profiler_target == "torch":
        if device_name == "metax":
            raise ValueError(
                "Torch profiler target is unavailable for MetaX because "
                "the current MetaX reference tensors are hosted on CPU."
            )

        target_fn = lambda: torch_rope(
            y_ref,
            x_ref,
            pos_ref,
            theta,
        )
        synchronize = lambda: _torch_profiler_synchronize(device_name)
        config_status = "reference"
        config = {}

        target_label = (
            f"LLAISYS_PROFILE:rope:torch:{device_name}:"
            f"shape={'x'.join(str(dim) for dim in shape)}:"
            f"pos={position_start}:theta={theta:g}:dtype={dtype_name}"
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
            run_llaisys_rope(
                y,
                x,
                pos,
                theta,
                backend,
            )

            assert check_equal(y, y_ref, atol=atol, rtol=rtol), (
                f"RoPE profiler correctness mismatch: "
                f"shape={shape}, range={start_end}, dtype={dtype_name}, "
                f"device={device_name}, backend={backend}"
            )

        print(
            "Profiler note: Torch RoPE is a vectorized eager expression and "
            "can expand into multiple kernels. Use a timeline profiler for "
            "the whole expression instead of assuming one Torch launch."
        )

    else:
        config_status, config = get_rope_config(
            y,
            backend,
            device_name,
        )

        if show_config:
            print(
                f"        "
                f"{get_rope_config_label(y, backend, device_name)}"
            )

        target_fn = lambda: run_llaisys_rope(
            y,
            x,
            pos,
            theta,
            backend,
        )

        api = llaisys.RuntimeAPI(y.device_type())
        synchronize = api.device_synchronize

        config_tag = ",".join(
            f"{key}={value}"
            for key, value in config.items()
        )

        target_label = (
            f"LLAISYS_PROFILE:rope:{backend}:{backend_variant}:{device_name}:"
            f"shape={'x'.join(str(dim) for dim in shape)}:"
            f"pos={position_start}:theta={theta:g}:dtype={dtype_name}:"
            f"{config_tag}"
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
                _end_profiler_range(range_pushed)

        if backend == "triton":
            with execution_context(
                y.device_type(),
                y.device_id(),
            ):
                execute_target()
        else:
            execute_target()

        if profiler_check:
            torch_rope(
                y_ref,
                x_ref,
                pos_ref,
                theta,
            )
            _torch_profiler_synchronize(device_name)

            assert check_equal(y, y_ref, atol=atol, rtol=rtol), (
                f"RoPE profiler correctness mismatch: "
                f"shape={shape}, range={start_end}, dtype={dtype_name}, "
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
                f"rope_kernel; use --kernel-name rope_kernel "
                f"--launch-skip {profiler_warmup} "
                f"--launch-count {profiler_launches} after confirming the name."
            )
        else:
            print(
                "Profiler hint: Native RoPE may select direct/cached kernels. "
                "First run discovery profiling, confirm the selected kernel "
                "name, then apply an exact kernel-name filter."
            )

    if profiler_check:
        print("Profiler post-check: passed")

    return {
        "target": profiler_target,
        "shape": shape,
        "sequence_length": seq_len,
        "head_count": head_count,
        "head_dim": head_dim,
        "position_start": position_start,
        "position_end": start_end[1],
        "theta": theta,
        "dtype": dtype_name,
        "config_status": config_status,
        "config": config,
        "warmup": profiler_warmup,
        "launches": profiler_launches,
        "range": target_label,
        "numel": numel,
    }


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
        help="Run the paper-oriented RoPE microbenchmark suite.",
    )

    execution_mode.add_argument(
        "--profiler-mode",
        action="store_true",
        help=(
            "Run one controlled RoPE workload for external profilers. "
            "This mode does not run the normal benchmark loop or write "
            "microbenchmark JSONL."
        ),
    )

    parser.add_argument(
        "--case-shape",
        default=None,
        type=parse_case_shape,
        help=(
            "Profiler shape. Examples: 128 -> (1,1,128), "
            "12,128 -> (1,12,128), 512,12,128."
        ),
    )

    parser.add_argument(
        "--case-position-start",
        default=512,
        type=int,
        help="Starting position ID for --profiler-mode. Default: 512.",
    )

    parser.add_argument(
        "--case-dtype",
        default="f16",
        choices=["f32", "f16", "bf16"],
        type=str,
    )

    parser.add_argument(
        "--theta",
        default=DEFAULT_THETA,
        type=float,
        help=f"RoPE theta. Default: {DEFAULT_THETA:g}.",
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
        help="Show the effective/requested RoPE configuration.",
    )

    parser.add_argument(
        "--show-bandwidth",
        action="store_true",
        help="Show logical/equivalent RoPE I/O bandwidth.",
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
        help="Skip deterministic/contract-sensitive RoPE semantic cases.",
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

    # --------------------------------------------------------
    # Cross-vendor device metadata overrides
    # --------------------------------------------------------

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

    # ========================================================
    # Argument validation
    # ========================================================

    if args.backend == "triton" and args.device == "cpu":
        raise ValueError("Triton RoPE requires a GPU device")

    if not math.isfinite(args.theta) or args.theta <= 0.0:
        raise ValueError("--theta must be finite and greater than zero")

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
    # DTypes / established tolerances
    # ========================================================

    test_dtype_prec = [
        ("f32", 2e-4, 1e-4),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    # ========================================================
    # Correctness workload suite
    #
    # Important boundaries:
    #
    #   - minimal even head dimensions
    #   - half-dim 64 neighborhood: head_dim 126/128/130
    #   - Triton half-dim tile 128 neighborhood: 254/256/258
    #   - larger head dimensions
    #   - sequence/head-count variation
    #   - large position IDs
    #   - 4096 head-dim stress neighborhood
    #
    # Every range length exactly equals seq_len.
    # ========================================================

    correctness_cases = [
        # Tiny / minimal.
        ((1, 1, 2), (0, 1)),
        ((2, 1, 4), (0, 2)),
        ((3, 2, 6), (7, 10)),

        # half_dim ~= 64.
        ((2, 3, 126), (17, 19)),
        ((2, 3, 128), (17, 19)),
        ((2, 3, 130), (17, 19)),

        # Triton half-dim tile boundary: half_dim ~= 128.
        ((2, 3, 254), (64, 66)),
        ((2, 3, 256), (64, 66)),
        ((2, 3, 258), (64, 66)),

        # Sequence-length coverage.
        ((1, 12, 128), (512, 513)),
        ((31, 4, 128), (512, 543)),
        ((32, 4, 128), (512, 544)),
        ((33, 4, 128), (512, 545)),

        # Head-count coverage.
        ((4, 1, 128), (32, 36)),
        ((4, 12, 128), (32, 36)),
        ((4, 32, 128), (32, 36)),

        # Large position IDs.
        ((3, 4, 128), (8192, 8195)),
        ((3, 4, 256), (32768, 32771)),

        # Large head dimensions without making correctness too expensive.
        ((2, 4, 1024), (512, 514)),
        ((1, 2, 4094), (512, 513)),
        ((1, 2, 4096), (512, 513)),
        ((1, 2, 4098), (512, 513)),
    ]

    # ========================================================
    # Synthetic performance sweep
    #
    # Separately varies:
    #
    #   1. head_dim / per-vector work
    #   2. sequence length / independent token count
    #
    # 126/128/130 exposes the small cached-policy neighborhood used
    # in prior MetaX RoPE work.
    #
    # 254/256/258 crosses the current Triton half-dimension tile.
    # ========================================================

    sweep_cases = [
        # Head-dimension sweep, decode-shaped.
        ((1, 12, 64), 512),
        ((1, 12, 126), 512),
        ((1, 12, 128), 512),
        ((1, 12, 130), 512),
        ((1, 12, 254), 512),
        ((1, 12, 256), 512),
        ((1, 12, 258), 512),
        ((1, 12, 512), 512),
        ((1, 4, 4096), 512),

        # Sequence-length sweep at common head_dim=128.
        ((32, 12, 128), 512),
        ((128, 12, 128), 512),
        ((512, 12, 128), 512),
        ((2048, 12, 128), 512),
    ]

    # ========================================================
    # LLM-representative shapes
    #
    # Keep the historical 12-head / 128-dim cases for continuity,
    # and add 32-head cases plus head_dim=256.
    # ========================================================

    llm_cases = [
        ((1, 12, 128), 512),
        ((32, 12, 128), 512),
        ((512, 12, 128), 512),
        ((1, 32, 128), 512),
        ((32, 32, 128), 512),
        ((512, 32, 128), 512),
        ((1, 32, 256), 512),
        ((512, 32, 256), 512),

        # Historical large-d stress case from prior RoPE work.
        ((512, 4, 4096), 512),
    ]

    # ========================================================
    # Experiment metadata
    # ========================================================

    backend_metadata = collect_backend_metadata(
        args.backend,
        args.device,
        variant=args.backend_variant,
        implementation=args.backend_implementation,
    )

    filename_config = get_rope_output_filename_config(
        args.backend,
        args.device,
    )

    if args.output is not None:
        output_path = args.output

    elif args.profile and not args.profiler_mode and not args.no_record:
        output_path = build_experiment_output_path(
            args.output_dir,
            op="rope",
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
            "torch": "vectorized_eager_half_split_rope",
            "torch_output_preallocated": True,
            "llaisys_output_preallocated": True,
        },
        "input_distribution": "uniform[-1,1)",
        "theta": args.theta,
        "profiler_mode": args.profiler_mode,
        "profiler_case": {
            "shape": list(args.case_shape)
            if args.case_shape is not None
            else None,
            "position_start": args.case_position_start,
            "dtype": args.case_dtype,
            "theta": args.theta,
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
        software_overrides["accelerator_stack"] = (
            accelerator_stack_overrides
        )

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

    # ========================================================
    # Profiler mode
    # ========================================================

    if args.profiler_mode:
        dtype_tolerance = {
            dtype_name: (atol, rtol)
            for dtype_name, atol, rtol in test_dtype_prec
        }
        atol, rtol = dtype_tolerance[args.case_dtype]

        print(
            f"Profiling Ops.rope "
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

        run_rope_profiler_case(
            shape=args.case_shape,
            position_start=args.case_position_start,
            theta=args.theta,
            dtype_name=args.case_dtype,
            atol=atol,
            rtol=rtol,
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

    # ========================================================
    # Standard correctness / benchmark run
    # ========================================================

    print(
        f"Testing Ops.rope "
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
        "Benchmark protocol: "
        f"warmup={args.warmup}, "
        f"repeat={args.repeat}, "
        f"rounds={args.rounds}, "
        f"order={args.benchmark_order}"
    )
    print(f"RoPE theta: {args.theta:g}")
    print(f"Using llaisys from: {llaisys.__file__}")

    if recorder.enabled:
        print(f"Recording JSONL: {recorder.output_path}")
        print(f"Run ID: {recorder.run_id}")

    if not args.skip_correctness:
        if not args.skip_semantic:
            print()
            print("=== Correctness: semantic / contract-sensitive cases ===")

            for dtype_name, atol, rtol in test_dtype_prec:
                run_rope_semantic_tests(
                    dtype_name,
                    atol,
                    rtol,
                    args.device,
                    args.backend,
                    args.theta,
                )

        print()
        print("=== Correctness: shape / boundary coverage ===")

        for shape, start_end in correctness_cases:
            for dtype_name, atol, rtol in test_dtype_prec:
                test_op_rope(
                    shape,
                    start_end,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    theta=args.theta,
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
            profile_cases.extend(
                ("sweep", shape, position_start)
                for shape, position_start in sweep_cases
            )

        if args.profile_suite in ("llm", "all"):
            profile_cases.extend(
                ("llm", shape, position_start)
                for shape, position_start in llm_cases
            )

        for suite, shape, position_start in profile_cases:
            start_end = (
                position_start,
                position_start + shape[0],
            )

            for dtype_name, atol, rtol in test_dtype_prec:
                test_op_rope(
                    shape,
                    start_end,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    theta=args.theta,
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
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
from llaisys.triton.ops import swiglu as triton_swiglu

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

DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2}


# ============================================================
# PyTorch reference / performance baseline
# ============================================================
#
# PyTorch does not expose a torch.nn.functional.swiglu operator in the
# same style as rms_norm, so use the framework SiLU operator directly:
#
#     SwiGLU(gate, up) = SiLU(gate) * up
#
# Keep the LLAISYS numerical contract by promoting gate/up to FP32 for
# the activation and multiplication, then cast once to the input dtype.
#
# The benchmark times this functional expression itself and does not add
# an extra output copy inside the timed Torch function.
# ============================================================


def torch_swiglu(gate, up):
    gate_f32 = gate.float()
    up_f32 = up.float()
    return (torch.nn.functional.silu(gate_f32) * up_f32).to(gate.dtype)


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_swiglu(out, gate, up, backend):
    if backend == "native":
        llaisys.Ops.swiglu(out, gate, up)
        return

    if backend == "triton":
        triton_swiglu(out, gate, up)
        return

    raise ValueError(f"Unsupported SwiGLU backend: {backend}")


# ============================================================
# Effective configuration
# ============================================================


def get_swiglu_config(tensor, backend):
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
        numel = math.prod(tensor.shape())
        config = triton_backend.swiglu_config(numel)

        return "effective", {"BLOCK_SIZE": config["BLOCK_SIZE"], "num_warps": config["num_warps"]}

    raise ValueError(f"Unsupported SwiGLU backend: {backend}")


def get_swiglu_config_label(tensor, backend):
    status, config = get_swiglu_config(tensor, backend)

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


def get_swiglu_output_filename_config(backend):
    if backend == "native":
        return {"BLOCK_SIZE": _parse_env_config_value("LLAISYS_BLOCK_SIZE")}

    if backend == "triton":
        return {
            "BLOCK_SIZE": _parse_env_config_value("LLAISYS_TRITON_BLOCK_SIZE"),
            "NUM_WARPS": _parse_env_config_value("LLAISYS_TRITON_NUM_WARPS"),
        }

    raise ValueError(f"Unsupported SwiGLU backend: {backend}")


# ============================================================
# Derived performance metrics
# ============================================================
#
# A fused SwiGLU kernel has minimum logical I/O traffic:
#
#     read gate
#     read up
#     write out
#
# Therefore nominal I/O traffic is 3 * numel * element_size.
#
# The Torch functional baseline can still materialize FP32 intermediates and
# therefore moves more data than the minimum fused logical traffic. The Torch
# value below is explicitly called equivalent I/O bandwidth rather than
# measured DRAM bandwidth.
# ============================================================


def get_swiglu_nominal_io_traffic_bytes(numel, dtype_name):
    return 3 * numel * DTYPE_BYTES[dtype_name]


def get_effective_bandwidth_gbs(traffic_bytes, median_ms):
    return traffic_bytes / median_ms / 1_000_000.0


def get_element_throughput_gelem_s(numel, median_ms):
    return numel / median_ms / 1_000_000.0


def get_swiglu_derived_metrics(stats, numel, dtype_name):
    traffic_bytes = get_swiglu_nominal_io_traffic_bytes(numel, dtype_name)
    llaisys_stats = stats["llaisys"]
    torch_stats = stats.get("torch")

    derived = {
        "nominal_io_traffic_bytes": traffic_bytes,
        "llaisys_effective_io_bandwidth_gbs": get_effective_bandwidth_gbs(traffic_bytes, llaisys_stats["median_ms"]),
        "llaisys_element_throughput_gelem_s": get_element_throughput_gelem_s(numel, llaisys_stats["median_ms"]),
    }

    if torch_stats is not None:
        derived.update(
            {
                "torch_equivalent_io_bandwidth_gbs": get_effective_bandwidth_gbs(
                    traffic_bytes, torch_stats["median_ms"]
                ),
                "torch_element_throughput_gelem_s": get_element_throughput_gelem_s(numel, torch_stats["median_ms"]),
            }
        )

    return derived


def print_swiglu_derived_metrics(derived, device_name, show_bandwidth, show_throughput):
    if show_bandwidth:
        print(
            f"        LLAISYS {device_name} effective I/O bandwidth: "
            f"{derived['llaisys_effective_io_bandwidth_gbs']:.2f} GB/s"
        )

        torch_bandwidth = derived.get("torch_equivalent_io_bandwidth_gbs")
        if torch_bandwidth is not None:
            print(f"        Torch {device_name} equivalent I/O bandwidth: {torch_bandwidth:.2f} GB/s")

    if show_throughput:
        print(
            f"        LLAISYS {device_name} element throughput: "
            f"{derived['llaisys_element_throughput_gelem_s']:.3f} GElem/s"
        )

        torch_throughput = derived.get("torch_element_throughput_gelem_s")
        if torch_throughput is not None:
            print(f"        Torch {device_name} element throughput: {torch_throughput:.3f} GElem/s")


# ============================================================
# Deterministic semantic inputs
# ============================================================


def tensor_pair_from_values(values, dtype_name, device_name, device_id=0):
    torch_tensor = torch.tensor(
        values, dtype=torch_dtype(dtype_name), device=reference_torch_device(device_name, device_id)
    )

    llaisys_tensor = llaisys.Tensor(
        torch_tensor.shape, dtype=llaisys_dtype(dtype_name), device=llaisys_device(device_name), device_id=device_id
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    bytes_ = torch_tensor.numel() * torch_tensor.element_size()

    api.memcpy_sync(
        llaisys_tensor.data_ptr(), torch_tensor.data_ptr(), bytes_, torch_to_llaisys_memcpy_kind(device_name)
    )

    return torch_tensor, llaisys_tensor


def test_swiglu_semantics(dtype_name, atol, rtol, device_name, backend):
    print(f"   semantic saturation/sign cases dtype <{dtype_name}> device <{device_name}> backend <{backend}>")

    # SwiGLU's public operator contract requires two-dimensional tensors.
    # Use a 3 x 6 matrix so the semantic suite still covers large negative,
    # near-zero, positive, and saturating gate values without violating the
    # operator's dimensionality requirement.
    gate_values = [
        [-20.0, -12.0, -8.0, -4.0, -2.0, -1.0],
        [-0.25, -0.01, -0.0, 0.0, 0.01, 0.25],
        [1.0, 2.0, 4.0, 8.0, 12.0, 20.0],
    ]

    up_values = [
        [-2.0, -1.0, 0.0, 1.0, 2.0, -0.5],
        [0.5, -1.5, 1.5, -2.0, 2.0, -0.25],
        [0.25, -1.0, 1.0, -2.0, 2.0, 0.5],
    ]

    semantic_shape = (3, 6)

    gate_ref, gate = tensor_pair_from_values(gate_values, dtype_name, device_name)

    up_ref, up = tensor_pair_from_values(up_values, dtype_name, device_name)

    out_ref, out = zero_tensor(semantic_shape, dtype_name, device_name)

    out_ref.copy_(torch_swiglu(gate_ref, up_ref))
    run_llaisys_swiglu(out, gate, up, backend)

    assert check_equal(out, out_ref, atol=atol, rtol=rtol), (
        f"SwiGLU semantic mismatch: dtype={dtype_name}, device={device_name}, backend={backend}"
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
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}; expected forms such as 4096 or 2048,4096") from exc

    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("all shape dimensions must be greater than zero")

    # SwiGLU requires a 2D tensor. For profiler convenience, allow a single
    # number and interpret it as one row with that many features.
    if len(shape) == 1:
        return (1, shape[0])

    if len(shape) != 2:
        raise argparse.ArgumentTypeError("SwiGLU requires a 2D shape; use forms such as 4096, 1,4096, or 2048,4096")

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


def run_swiglu_profiler_case(
    *,
    shape,
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
    numel = math.prod(shape)

    print()
    print("=== Profiler single case ===")
    print(
        f"   target <{profiler_target}> "
        f"shape {shape} "
        f"numel {numel} "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    gate_ref, gate = random_tensor(shape, dtype_name, device_name, scale=16.0, bias=-8.0)

    up_ref, up = random_tensor(shape, dtype_name, device_name, scale=4.0, bias=-2.0)

    out_ref, out = zero_tensor(shape, dtype_name, device_name)

    if profiler_target == "torch":
        if device_name == "metax":
            raise ValueError(
                "Torch profiler target is unavailable for MetaX because the "
                "current MetaX reference tensor is hosted on CPU."
            )

        target_fn = lambda: torch_swiglu(gate_ref, up_ref)
        synchronize = lambda: _torch_profiler_synchronize(device_name)
        config_status = "reference"
        config = {}

        target_label = (
            f"LLAISYS_PROFILE:swiglu:torch:{device_name}:shape={'x'.join(str(dim) for dim in shape)}:dtype={dtype_name}"
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
            run_llaisys_swiglu(out, gate, up, backend)
            assert check_equal(out, out_ref, atol=atol, rtol=rtol), (
                f"SwiGLU profiler correctness mismatch: "
                f"shape={shape}, dtype={dtype_name}, "
                f"device={device_name}, backend={backend}"
            )

        print(
            "Profiler note: Torch target uses torch.nn.functional.silu(gate.float()) "
            "* up.float(), followed by the final dtype cast. This is still a small "
            "functional expression rather than a single fused torch.nn.functional "
            "SwiGLU operator, so use a timeline profiler to inspect its decomposition."
        )

    else:
        config_status, config = get_swiglu_config(out, backend)

        if show_config:
            print(f"        {get_swiglu_config_label(out, backend)}")

        target_fn = lambda: run_llaisys_swiglu(out, gate, up, backend)
        api = llaisys.RuntimeAPI(out.device_type())
        synchronize = api.device_synchronize

        config_tag = ",".join(f"{key}={value}" for key, value in config.items())
        target_label = (
            f"LLAISYS_PROFILE:swiglu:{backend}:{backend_variant}:{device_name}:"
            f"shape={'x'.join(str(dim) for dim in shape)}:dtype={dtype_name}:"
            f"{config_tag}"
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
            with execution_context(out.device_type(), out.device_id()):
                execute_target()
        else:
            execute_target()

        if profiler_check:
            out_ref.copy_(torch_swiglu(gate_ref, up_ref))
            _torch_profiler_synchronize(device_name)

            assert check_equal(out, out_ref, atol=atol, rtol=rtol), (
                f"SwiGLU profiler correctness mismatch: "
                f"shape={shape}, dtype={dtype_name}, "
                f"device={device_name}, backend={backend}"
            )

    print(f"Profiler target range: {target_label}")
    print(f"Profiler launches: warmup={profiler_warmup}, target={profiler_launches}")

    if profiler_target == "llaisys":
        if backend == "triton":
            print(
                "NCU hint: the Triton target kernel is expected to be "
                f"swiglu_kernel; use --kernel-name swiglu_kernel "
                f"--launch-skip {profiler_warmup} "
                f"--launch-count {profiler_launches} after confirming the name."
            )
        else:
            print(
                "NCU hint: first run a discovery profile to confirm the Native "
                "SwiGLU kernel name, then use a precise --kernel-name filter."
            )

    if profiler_check:
        print("Profiler post-check: passed")

    return {
        "target": profiler_target,
        "shape": shape,
        "numel": numel,
        "dtype": dtype_name,
        "config_status": config_status,
        "config": config,
        "warmup": profiler_warmup,
        "launches": profiler_launches,
        "range": target_label,
    }


# ============================================================
# Benchmark
# ============================================================


def benchmark_swiglu(
    torch_out,
    torch_gate,
    torch_up,
    llaisys_out,
    llaisys_gate,
    llaisys_up,
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
    numel = math.prod(shape)
    config_status, config = get_swiglu_config(llaisys_out, backend)

    label = f"SwiGLU shape={shape} numel={numel} dtype={dtype_name} backend={backend}"

    if show_config:
        label += f" {get_swiglu_config_label(llaisys_out, backend)}"

    print(f"        {label}:")

    torch_fn = lambda: torch_swiglu(torch_gate, torch_up)
    llaisys_fn = lambda: run_llaisys_swiglu(llaisys_out, llaisys_gate, llaisys_up, backend)

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
        with execution_context(llaisys_out.device_type(), llaisys_out.device_id()):
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
        raise ValueError(f"Unsupported SwiGLU backend: {backend}")

    derived = get_swiglu_derived_metrics(stats, numel, dtype_name)

    if show_bandwidth or show_throughput:
        print_swiglu_derived_metrics(derived, device_name, show_bandwidth, show_throughput)

    recorder.record_microbenchmark(
        op="swiglu",
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
            "torch_reference": "torch_functional_silu_fp32_mul",
            "gate_input_range": [-8.0, 8.0],
            "up_input_range": [-2.0, 2.0],
        },
        device_metadata=device_metadata,
    )


# ============================================================
# One random correctness/performance case
# ============================================================


def test_op_swiglu(
    shape,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
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
    if recorder is None:
        recorder = BenchmarkRecorder()

    numel = math.prod(shape)

    print(f"   shape {shape} numel {numel} dtype <{dtype_name}> device <{device_name}> backend <{backend}>")

    gate_ref, gate = random_tensor(shape, dtype_name, device_name, scale=16.0, bias=-8.0)

    up_ref, up = random_tensor(shape, dtype_name, device_name, scale=4.0, bias=-2.0)

    out_ref, out = zero_tensor(shape, dtype_name, device_name)

    out_ref.copy_(torch_swiglu(gate_ref, up_ref))
    run_llaisys_swiglu(out, gate, up, backend)

    assert check_equal(out, out_ref, atol=atol, rtol=rtol), (
        f"SwiGLU mismatch: shape={shape}, numel={numel}, dtype={dtype_name}, device={device_name}, backend={backend}"
    )

    if profile:
        benchmark_swiglu(
            out_ref,
            gate_ref,
            up_ref,
            out,
            gate,
            up,
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

    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia", "metax", "amd"], type=str)

    parser.add_argument("--backend", default="native", choices=["native", "triton"], type=str)

    parser.add_argument(
        "--backend-variant",
        default="unspecified",
        type=str,
        help=("Experiment variant label, for example baseline, tuned, autotuned, or vendor-specific."),
    )

    parser.add_argument(
        "--backend-implementation",
        default=None,
        type=str,
        help=("Optional implementation override. Native normally maps to cpu/cuda/maca/hip and Triton maps to triton."),
    )

    execution_mode = parser.add_mutually_exclusive_group()

    execution_mode.add_argument("--profile", action="store_true", help="Run the paper-oriented microbenchmark suite.")

    execution_mode.add_argument(
        "--profiler-mode",
        action="store_true",
        help=(
            "Run one controlled workload for ncu/nsys/mcProfiler/rocprof. "
            "This does not run the normal benchmark loop or write "
            "microbenchmark JSONL."
        ),
    )

    parser.add_argument(
        "--case-shape",
        default=None,
        type=parse_case_shape,
        help=(
            "Single profiler shape. A scalar such as 4096 is interpreted as "
            "(1, 4096); 2048,4096 and 2048x4096 are also accepted. "
            "Required with --profiler-mode."
        ),
    )

    parser.add_argument(
        "--case-dtype",
        default="f16",
        choices=["f32", "f16", "bf16"],
        type=str,
        help="Single profiler-case dtype. Default: f16.",
    )

    parser.add_argument(
        "--profiler-target",
        default="llaisys",
        choices=["llaisys", "torch"],
        type=str,
        help="Implementation executed inside the profiler target range.",
    )

    parser.add_argument(
        "--profiler-warmup", default=1, type=int, help="Target warmup invocations before the profiler range."
    )

    parser.add_argument(
        "--profiler-launches",
        default=1,
        type=int,
        help="Exact number of target operator invocations inside the profiler range.",
    )

    parser.add_argument(
        "--profiler-check", action="store_true", help="Validate correctness after the profiled target range."
    )

    parser.add_argument("--show-config", action="store_true", help="Show the effective backend configuration.")

    parser.add_argument("--show-bandwidth", action="store_true", help="Show fused-equivalent effective I/O bandwidth.")

    parser.add_argument("--show-throughput", action="store_true", help="Show processed-element throughput in GElem/s.")

    parser.add_argument(
        "--skip-correctness", action="store_true", help="Skip standalone random and semantic correctness suites."
    )

    parser.add_argument(
        "--profile-suite",
        default="all",
        choices=["sweep", "llm", "all"],
        help="Performance workload suite: synthetic sweep, LLM shapes, or both.",
    )

    parser.add_argument("--seed", default=0, type=int, help="Random seed used for reproducible inputs.")

    parser.add_argument("--warmup", default=10, type=int, help="Number of untimed warmup iterations.")

    parser.add_argument("--repeat", default=100, type=int, help="Number of operator invocations per timed round.")

    parser.add_argument("--rounds", default=10, type=int, help="Number of timed rounds.")

    parser.add_argument(
        "--benchmark-order",
        default="alternating",
        choices=["llaisys_then_torch", "torch_then_llaisys", "alternating"],
        type=str,
        help="Torch/LLAISYS benchmark order.",
    )

    parser.add_argument(
        "--output-dir", default="results", type=str, help="Directory for automatically named JSONL result files."
    )

    parser.add_argument("--no-record", action="store_true", help="Disable automatic JSONL recording.")

    parser.add_argument("--output", default=None, type=str, help=argparse.SUPPRESS)

    parser.add_argument("--run-id", default=None, type=str, help="Optional run identifier.")

    parser.add_argument("--run-note", default=None, type=str, help="Optional free-form run note.")

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
        raise ValueError("Triton SwiGLU requires a GPU device")

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

    if args.device_compute_fraction is not None and not 0.0 <= args.device_compute_fraction <= 1.0:
        raise ValueError("--device-compute-fraction must be within [0, 1]")

    torch.manual_seed(args.seed)

    if args.device in ("nvidia", "amd") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ========================================================
    # DTypes / tolerances
    # ========================================================

    test_dtype_prec = [("f32", 1e-5, 1e-5), ("f16", 2e-3, 2e-3), ("bf16", 2e-2, 2e-2)]

    # ========================================================
    # Correctness workload suite
    #
    # Covers:
    #   - tiny 2D tensors
    #   - warp-scale boundaries in total element count
    #   - block-size boundaries around 256
    #   - multi-block boundaries around 1024
    #   - irregular 2D row/column layouts and tails
    #   - representative LLM-sized tensors
    # ========================================================

    correctness_shapes = [
        # Tiny 2D tensors.
        (1, 1),
        (1, 3),
        (2, 1),
        (2, 3),
        # Warp-scale boundaries in total element count.
        (1, 31),
        (1, 32),
        (1, 33),
        # Default BLOCK_SIZE=256 boundaries.
        (1, 255),
        (1, 256),
        (1, 257),
        # Multi-block boundaries around 1024 elements.
        (1, 1023),
        (1, 1024),
        (1, 1025),
        # Non-trivial row/column layouts and masked tails.
        (3, 5),
        (7, 13),
        (17, 33),
        (33, 65),
        # Representative LLM-like tensors.
        (1, 4096),
        (32, 4096),
        (512, 4096),
    ]

    # ========================================================
    # Synthetic performance sweep
    # ========================================================

    sweep_shapes = [
        (1, 1 << 8),
        (1, 1 << 10),
        (1, 1 << 12),
        (1, 1 << 14),
        (1, 1 << 16),
        (1, 1 << 18),
        (1, 1 << 20),
        (1, 1 << 22),
        (1, 1 << 24),
    ]

    # ========================================================
    # LLM-representative feature tensors
    #
    # Keep the same token-count x 4096 matrix as Add so cross-op
    # comparisons are straightforward. Model-specific FFN intermediate
    # widths can be added later as application-level workloads.
    # ========================================================

    llm_shapes = [(1, 4096), (32, 4096), (512, 4096), (2048, 4096)]

    backend_metadata = collect_backend_metadata(
        args.backend, args.device, variant=args.backend_variant, implementation=args.backend_implementation
    )

    filename_config = get_swiglu_output_filename_config(args.backend)

    if args.output is not None:
        output_path = args.output
    elif args.profile and not args.profiler_mode and not args.no_record:
        output_path = build_experiment_output_path(
            args.output_dir, op="swiglu", device_name=args.device, backend=backend_metadata, config=filename_config
        )
    else:
        output_path = None

    run_metadata = {
        "profile_suite": args.profile_suite,
        "benchmark_order": args.benchmark_order,
        "note": args.run_note,
        "reference": {"torch": "torch_functional_silu_fp32_mul"},
        "input_distribution": {"gate": "uniform[-8,8)", "up": "uniform[-2,2)"},
        "profiler_mode": args.profiler_mode,
        "profiler_case": {
            "shape": list(args.case_shape) if args.case_shape is not None else None,
            "dtype": args.case_dtype,
            "target": args.profiler_target,
            "warmup": args.profiler_warmup,
            "launches": args.profiler_launches,
        },
        "output": {"automatic": args.output is None, "directory": args.output_dir, "filename_config": filename_config},
    }

    software_overrides = {}
    accelerator_stack_overrides = {}

    if args.accelerator_runtime_version is not None:
        accelerator_stack_overrides["runtime_version"] = args.accelerator_runtime_version

    if args.accelerator_driver_version is not None:
        accelerator_stack_overrides["driver_version"] = args.accelerator_driver_version

    if args.accelerator_compiler_version is not None:
        accelerator_stack_overrides["compiler"] = {"version": args.accelerator_compiler_version}

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
        device_metadata["total_memory_bytes"] = int(args.device_total_memory_gb * 1_000_000_000)

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
        resource_limits["memory_limit_bytes"] = int(args.device_memory_limit_gb * 1_000_000_000)

    if args.device_compute_fraction is not None:
        resource_limits["compute_fraction"] = args.device_compute_fraction

    if args.device_power_limit_w is not None:
        resource_limits["power_limit_w"] = args.device_power_limit_w

    if resource_limits:
        device_metadata["resource_limits"] = resource_limits

    if args.profiler_mode:
        dtype_tolerance = {dtype_name: (atol, rtol) for dtype_name, atol, rtol in test_dtype_prec}
        atol, rtol = dtype_tolerance[args.case_dtype]

        print(f"Profiling Ops.swiglu on {args.device} with {args.backend} backend")
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

        run_swiglu_profiler_case(
            shape=args.case_shape,
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

    print(f"Testing Ops.swiglu on {args.device} with {args.backend} backend")
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

    if recorder.enabled:
        print(f"Recording JSONL: {recorder.output_path}")
        print(f"Run ID: {recorder.run_id}")

    if not args.skip_correctness:
        print()
        print("=== Correctness: semantic cases ===")

        for dtype_name, atol, rtol in test_dtype_prec:
            test_swiglu_semantics(dtype_name, atol, rtol, args.device, args.backend)

        print()
        print("=== Correctness: shape/range coverage ===")

        for shape in correctness_shapes:
            for dtype_name, atol, rtol in test_dtype_prec:
                test_op_swiglu(
                    shape,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
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
                test_op_swiglu(
                    shape,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
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
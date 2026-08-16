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


import torch

import llaisys

from llaisys.triton import execution_context
from llaisys.triton.backends.registry import get_triton_backend
from llaisys.triton.ops import add as triton_add

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
# DType metadata
# ============================================================

DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2}


# ============================================================
# PyTorch reference / performance baseline
# ============================================================
#
# torch.add(..., out=out) is already a single PyTorch Add operator.
# Unlike the old explicit RMSNorm reference, this does not decompose
# Add into a sequence of eager tensor expressions.
#
# The out= form also reuses a preallocated output tensor, matching
# the LLAISYS API contract:
#
#     llaisys.Ops.add(out, a, b)
#
# Therefore this is the preferred apples-to-apples Torch baseline.
# ============================================================


def torch_add(out, a, b):
    torch.add(a, b, out=out)


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_add(out, a, b, backend):
    if backend == "native":
        llaisys.Ops.add(out, a, b)
        return

    if backend == "triton":
        triton_add(out, a, b)
        return

    raise ValueError(f"Unsupported Add backend: {backend}")


# ============================================================
# Effective configuration
# ============================================================


def get_add_config(tensor, backend):
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
        config = triton_backend.add_config(numel)

        return "effective", {"BLOCK_SIZE": config["BLOCK_SIZE"], "num_warps": config["num_warps"]}

    raise ValueError(f"Unsupported Add backend: {backend}")


def get_add_config_label(tensor, backend):
    status, config = get_add_config(tensor, backend)

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


def get_add_output_filename_config(backend):
    """
    Return the run-level configuration used only for automatic filenames.

    Per-case JSONL records still store the backend-resolved effective config.
    The filename records the explicit run knobs so config sweeps are easy to
    distinguish without opening the JSONL file.
    """
    if backend == "native":
        return {"BLOCK_SIZE": _parse_env_config_value("LLAISYS_BLOCK_SIZE")}

    if backend == "triton":
        return {
            "BLOCK_SIZE": _parse_env_config_value("LLAISYS_TRITON_BLOCK_SIZE"),
            "NUM_WARPS": _parse_env_config_value("LLAISYS_TRITON_NUM_WARPS"),
        }

    raise ValueError(f"Unsupported Add backend: {backend}")


# ============================================================
# Effective memory bandwidth
# ============================================================


def get_add_nominal_traffic_bytes(numel, dtype_name):
    return 3 * numel * DTYPE_BYTES[dtype_name]


def get_add_effective_bandwidth(numel, dtype_name, median_ms):
    traffic_bytes = get_add_nominal_traffic_bytes(numel, dtype_name)
    return traffic_bytes / median_ms / 1_000_000.0


def get_add_element_throughput(numel, median_ms):
    return numel / median_ms / 1_000_000.0


def get_add_derived_metrics(stats, numel, dtype_name):
    traffic_bytes = get_add_nominal_traffic_bytes(numel, dtype_name)

    derived = {
        "nominal_traffic_bytes": traffic_bytes,
        "llaisys_effective_bandwidth_gbs": get_add_effective_bandwidth(
            numel, dtype_name, stats["llaisys"]["median_ms"]
        ),
        "llaisys_element_throughput_gelem_s": get_add_element_throughput(numel, stats["llaisys"]["median_ms"]),
        "torch_effective_bandwidth_gbs": None,
        "torch_element_throughput_gelem_s": None,
    }

    if stats["torch"] is not None:
        derived["torch_effective_bandwidth_gbs"] = get_add_effective_bandwidth(
            numel, dtype_name, stats["torch"]["median_ms"]
        )
        derived["torch_element_throughput_gelem_s"] = get_add_element_throughput(numel, stats["torch"]["median_ms"])

    return derived


def print_add_effective_bandwidth(derived, device_name):
    print(f"        LLAISYS {device_name} effective I/O bandwidth: {derived['llaisys_effective_bandwidth_gbs']:.2f} GB/s")

    torch_bandwidth = derived["torch_effective_bandwidth_gbs"]

    if torch_bandwidth is not None:
        print(f"        Torch {device_name} effective I/O bandwidth:   {torch_bandwidth:.2f} GB/s")


def print_add_element_throughput(derived, device_name):
    print(
        f"        LLAISYS {device_name} element throughput: {derived['llaisys_element_throughput_gelem_s']:.3f} GElem/s"
    )

    torch_throughput = derived["torch_element_throughput_gelem_s"]

    if torch_throughput is not None:
        print(f"        Torch {device_name} element throughput:   {torch_throughput:.3f} GElem/s")


# ============================================================
# Profiler single-case mode
# ============================================================


def parse_case_shape(value):
    """Parse shapes such as 4096, 2048,4096, or 2048x4096."""
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

    return shape


def _torch_profiler_synchronize(device_name):
    if device_name in ("nvidia", "amd"):
        torch.cuda.synchronize()


def _begin_profiler_range(label, device_name):
    """Best-effort NVTX range for Nsight Systems/Compute correlation."""
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


def run_add_profiler_case(
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
    """
    Run one controlled Add case for external GPU profilers.

    This deliberately does NOT call benchmark(). The target launch count is
    kept small and deterministic so ncu/nsys/mcProfiler/rocprof can profile a
    single workload without seeing the normal 100 x 10 benchmark loop.

    The default profiler_warmup=1 is useful for Triton because it lets the
    first invocation perform JIT compilation/caching before the profiled NVTX
    target range. For ncu, when the kernel filter matches only the target Add
    kernel, use --launch-skip=<profiler_warmup>.
    """
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

    a, a_ = random_tensor(shape, dtype_name, device_name, scale=2.0, bias=-1.0)

    b, b_ = random_tensor(shape, dtype_name, device_name, scale=2.0, bias=-1.0)

    out, out_ = zero_tensor(shape, dtype_name, device_name)

    if profiler_target == "torch":
        if device_name == "metax":
            raise ValueError(
                "Torch profiler target is unavailable for MetaX because the "
                "current MetaX reference tensor is hosted on CPU."
            )

        target_fn = lambda: torch_add(out, a, b)
        synchronize = lambda: _torch_profiler_synchronize(device_name)
        config_status = "reference"
        config = {}

        target_label = (
            f"LLAISYS_PROFILE:add:torch:{device_name}:shape={'x'.join(str(dim) for dim in shape)}:dtype={dtype_name}"
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

    else:
        config_status, config = get_add_config(out_, backend)

        if show_config:
            print(f"        {get_add_config_label(out_, backend)}")

        target_fn = lambda: run_llaisys_add(out_, a_, b_, backend)

        api = llaisys.RuntimeAPI(out_.device_type())
        synchronize = api.device_synchronize

        config_tag = ",".join(f"{key}={value}" for key, value in config.items())

        target_label = (
            f"LLAISYS_PROFILE:add:{backend}:{backend_variant}:{device_name}:"
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
            with execution_context(out_.device_type(), out_.device_id()):
                execute_target()
        else:
            execute_target()

        # Optional correctness is intentionally performed AFTER the profiled
        # target range. This keeps the target range clean. For NCU with
        # --launch-count equal to profiler_launches, post-check kernels are not
        # part of the captured target launches.
        if profiler_check:
            torch_add(out, a, b)
            _torch_profiler_synchronize(device_name)

            assert check_equal(out_, out, atol=atol, rtol=rtol), (
                f"Add profiler correctness mismatch: "
                f"shape={shape}, "
                f"dtype={dtype_name}, "
                f"device={device_name}, "
                f"backend={backend}"
            )

    print(f"Profiler target range: {target_label}")
    print(f"Profiler launches: warmup={profiler_warmup}, target={profiler_launches}")

    if profiler_target == "llaisys":
        print(
            "NCU hint: when your --kernel-name filter matches only this Add "
            f"kernel, use --launch-skip {profiler_warmup} "
            f"--launch-count {profiler_launches}."
        )

    if profiler_check and profiler_target == "llaisys":
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


def benchmark_add(
    torch_out,
    torch_a,
    torch_b,
    llaisys_out,
    llaisys_a,
    llaisys_b,
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

    config_status, config = get_add_config(llaisys_out, backend)

    label = f"Add shape={shape} numel={numel} dtype={dtype_name} backend={backend}"

    if show_config:
        label += f" {get_add_config_label(llaisys_out, backend)}"

    print(f"        {label}:")

    torch_fn = lambda: torch_add(torch_out, torch_a, torch_b)
    llaisys_fn = lambda: run_llaisys_add(llaisys_out, llaisys_a, llaisys_b, backend)

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
        raise ValueError(f"Unsupported Add backend: {backend}")

    derived = get_add_derived_metrics(stats, numel, dtype_name)

    if show_bandwidth:
        print_add_effective_bandwidth(derived, device_name)

    if show_throughput:
        print_add_element_throughput(derived, device_name)

    recorder.record_microbenchmark(
        op="add",
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
            "torch_reference": "torch_add_out",
            "input_range": [-1.0, 1.0],
            "output_preallocated": True,
        },
        device_metadata=device_metadata,
    )


# ============================================================
# One test case
# ============================================================


def test_op_add(
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

    a, a_ = random_tensor(shape, dtype_name, device_name, scale=2.0, bias=-1.0)

    b, b_ = random_tensor(shape, dtype_name, device_name, scale=2.0, bias=-1.0)

    out, out_ = zero_tensor(shape, dtype_name, device_name)

    torch_add(out, a, b)
    run_llaisys_add(out_, a_, b_, backend)

    assert check_equal(out_, out, atol=atol, rtol=rtol), (
        f"Add mismatch: shape={shape}, numel={numel}, dtype={dtype_name}, device={device_name}, backend={backend}"
    )

    if profile:
        benchmark_add(
            out,
            a,
            b,
            out_,
            a_,
            b_,
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

    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax", "amd"], type=str)

    parser.add_argument("--backend", default="native", choices=["native", "triton"], type=str)

    parser.add_argument(
        "--backend-variant",
        default="unspecified",
        type=str,
        help=(
            "Experiment variant label stored in the schema, for example baseline, tuned, autotuned, or vendor-specific."
        ),
    )

    parser.add_argument(
        "--backend-implementation",
        default=None,
        type=str,
        help=(
            "Optional backend implementation override. By default native maps "
            "to cpu/cuda/maca/hip by device and triton maps to triton."
        ),
    )

    execution_mode = parser.add_mutually_exclusive_group()

    execution_mode.add_argument(
        "--profile", action="store_true", help="Run the normal paper-oriented microbenchmark suite."
    )

    execution_mode.add_argument(
        "--profiler-mode",
        action="store_true",
        help=(
            "Run exactly one controlled workload for external profilers "
            "such as ncu, nsys, mcProfiler, or rocprof. This mode does not "
            "run the normal benchmark loop or write microbenchmark JSONL."
        ),
    )

    parser.add_argument(
        "--case-shape",
        default=None,
        type=parse_case_shape,
        help=("Single profiler-case shape, for example 4096, 2048,4096, or 2048x4096. Required with --profiler-mode."),
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
        help=("Implementation launched inside the profiler target range. Default: llaisys."),
    )

    parser.add_argument(
        "--profiler-warmup",
        default=1,
        type=int,
        help=(
            "Number of target warmup launches before the profiler NVTX range. Default 1 also warms Triton JIT/caches."
        ),
    )

    parser.add_argument(
        "--profiler-launches", default=1, type=int, help="Exact number of target launches inside the profiler range."
    )

    parser.add_argument(
        "--profiler-check",
        action="store_true",
        help=(
            "After the profiled target range, compute the Torch reference and "
            "validate the LLAISYS output. Kept outside the target range so the "
            "profiled launches stay clean."
        ),
    )

    parser.add_argument(
        "--show-config", action="store_true", help="Include effective backend configuration in benchmark output."
    )

    parser.add_argument(
        "--show-bandwidth",
        action="store_true",
        help="Print effective memory bandwidth after each performance benchmark.",
    )

    parser.add_argument(
        "--show-throughput",
        action="store_true",
        help="Print element throughput (GElem/s) after each performance benchmark.",
    )

    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help=(
            "Skip the standalone correctness suite. Profile cases still perform one correctness check before timing."
        ),
    )

    parser.add_argument(
        "--profile-suite",
        default="all",
        choices=["sweep", "llm", "all"],
        help="Performance workload suite: log-scale sweep, LLM shapes, or both.",
    )

    parser.add_argument("--seed", default=0, type=int, help="Random seed used for reproducible inputs.")

    parser.add_argument("--warmup", default=10, type=int, help="Number of untimed warmup iterations.")

    parser.add_argument("--repeat", default=100, type=int, help="Number of operator invocations per timed round.")

    parser.add_argument(
        "--rounds", default=10, type=int, help="Number of timed rounds used to produce benchmark statistics."
    )

    parser.add_argument(
        "--benchmark-order",
        default="alternating",
        choices=["llaisys_then_torch", "torch_then_llaisys", "alternating"],
        type=str,
        help=("Order used to benchmark LLAISYS and Torch. The paper-oriented default alternates the order by round."),
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        type=str,
        help=(
            "Directory for automatically named JSONL benchmark files. "
            "The filename includes op, device, backend, variant, config, "
            "and a UTC timestamp."
        ),
    )

    parser.add_argument("--no-record", action="store_true", help="Disable automatic JSONL recording for this run.")

    # Backward-compatible explicit override. Normal runs should use the
    # automatic filename generated under --output-dir.
    parser.add_argument("--output", default=None, type=str, help=argparse.SUPPRESS)

    parser.add_argument(
        "--run-id", default=None, type=str, help="Optional run identifier. A UUID is generated when omitted."
    )

    parser.add_argument("--run-note", default=None, type=str, help="Optional free-form note stored in run metadata.")

    parser.add_argument("--device-model", default=None, type=str, help="Optional device-model override.")

    parser.add_argument(
        "--device-architecture",
        default=None,
        type=str,
        help=("Optional architecture override, for example sm_86, gfx942, or a MetaX architecture identifier."),
    )

    parser.add_argument(
        "--device-total-memory-gb",
        default=None,
        type=float,
        help="Optional physical/visible device memory override in GB.",
    )

    parser.add_argument(
        "--device-partition",
        default=None,
        type=str,
        help=("Free-form partition/slice description. Kept for convenient experiment annotation."),
    )

    parser.add_argument(
        "--device-partition-kind",
        default=None,
        type=str,
        help=("Partition mechanism, for example mig, sgpu, or amd-partition."),
    )

    parser.add_argument(
        "--device-partition-mode",
        default=None,
        type=str,
        help=("Partition mode/profile, for example a MIG profile or an AMD compute partition mode."),
    )

    parser.add_argument(
        "--device-partition-instance", default=None, type=str, help="Optional partition/instance identifier."
    )

    parser.add_argument(
        "--device-compute-partition", default=None, type=str, help="Optional compute-partition description."
    )

    parser.add_argument(
        "--device-memory-partition", default=None, type=str, help="Optional memory-partition description."
    )

    parser.add_argument(
        "--device-memory-limit-gb", default=None, type=float, help="Optional experiment memory quota/limit in GB."
    )

    parser.add_argument(
        "--device-compute-fraction",
        default=None,
        type=float,
        help=("Optional compute-share fraction in [0, 1], useful for slices/quotas."),
    )

    parser.add_argument(
        "--device-power-limit-w", default=None, type=float, help="Optional configured device power limit in watts."
    )

    parser.add_argument(
        "--accelerator-runtime-version",
        default=None,
        type=str,
        help=("Optional runtime-version override when automatic CUDA/MACA/HIP discovery is unavailable."),
    )

    parser.add_argument(
        "--accelerator-driver-version", default=None, type=str, help="Optional accelerator-driver-version override."
    )

    parser.add_argument(
        "--accelerator-compiler-version", default=None, type=str, help="Optional accelerator-compiler-version override."
    )

    args = parser.parse_args()

    if args.backend == "triton" and args.device == "cpu":
        raise ValueError("Triton Add requires a GPU device")

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

    test_dtype_prec = [("f32", 1e-5, 1e-5), ("f16", 1e-3, 1e-3), ("bf16", 1e-3, 1e-3)]

    correctness_shapes = [
        (1,),
        (2,),
        (3,),
        (31,),
        (32,),
        (33,),
        (63,),
        (64,),
        (65,),
        (127,),
        (128,),
        (129,),
        (255,),
        (256,),
        (257,),
        (511,),
        (512,),
        (513,),
        (1023,),
        (1024,),
        (1025,),
        (3, 5),
        (33, 65),
        (17, 33, 5),
        (7, 11, 13, 3),
        (512, 4096),
    ]

    sweep_shapes = [
        (1 << 8,),
        (1 << 10,),
        (1 << 12,),
        (1 << 14,),
        (1 << 16,),
        (1 << 18,),
        (1 << 20,),
        (1 << 22,),
        (1 << 24,),
    ]

    llm_shapes = [(1, 4096), (32, 4096), (512, 4096), (2048, 4096)]

    backend_metadata = collect_backend_metadata(
        args.backend, args.device, variant=args.backend_variant, implementation=args.backend_implementation
    )

    filename_config = get_add_output_filename_config(args.backend)

    if args.output is not None:
        output_path = args.output
    elif args.profile and not args.profiler_mode and not args.no_record:
        output_path = build_experiment_output_path(
            args.output_dir, op="add", device_name=args.device, backend=backend_metadata, config=filename_config
        )
    else:
        output_path = None

    run_metadata = {
        "profile_suite": args.profile_suite,
        "benchmark_order": args.benchmark_order,
        "note": args.run_note,
        "reference": {
            "torch": "torch_add_out",
        },
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
        dtype_tolerance = {"f32": (1e-5, 1e-5), "f16": (1e-3, 1e-3), "bf16": (1e-3, 1e-3)}
        atol, rtol = dtype_tolerance[args.case_dtype]

        print(f"Profiling Ops.add on {args.device} with {args.backend} backend")
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

        run_add_profiler_case(
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

    print(f"Testing Ops.add on {args.device} with {args.backend} backend")
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
        print("=== Correctness ===")

        for shape in correctness_shapes:
            for dtype_name, atol, rtol in test_dtype_prec:
                test_op_add(
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
                test_op_add(
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
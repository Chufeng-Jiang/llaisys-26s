import argparse
import math
import os
import sys
from ctypes import c_void_p


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
from llaisys.triton.ops import embedding as triton_embedding

from test_utils import (
    BenchmarkRecorder,
    benchmark,
    build_experiment_output_path,
    check_equal,
    collect_backend_metadata,
    random_int_tensor,
    random_tensor,
    zero_tensor,
)


# ============================================================
# DType metadata
# ============================================================

DTYPE_BYTES = {"f32": 4, "f16": 2, "bf16": 2}
INDEX_BYTES = 8  # Embedding indices are int64.


# ============================================================
# PyTorch reference / performance baseline
# ============================================================
#
# Use torch.index_select(..., out=out) as the framework baseline.
#
# This is already a single PyTorch gather operator and reuses a
# preallocated output tensor, matching the LLAISYS API contract:
#
#     llaisys.Ops.embedding(out, idx, embd)
#
# Avoid advanced indexing such as embd[idx], which creates a returned
# tensor and makes the output-allocation behavior less comparable.
# ============================================================


def torch_embedding(out, idx, embd):
    torch.index_select(embd, 0, idx, out=out)


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
# Effective configuration
# ============================================================


def get_embedding_config(tensor, embedding_dim, backend):
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
        config = triton_backend.embedding_config(embedding_dim)

        return "effective", {
            "BLOCK_SIZE": config["BLOCK_SIZE"],
            "num_warps": config["num_warps"],
        }

    raise ValueError(f"Unsupported Embedding backend: {backend}")


def get_embedding_config_label(tensor, embedding_dim, backend):
    status, config = get_embedding_config(tensor, embedding_dim, backend)

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


def get_embedding_output_filename_config(backend):
    if backend == "native":
        return {
            "BLOCK_SIZE": _parse_env_config_value("LLAISYS_BLOCK_SIZE"),
        }

    if backend == "triton":
        return {
            "BLOCK_SIZE": _parse_env_config_value("LLAISYS_TRITON_BLOCK_SIZE"),
            "NUM_WARPS": _parse_env_config_value("LLAISYS_TRITON_NUM_WARPS"),
        }

    raise ValueError(f"Unsupported Embedding backend: {backend}")


# ============================================================
# Derived performance metrics
# ============================================================
#
# Logical minimum I/O traffic for N embedding lookups of width D:
#
#     read N int64 indices                 -> N * 8 bytes
#     read N selected embedding rows       -> N * D * element_size
#     write N output rows                  -> N * D * element_size
#
# Therefore:
#
#     nominal_io_traffic_bytes
#       = N * 8 + 2 * N * D * element_size
#
# This is a logical/effective bandwidth metric, not a hardware DRAM-counter
# measurement. Repeated indices and cache hits can reduce physical DRAM traffic.
# The Torch value uses the same logical byte model, so it should be interpreted
# as equivalent logical I/O bandwidth rather than measured DRAM bandwidth.
# ============================================================


def get_embedding_nominal_io_traffic_bytes(index_count, embedding_dim, dtype_name):
    output_elements = index_count * embedding_dim
    index_bytes = index_count * INDEX_BYTES
    weight_read_bytes = output_elements * DTYPE_BYTES[dtype_name]
    output_write_bytes = output_elements * DTYPE_BYTES[dtype_name]

    return {
        "index_bytes": index_bytes,
        "weight_read_bytes": weight_read_bytes,
        "output_write_bytes": output_write_bytes,
        "total_bytes": index_bytes + weight_read_bytes + output_write_bytes,
    }


def get_effective_bandwidth_gbs(traffic_bytes, median_ms):
    return traffic_bytes / median_ms / 1_000_000.0


def get_element_throughput_gelem_s(output_elements, median_ms):
    return output_elements / median_ms / 1_000_000.0


def get_embedding_derived_metrics(stats, index_count, embedding_dim, dtype_name):
    traffic = get_embedding_nominal_io_traffic_bytes(index_count, embedding_dim, dtype_name)
    output_elements = index_count * embedding_dim
    llaisys_stats = stats["llaisys"]
    torch_stats = stats.get("torch")

    derived = {
        "nominal_index_bytes": traffic["index_bytes"],
        "nominal_weight_read_bytes": traffic["weight_read_bytes"],
        "nominal_output_write_bytes": traffic["output_write_bytes"],
        "nominal_io_traffic_bytes": traffic["total_bytes"],
        "llaisys_effective_io_bandwidth_gbs": get_effective_bandwidth_gbs(
            traffic["total_bytes"], llaisys_stats["median_ms"]
        ),
        "torch_equivalent_io_bandwidth_gbs": None,
        "llaisys_element_throughput_gelem_s": get_element_throughput_gelem_s(
            output_elements, llaisys_stats["median_ms"]
        ),
        "torch_element_throughput_gelem_s": None,
    }

    if torch_stats is not None:
        derived["torch_equivalent_io_bandwidth_gbs"] = get_effective_bandwidth_gbs(
            traffic["total_bytes"], torch_stats["median_ms"]
        )
        derived["torch_element_throughput_gelem_s"] = get_element_throughput_gelem_s(
            output_elements, torch_stats["median_ms"]
        )

    return derived


def print_embedding_effective_bandwidth(derived, device_name):
    print(
        f"        LLAISYS {device_name} effective I/O bandwidth: "
        f"{derived['llaisys_effective_io_bandwidth_gbs']:.2f} GB/s"
    )

    torch_bandwidth = derived["torch_equivalent_io_bandwidth_gbs"]
    if torch_bandwidth is not None:
        print(
            f"        Torch {device_name} equivalent I/O bandwidth: "
            f"{torch_bandwidth:.2f} GB/s"
        )


def print_embedding_element_throughput(derived, device_name):
    print(
        f"        LLAISYS {device_name} element throughput: "
        f"{derived['llaisys_element_throughput_gelem_s']:.3f} GElem/s"
    )

    torch_throughput = derived["torch_element_throughput_gelem_s"]
    if torch_throughput is not None:
        print(
            f"        Torch {device_name} element throughput: "
            f"{torch_throughput:.3f} GElem/s"
        )


# ============================================================
# Deterministic semantic cases
# ============================================================


def _load_exact_indices(indices, idx_tensor):
    idx_host = torch.tensor(indices, dtype=torch.int64, device="cpu").contiguous()
    idx_tensor.load(c_void_p(idx_host.data_ptr()))
    return idx_host


def test_embedding_semantic_case(name, indices, embd_shape, dtype_name, device_name, backend):
    index_count = len(indices)
    vocab_size, embedding_dim = embd_shape
    idx_shape = (index_count,)
    out_shape = (index_count, embedding_dim)

    print(
        f"   semantic {name} "
        f"index_count <{index_count}> "
        f"embd_shape {embd_shape} "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    embd, embd_ = random_tensor(embd_shape, dtype_name, device_name, scale=2.0, bias=-1.0)

    _, idx_ = random_int_tensor(idx_shape, device_name, high=max(vocab_size, 1))
    idx_host = _load_exact_indices(indices, idx_)
    idx_reference = idx_host.to(device=embd.device)

    out, out_ = zero_tensor(out_shape, dtype_name, device_name)

    torch_embedding(out, idx_reference, embd)
    run_llaisys_embedding(out_, idx_, embd_, backend)

    assert check_equal(out_, out, strict=True), (
        f"Embedding semantic mismatch: case={name}, "
        f"embd_shape={embd_shape}, dtype={dtype_name}, "
        f"device={device_name}, backend={backend}"
    )


def run_embedding_semantic_tests(device_name, dtype_name, backend):
    # D=17 intentionally crosses a common 16-element vector width and remains
    # non-aligned for the Triton BLOCK_SIZE=128 baseline.
    embd_shape = (8, 17)

    cases = [
        ("first_row", [0]),
        ("last_row", [7]),
        ("first_and_last", [0, 7]),
        ("duplicate_row", [3, 3, 3, 3, 3]),
        ("alternating_extremes", [0, 7, 0, 7, 0, 7]),
        ("all_rows_forward", list(range(8))),
        ("all_rows_reverse", list(range(7, -1, -1))),
        ("mixed_unsorted", [4, 1, 6, 2, 0, 7, 3, 5]),
        ("repeated_tail_33", [i % 8 for i in range(33)]),
    ]

    for name, indices in cases:
        test_embedding_semantic_case(name, indices, embd_shape, dtype_name, device_name, backend)


# ============================================================
# Profiler single-case mode
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


def run_embedding_profiler_case(
    *,
    index_count,
    vocab_size,
    embedding_dim,
    dtype_name,
    device_name,
    backend,
    backend_variant,
    profiler_target,
    profiler_warmup,
    profiler_launches,
    profiler_check,
    show_config,
):
    idx_shape = (index_count,)
    embd_shape = (vocab_size, embedding_dim)
    out_shape = (index_count, embedding_dim)
    output_elements = index_count * embedding_dim

    print()
    print("=== Profiler single case ===")
    print(
        f"   target <{profiler_target}> "
        f"index_count {index_count} "
        f"vocab_size {vocab_size} "
        f"embedding_dim {embedding_dim} "
        f"output_elements {output_elements} "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    embd, embd_ = random_tensor(embd_shape, dtype_name, device_name, scale=2.0, bias=-1.0)
    idx, idx_ = random_int_tensor(idx_shape, device_name, high=vocab_size)
    out, out_ = zero_tensor(out_shape, dtype_name, device_name)

    if profiler_target == "torch":
        if device_name == "metax":
            raise ValueError(
                "Torch profiler target is unavailable for MetaX because the "
                "current MetaX reference tensors are hosted on CPU."
            )

        target_fn = lambda: torch_embedding(out, idx, embd)
        synchronize = lambda: _torch_profiler_synchronize(device_name)
        config_status = "reference"
        config = {}
        target_label = (
            f"LLAISYS_PROFILE:embedding:torch:{device_name}:"
            f"n={index_count}:v={vocab_size}:d={embedding_dim}:dtype={dtype_name}"
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
            run_llaisys_embedding(out_, idx_, embd_, backend)

            assert check_equal(out_, out, strict=True), (
                f"Embedding profiler correctness mismatch: "
                f"index_count={index_count}, vocab_size={vocab_size}, "
                f"embedding_dim={embedding_dim}, dtype={dtype_name}, "
                f"device={device_name}, backend={backend}"
            )

    else:
        config_status, config = get_embedding_config(out_, embedding_dim, backend)

        if show_config:
            print(f"        {get_embedding_config_label(out_, embedding_dim, backend)}")

        target_fn = lambda: run_llaisys_embedding(out_, idx_, embd_, backend)
        api = llaisys.RuntimeAPI(out_.device_type())
        synchronize = api.device_synchronize
        config_tag = ",".join(f"{key}={value}" for key, value in config.items())
        target_label = (
            f"LLAISYS_PROFILE:embedding:{backend}:{backend_variant}:{device_name}:"
            f"n={index_count}:v={vocab_size}:d={embedding_dim}:dtype={dtype_name}:"
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

        if profiler_check:
            torch_embedding(out, idx, embd)
            _torch_profiler_synchronize(device_name)

            assert check_equal(out_, out, strict=True), (
                f"Embedding profiler correctness mismatch: "
                f"index_count={index_count}, vocab_size={vocab_size}, "
                f"embedding_dim={embedding_dim}, dtype={dtype_name}, "
                f"device={device_name}, backend={backend}"
            )

    print(f"Profiler target range: {target_label}")
    print(f"Profiler launches: warmup={profiler_warmup}, target={profiler_launches}")

    if profiler_target == "llaisys":
        print(
            "NCU hint: when your --kernel-name filter matches only the target "
            f"Embedding kernel, use --launch-skip {profiler_warmup} "
            f"--launch-count {profiler_launches}."
        )

    if profiler_check:
        print("Profiler post-check: passed")

    return {
        "target": profiler_target,
        "index_count": index_count,
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "output_elements": output_elements,
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


def benchmark_embedding(
    torch_out,
    torch_idx,
    torch_embd,
    llaisys_out,
    llaisys_idx,
    llaisys_embd,
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
    out_shape = llaisys_out.shape()
    index_count = out_shape[0]
    embedding_dim = out_shape[1]
    vocab_size = llaisys_embd.shape()[0]
    output_elements = index_count * embedding_dim

    config_status, config = get_embedding_config(llaisys_out, embedding_dim, backend)

    label = (
        f"Embedding index_count={index_count} "
        f"vocab_size={vocab_size} "
        f"embedding_dim={embedding_dim} "
        f"output_elements={output_elements} "
        f"dtype={dtype_name} backend={backend}"
    )

    if show_config:
        label += f" {get_embedding_config_label(llaisys_out, embedding_dim, backend)}"

    print(f"        {label}:")

    torch_fn = lambda: torch_embedding(torch_out, torch_idx, torch_embd)
    llaisys_fn = lambda: run_llaisys_embedding(llaisys_out, llaisys_idx, llaisys_embd, backend)

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
        raise ValueError(f"Unsupported Embedding backend: {backend}")

    derived = get_embedding_derived_metrics(stats, index_count, embedding_dim, dtype_name)

    if show_bandwidth:
        print_embedding_effective_bandwidth(derived, device_name)

    if show_throughput:
        print_embedding_element_throughput(derived, device_name)

    recorder.record_microbenchmark(
        op="embedding",
        backend_name=backend,
        backend_variant=backend_variant,
        backend_implementation=backend_implementation,
        suite=suite,
        device_name=device_name,
        device_id=llaisys_out.device_id(),
        shape=out_shape,
        numel=output_elements,
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
            "index_shape": [index_count],
            "index_count": index_count,
            "index_dtype": "i64",
            "embedding_table_shape": [vocab_size, embedding_dim],
            "vocabulary_size": vocab_size,
            "embedding_dim": embedding_dim,
            "output_shape": list(out_shape),
            "output_elements": output_elements,
            "access_pattern": "random_valid_reused",
            "torch_reference": "torch_index_select_out",
            "torch_output_preallocated": True,
            "llaisys_output_preallocated": True,
        },
        device_metadata=device_metadata,
    )


# ============================================================
# One random valid-index correctness/performance case
# ============================================================


def test_op_embedding(
    idx_shape,
    embd_shape,
    dtype_name="f32",
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
    if len(idx_shape) != 1:
        raise ValueError(f"Embedding index tensor must be 1D, got {idx_shape}")

    if len(embd_shape) != 2:
        raise ValueError(f"Embedding table must be 2D, got {embd_shape}")

    index_count = idx_shape[0]
    vocab_size, embedding_dim = embd_shape
    out_shape = (index_count, embedding_dim)
    output_elements = index_count * embedding_dim

    print(
        f"   random "
        f"index_count {index_count} "
        f"embd_shape {embd_shape} "
        f"output_elements {output_elements} "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    embd, embd_ = random_tensor(embd_shape, dtype_name, device_name, scale=2.0, bias=-1.0)
    idx, idx_ = random_int_tensor(idx_shape, device_name, high=vocab_size)
    out, out_ = zero_tensor(out_shape, dtype_name, device_name)

    torch_embedding(out, idx, embd)
    run_llaisys_embedding(out_, idx_, embd_, backend)

    assert check_equal(out_, out, strict=True), (
        f"Embedding mismatch: idx_shape={idx_shape}, embd_shape={embd_shape}, "
        f"dtype={dtype_name}, device={device_name}, backend={backend}"
    )

    if profile:
        if recorder is None:
            recorder = BenchmarkRecorder()

        benchmark_embedding(
            out,
            idx,
            embd,
            out_,
            idx_,
            embd_,
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
        default="cpu",
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
        help="Experiment variant label, for example baseline, tuned, autotuned, or vendor-specific.",
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
        "--profile",
        action="store_true",
        help="Run the paper-oriented microbenchmark suites.",
    )

    execution_mode.add_argument(
        "--profiler-mode",
        action="store_true",
        help=(
            "Run one controlled workload for an external profiler such as "
            "ncu, nsys, mcProfiler, or rocprof."
        ),
    )

    parser.add_argument(
        "--case-index-count",
        default=None,
        type=int,
        help="Number of embedding lookups for --profiler-mode. Required in profiler mode.",
    )

    parser.add_argument(
        "--case-vocab-size",
        default=4096,
        type=int,
        help="Embedding vocabulary size for --profiler-mode. Default: 4096.",
    )

    parser.add_argument(
        "--case-embedding-dim",
        default=4096,
        type=int,
        help="Embedding row width for --profiler-mode. Default: 4096.",
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
        help="Implementation launched inside the profiler target range.",
    )

    parser.add_argument(
        "--profiler-warmup",
        default=1,
        type=int,
        help="Warmup launches before the profiler target range. Default: 1.",
    )

    parser.add_argument(
        "--profiler-launches",
        default=1,
        type=int,
        help="Exact number of target launches inside the profiler range.",
    )

    parser.add_argument(
        "--profiler-check",
        action="store_true",
        help="Validate correctness after the profiled target range.",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Include the effective backend configuration in benchmark output.",
    )

    parser.add_argument(
        "--show-bandwidth",
        action="store_true",
        help="Print effective/equivalent logical I/O bandwidth.",
    )

    parser.add_argument(
        "--show-throughput",
        action="store_true",
        help="Print output-element throughput in GElem/s.",
    )

    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the standalone correctness suites.",
    )

    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip deterministic valid-index semantic cases.",
    )

    parser.add_argument(
        "--profile-suite",
        default="all",
        choices=["sweep", "llm", "all"],
        help="Performance suite: index-count sweep, LLM-like cases, or both.",
    )

    parser.add_argument("--seed", default=0, type=int, help="Random seed for reproducible inputs.")
    parser.add_argument("--warmup", default=10, type=int, help="Untimed warmup iterations.")
    parser.add_argument("--repeat", default=100, type=int, help="Operator invocations per timed round.")
    parser.add_argument("--rounds", default=10, type=int, help="Timed benchmark rounds.")

    parser.add_argument(
        "--benchmark-order",
        default="alternating",
        choices=["llaisys_then_torch", "torch_then_llaisys", "alternating"],
        type=str,
        help="Order used to benchmark LLAISYS and Torch.",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        type=str,
        help="Directory for automatically named JSONL benchmark files.",
    )

    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable automatic JSONL recording.",
    )

    parser.add_argument("--output", default=None, type=str, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", default=None, type=str, help="Optional run identifier.")
    parser.add_argument("--run-note", default=None, type=str, help="Optional note stored in run metadata.")

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
        raise ValueError("Triton Embedding requires a GPU device")

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

    if args.profiler_mode and args.case_index_count is None:
        raise ValueError("--case-index-count is required with --profiler-mode")

    if args.case_index_count is not None and args.case_index_count <= 0:
        raise ValueError("--case-index-count must be greater than zero")

    if args.case_vocab_size <= 0:
        raise ValueError("--case-vocab-size must be greater than zero")

    if args.case_embedding_dim <= 0:
        raise ValueError("--case-embedding-dim must be greater than zero")

    if args.device_compute_fraction is not None and not 0.0 <= args.device_compute_fraction <= 1.0:
        raise ValueError("--device-compute-fraction must be within [0, 1]")

    # ========================================================
    # Reproducibility
    # ========================================================

    torch.manual_seed(args.seed)

    if args.device in ("nvidia", "amd") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    test_dtypes = ["f32", "f16", "bf16"]

    # ========================================================
    # Correctness matrix
    #
    # Covers:
    #   - minimal shapes
    #   - row widths around 16 / 32 / 64 / 128 / 256
    #   - index-count boundaries around 32 and 256
    #   - non-aligned vector/tile tails
    #   - large 4096-width embedding rows
    #
    # Only valid indices are generated here; invalid-index behavior is not
    # part of this correctness suite.
    # ========================================================

    correctness_cases = [
        # Minimal / tiny.
        ((1,), (2, 1)),
        ((1,), (2, 3)),
        ((3,), (5, 7)),

        # 16-column boundary.
        ((7,), (17, 15)),
        ((8,), (17, 16)),
        ((9,), (17, 17)),

        # 32-column boundary.
        ((15,), (31, 31)),
        ((16,), (31, 32)),
        ((17,), (31, 33)),

        # 64-column boundary + index-count 31/32/33.
        ((31,), (127, 63)),
        ((32,), (127, 64)),
        ((33,), (127, 65)),

        # Triton baseline BLOCK_SIZE=128 boundary.
        ((7,), (17, 127)),
        ((8,), (17, 128)),
        ((9,), (17, 129)),

        # 256-column boundary.
        ((31,), (257, 255)),
        ((32,), (257, 256)),
        ((33,), (257, 257)),

        # Index-count 255/256/257 boundary with irregular row width.
        ((255,), (1024, 33)),
        ((256,), (1024, 33)),
        ((257,), (1024, 33)),

        # Large row-width tail/aligned cases.
        ((50,), (512, 4095)),
        ((50,), (512, 4096)),
        ((50,), (512, 4097)),
    ]

    # ========================================================
    # Performance suites
    #
    # Sweep:
    #   fixed V=4096, D=4096; vary the number of gathered rows.
    #   This moves from tiny decode-like lookup counts toward large batched
    #   gather workloads while keeping the embedding-table geometry fixed.
    #
    # LLM-like:
    #   V=8192, D=4096 with representative token counts. These are synthetic
    #   representative workloads rather than one specific model's vocabulary.
    # ========================================================

    sweep_cases = [
        ((1,), (4096, 4096)),
        ((4,), (4096, 4096)),
        ((16,), (4096, 4096)),
        ((64,), (4096, 4096)),
        ((256,), (4096, 4096)),
        ((1024,), (4096, 4096)),
        ((4096,), (4096, 4096)),
    ]

    llm_cases = [
        ((1,), (8192, 4096)),
        ((32,), (8192, 4096)),
        ((512,), (8192, 4096)),
        ((2048,), (8192, 4096)),
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

    filename_config = get_embedding_output_filename_config(args.backend)

    if args.output is not None:
        output_path = args.output
    elif args.profile and not args.profiler_mode and not args.no_record:
        output_path = build_experiment_output_path(
            args.output_dir,
            op="embedding",
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
            "torch": "torch_index_select_out",
            "torch_output_preallocated": True,
            "llaisys_output_preallocated": True,
        },
        "profiler_mode": args.profiler_mode,
        "profiler_case": {
            "index_count": args.case_index_count,
            "vocabulary_size": args.case_vocab_size,
            "embedding_dim": args.case_embedding_dim,
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

    # ========================================================
    # Profiler mode
    # ========================================================

    if args.profiler_mode:
        print(f"Profiling Ops.embedding on {args.device} with {args.backend} backend")
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

        run_embedding_profiler_case(
            index_count=args.case_index_count,
            vocab_size=args.case_vocab_size,
            embedding_dim=args.case_embedding_dim,
            dtype_name=args.case_dtype,
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

    print(f"Testing Ops.embedding on {args.device} with {args.backend} backend")
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
    print(f"Using llaisys from: {llaisys.__file__}")

    if recorder.enabled:
        print(f"Recording JSONL: {recorder.output_path}")
        print(f"Run ID: {recorder.run_id}")

    if not args.skip_correctness:
        print()
        print("=== Random valid-index correctness ===")

        for idx_shape, embd_shape in correctness_cases:
            for dtype_name in test_dtypes:
                test_op_embedding(
                    idx_shape,
                    embd_shape,
                    dtype_name=dtype_name,
                    device_name=args.device,
                    backend=args.backend,
                    backend_variant=args.backend_variant,
                    backend_implementation=args.backend_implementation,
                    profile=False,
                    seed=args.seed,
                    recorder=recorder,
                    device_metadata=device_metadata,
                )

        if not args.skip_semantic:
            print()
            print("=== Deterministic valid-index semantics ===")

            for dtype_name in test_dtypes:
                run_embedding_semantic_tests(args.device, dtype_name, args.backend)

    if args.profile:
        print()
        print("=== Performance ===")

        profile_cases = []

        if args.profile_suite in ("sweep", "all"):
            profile_cases.extend(("sweep", idx_shape, embd_shape) for idx_shape, embd_shape in sweep_cases)

        if args.profile_suite in ("llm", "all"):
            profile_cases.extend(("llm", idx_shape, embd_shape) for idx_shape, embd_shape in llm_cases)

        for suite, idx_shape, embd_shape in profile_cases:
            for dtype_name in test_dtypes:
                test_op_embedding(
                    idx_shape,
                    embd_shape,
                    dtype_name=dtype_name,
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
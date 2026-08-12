import statistics
import sys
from pathlib import Path

import torch
import triton

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

from llaisys.libllaisys import (
    DeviceType,
)
from llaisys.runtime import RuntimeAPI
from llaisys.triton.backends.nvidia import (
    NvidiaTritonBackend,
)
from llaisys.triton.kernels.add import (
    add_kernel,
)
from llaisys.triton.tensor import (
    as_nvidia_triton_tensor,
)
from test_utils import (
    check_equal,
    random_tensor,
)

import llaisys

# ============================================================
# DType size
# ============================================================


_DTYPE_BYTES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
}


# ============================================================
# CUDA-event benchmark
# ============================================================
#
# IMPORTANT:
#
# This benchmark measures GPU-side elapsed time using CUDA
# events recorded on the SAME LLAISYS CUDA stream used by:
#
#     Native CUDA Add
#
# and:
#
#     Triton Add
#
# Host-side costs such as:
#
#     Python validation
#     ctypes calls
#     Tensor metadata access
#     stream-context enter / exit
#     TritonTensor wrapper construction
#
# are intentionally excluded.
#
# Multiple kernel launches are placed between two CUDA events
# and the elapsed time is divided by repeat. This amortizes
# event timing overhead and improves stability for tiny kernels.
# ============================================================


def benchmark_gpu_events(
    func,
    runtime,
    backend,
    stream_ptr,
    device_id,
    warmup=100,
    repeat=1000,
    rounds=20,
):
    # ========================================================
    # Warmup
    # ========================================================

    with backend.stream_context(
        stream_ptr,
        device_id,
    ):
        for _ in range(warmup):
            func()

    runtime.device_synchronize()

    # ========================================================
    # Timed rounds
    # ========================================================

    samples_us = []

    for _ in range(rounds):
        start_event = torch.cuda.Event(enable_timing=True)

        end_event = torch.cuda.Event(enable_timing=True)

        with backend.stream_context(
            stream_ptr,
            device_id,
        ):
            # ------------------------------------------------
            # Event is inserted into the LLAISYS stream.
            # ------------------------------------------------

            start_event.record()

            for _ in range(repeat):
                func()

            end_event.record()

        # ----------------------------------------------------
        # Wait until the ending event has completed.
        # ----------------------------------------------------

        end_event.synchronize()

        # CUDA Event elapsed_time() returns milliseconds.
        elapsed_ms = start_event.elapsed_time(end_event)

        latency_us = elapsed_ms * 1000.0 / repeat

        samples_us.append(latency_us)

    return {
        "median_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


# ============================================================
# Effective memory bandwidth
# ============================================================
#
# Add:
#
#     c = a + b
#
# Approximate memory traffic:
#
#     read a
#     read b
#     write c
#
# Therefore:
#
#     traffic = 3 * numel * dtype_size
#
# This is an approximate effective bandwidth metric.
# ============================================================


def effective_bandwidth_gbps(
    numel,
    dtype_name,
    latency_us,
):
    if latency_us <= 0:
        return 0.0

    bytes_per_element = _DTYPE_BYTES[dtype_name]

    total_bytes = 3 * numel * bytes_per_element

    latency_seconds = latency_us * 1e-6

    return total_bytes / latency_seconds / 1e9


# ============================================================
# Benchmark one workload
# ============================================================


def benchmark_case(
    shape,
    dtype_name,
):
    print()
    print("============================================================")

    print(f"shape={shape} dtype={dtype_name}")

    print("============================================================")

    # ========================================================
    # Tensors
    # ========================================================

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

    _, c_native = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, c_triton = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    device_id = a.device_id()

    # ========================================================
    # Runtime / backend
    # ========================================================

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    runtime.set_device(device_id)

    backend = NvidiaTritonBackend()

    stream_ptr = runtime.get_context_stream(device_id)

    # ========================================================
    # Numel
    # ========================================================

    numel = 1

    for dim in shape:
        numel *= dim

    # ========================================================
    # Native CUDA path
    # ========================================================
    #
    # Native Add explicitly launches on:
    #
    #     Runtime::_stream
    #
    # CUDA events are also inserted into exactly this stream,
    # because the surrounding backend.stream_context() wraps
    # the same raw cudaStream_t.
    # ========================================================

    def native_op():
        llaisys.Ops.add(
            c_native,
            a,
            b,
        )

    # ========================================================
    # Triton pre-bound path
    # ========================================================
    #
    # Prepare all host-side state outside the timed region.
    #
    # The CUDA events therefore measure the GPU execution
    # interval of repeated Triton kernels.
    # ========================================================

    config = backend.add_config(numel)

    block_size = config["BLOCK_SIZE"]

    grid = (
        triton.cdiv(
            numel,
            block_size,
        ),
    )

    c_triton_wrapper = as_nvidia_triton_tensor(c_triton)

    a_triton_wrapper = as_nvidia_triton_tensor(a)

    b_triton_wrapper = as_nvidia_triton_tensor(b)

    def triton_op():
        add_kernel[grid](
            c_triton_wrapper,
            a_triton_wrapper,
            b_triton_wrapper,
            numel,
            BLOCK_SIZE=block_size,
            num_warps=config["num_warps"],
        )

    # ========================================================
    # Correctness check before timing
    # ========================================================

    expected = a_ref + b_ref

    native_op()

    with backend.stream_context(
        stream_ptr,
        device_id,
    ):
        triton_op()

    runtime.device_synchronize()

    assert check_equal(
        c_native,
        expected,
        atol=1e-3,
        rtol=1e-3,
    )

    assert check_equal(
        c_triton,
        expected,
        atol=1e-3,
        rtol=1e-3,
    )

    # ========================================================
    # GPU timing
    # ========================================================

    native_result = benchmark_gpu_events(
        native_op,
        runtime,
        backend,
        stream_ptr,
        device_id,
    )

    triton_result = benchmark_gpu_events(
        triton_op,
        runtime,
        backend,
        stream_ptr,
        device_id,
    )

    # ========================================================
    # Extract results
    # ========================================================

    native_us = native_result["median_us"]

    triton_us = triton_result["median_us"]

    # ========================================================
    # Triton / Native ratio
    # ========================================================

    if native_us > 0:
        ratio = triton_us / native_us
    else:
        ratio = 0.0

    # ========================================================
    # Difference
    # ========================================================

    difference_us = triton_us - native_us

    # ========================================================
    # Effective bandwidth
    # ========================================================

    native_bw = effective_bandwidth_gbps(
        numel,
        dtype_name,
        native_us,
    )

    triton_bw = effective_bandwidth_gbps(
        numel,
        dtype_name,
        triton_us,
    )

    # ========================================================
    # Detailed output
    # ========================================================

    print()
    print("GPU kernel latency:")

    print(f"  Native CUDA median: {native_us:8.3f} us")

    print(f"  Native CUDA mean:   {native_result['mean_us']:8.3f} us")

    print(f"  Native CUDA min:    {native_result['min_us']:8.3f} us")

    print(f"  Native CUDA max:    {native_result['max_us']:8.3f} us")

    print()

    print(f"  Triton median:      {triton_us:8.3f} us")

    print(f"  Triton mean:        {triton_result['mean_us']:8.3f} us")

    print(f"  Triton min:         {triton_result['min_us']:8.3f} us")

    print(f"  Triton max:         {triton_result['max_us']:8.3f} us")

    print()
    print("Comparison:")

    print(f"  Triton / Native:    {ratio:8.3f} x")

    print(f"  Difference:         {difference_us:8.3f} us")

    print()
    print("Approx. effective bandwidth:")

    print(f"  Native:             {native_bw:8.2f} GB/s")

    print(f"  Triton:             {triton_bw:8.2f} GB/s")

    return {
        "shape": shape,
        "dtype": dtype_name,
        "numel": numel,
        "native_us": native_us,
        "triton_us": triton_us,
        "ratio": ratio,
        "difference_us": difference_us,
        "native_bw": native_bw,
        "triton_bw": triton_bw,
        "native_min_us": native_result["min_us"],
        "native_max_us": native_result["max_us"],
        "triton_min_us": triton_result["min_us"],
        "triton_max_us": triton_result["max_us"],
    }


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("============================================================")

    print("LLAISYS Native CUDA vs Triton Add")

    print("GPU Event Microbenchmark")

    print("============================================================")

    print()
    print("Timing method:")

    print("  CUDA events on the LLAISYS Runtime CUDA stream.")

    print("  Host-side Python/runtime integration is excluded.")

    # ========================================================
    # Test cases
    # ========================================================

    test_cases = [
        ((2, 3), "f32"),
        ((33, 65), "f32"),
        ((512, 4096), "f32"),
        ((2, 3), "f16"),
        ((512, 4096), "f16"),
        ((2, 3), "bf16"),
        ((512, 4096), "bf16"),
    ]

    results = []

    for shape, dtype_name in test_cases:
        result = benchmark_case(
            shape,
            dtype_name,
        )

        results.append(result)

    # ========================================================
    # Latency summary
    # ========================================================

    print()
    print("============================================================")

    print("GPU Kernel Latency Summary")

    print("============================================================")

    print(f"{'Shape':>16} {'DType':>6} {'Native':>10} {'Triton':>10} {'T/N':>8} {'Delta':>10}")

    print("-" * 70)

    for result in results:
        print(
            f"{result['shape']!s:>16} "
            f"{result['dtype']:>6} "
            f"{result['native_us']:>9.3f}u "
            f"{result['triton_us']:>9.3f}u "
            f"{result['ratio']:>7.3f}x "
            f"{result['difference_us']:>9.3f}u"
        )

    # ========================================================
    # Effective bandwidth summary
    # ========================================================

    print()
    print("============================================================")

    print("Approximate Effective Memory Bandwidth")

    print("============================================================")

    print(f"{'Shape':>16} {'DType':>6} {'Native':>14} {'Triton':>14}")

    print("-" * 58)

    for result in results:
        print(
            f"{result['shape']!s:>16} "
            f"{result['dtype']:>6} "
            f"{result['native_bw']:>11.2f} GB/s "
            f"{result['triton_bw']:>11.2f} GB/s"
        )

    # ========================================================
    # Notes
    # ========================================================

    print()
    print("NOTE:")

    print("  These numbers measure GPU-stream elapsed time using")

    print("  CUDA events.")

    print()
    print("  They intentionally exclude Python metadata access,")

    print("  ctypes calls, stream-context management, and other")

    print("  host-side integration overhead.")

    print()
    print("  Effective bandwidth assumes approximately:")

    print("      2 tensor reads + 1 tensor write")

    print("  and should be interpreted as an approximate metric.")

import statistics
import sys
import time
from pathlib import Path

import triton

# ============================================================
# Project paths
# ============================================================

repo_root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(repo_root / "python"))

sys.path.insert(0, str(repo_root / "test"))


# ============================================================
# Imports
# ============================================================

from llaisys.libllaisys import DeviceType
from llaisys.runtime import RuntimeAPI
from llaisys.triton import execution_context
from llaisys.triton.backends.nvidia import NvidiaTritonBackend
from llaisys.triton.kernels.add import add_kernel
from llaisys.triton.ops import add as triton_add
from llaisys.triton.tensor import as_nvidia_triton_tensor
from test_utils import random_tensor

import llaisys

# ============================================================
# Host-visible latency benchmark
# ============================================================
#
# This measures:
#
#     Python dispatch
#     + runtime integration
#     + kernel launch
#     + GPU execution
#
# Synchronization happens only at round boundaries.
#
# Therefore this is NOT pure GPU kernel latency.
# ============================================================


def benchmark_host_latency(func, synchronize, warmup=20, repeat=100, rounds=10):
    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(warmup):
        func()

    synchronize()

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    samples_us = []

    for _ in range(rounds):
        synchronize()

        start = time.perf_counter_ns()

        for _ in range(repeat):
            func()

        synchronize()

        end = time.perf_counter_ns()

        elapsed_us = (end - start) / 1e3

        samples_us.append(elapsed_us / repeat)

    return {
        "median_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


# ============================================================
# Benchmark one workload
# ============================================================


def benchmark_case(shape, dtype_name):
    print()
    print("============================================================")

    print(f"shape={shape} dtype={dtype_name}")

    print("============================================================")

    # ========================================================
    # Create tensors
    # ========================================================

    _, a = random_tensor(shape, dtype_name, "nvidia")

    _, b = random_tensor(shape, dtype_name, "nvidia")

    _, c_native = random_tensor(shape, dtype_name, "nvidia")

    _, c_current = random_tensor(shape, dtype_name, "nvidia")

    _, c_execution_context = random_tensor(shape, dtype_name, "nvidia")

    _, c_prebound = random_tensor(shape, dtype_name, "nvidia")

    device_id = c_native.device_id()

    # ========================================================
    # Runtime / backend
    # ========================================================

    runtime = RuntimeAPI(DeviceType.NVIDIA)

    runtime.set_device(device_id)

    backend = NvidiaTritonBackend()

    stream_ptr = runtime.get_context_stream(device_id)

    # ========================================================
    # A. Native integrated
    # ========================================================
    #
    # Production Native CUDA path.
    # ========================================================

    def native_op():
        llaisys.Ops.add(c_native, a, b)

    native_result = benchmark_host_latency(native_op, runtime.device_synchronize)

    # ========================================================
    # B. Current standalone Triton Add
    # ========================================================
    #
    # Each triton_add() call:
    #
    #     metadata validation
    #     wrappers
    #     get_context_stream()
    #     stream-context check
    #     enter stream context
    #     Triton launch
    #     exit stream context
    #
    # This represents standalone Triton operator usage.
    # ========================================================

    def current_op():
        triton_add(c_current, a, b)

    current_result = benchmark_host_latency(current_op, runtime.device_synchronize)

    # ========================================================
    # C. Formal execution-context Triton Add
    # ========================================================
    #
    # IMPORTANT:
    #
    # This tests the actual formal LLAISYS API:
    #
    #     with execution_context(...):
    #
    # The stream context is entered ONCE around the whole
    # benchmark region.
    #
    # triton_add() still performs its normal:
    #
    #     metadata validation
    #     config selection
    #     wrapper construction
    #     get_context_stream()
    #
    # but should detect the active execution context and avoid
    # entering torch.cuda.stream(...) again.
    # ========================================================

    def execution_context_op():
        triton_add(c_execution_context, a, b)

    with execution_context(DeviceType.NVIDIA, device_id=device_id):
        execution_context_result = benchmark_host_latency(execution_context_op, runtime.device_synchronize)

    # ========================================================
    # D. Pre-bound Triton reference
    # ========================================================
    #
    # Prepare:
    #
    #     metadata
    #     numel
    #     config
    #     grid
    #     TritonTensor wrappers
    #     stream context
    #
    # outside the measured operator function.
    #
    # The measured function therefore mainly contains:
    #
    #     Triton Python launcher
    #         +
    #     GPU kernel execution
    #
    # This is still NOT pure GPU kernel-only timing.
    # ========================================================

    numel = 1

    for dim in c_prebound.shape():
        numel *= dim

    config = backend.add_config(numel)

    block_size = config["BLOCK_SIZE"]

    grid = (triton.cdiv(numel, block_size),)

    c_prebound_triton = as_nvidia_triton_tensor(c_prebound)

    a_triton = as_nvidia_triton_tensor(a)

    b_triton = as_nvidia_triton_tensor(b)

    def prebound_op():
        add_kernel[grid](
            c_prebound_triton, a_triton, b_triton, numel, BLOCK_SIZE=block_size, num_warps=config["num_warps"]
        )

    with backend.stream_context(stream_ptr, device_id):
        prebound_result = benchmark_host_latency(prebound_op, runtime.device_synchronize)

    # ========================================================
    # Extract results
    # ========================================================

    native_us = native_result["median_us"]

    current_us = current_result["median_us"]

    execution_context_us = execution_context_result["median_us"]

    prebound_us = prebound_result["median_us"]

    # ========================================================
    # Derived values
    # ========================================================

    saved_us = current_us - execution_context_us

    remaining_us = execution_context_us - prebound_us

    if current_us > 0:
        saved_percent = saved_us / current_us * 100.0
    else:
        saved_percent = 0.0

    # ========================================================
    # Print detailed result
    # ========================================================

    print()
    print("Operator latency:")

    print(f"  Native integrated:        {native_us:8.3f} us")

    print(f"  Triton standalone:        {current_us:8.3f} us")

    print(f"  Triton execution context: {execution_context_us:8.3f} us")

    print(f"  Triton pre-bound:         {prebound_us:8.3f} us")

    print()
    print("Execution-context effect:")

    print(f"  Saved latency:            {saved_us:8.3f} us")

    print(f"  Saved percentage:         {saved_percent:8.2f} %")

    print(f"  ExecCtx - prebound:       {remaining_us:8.3f} us")

    return {
        "shape": shape,
        "dtype": dtype_name,
        "native_us": native_us,
        "current_us": current_us,
        "execution_context_us": execution_context_us,
        "prebound_us": prebound_us,
        "saved_us": saved_us,
        "saved_percent": saved_percent,
        "remaining_us": remaining_us,
    }


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("============================================================")

    print("LLAISYS Triton Formal Execution-Context Benchmark")

    print("============================================================")

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
        result = benchmark_case(shape, dtype_name)

        results.append(result)

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("============================================================")

    print("Summary")

    print("============================================================")

    print(
        f"{'Shape':>16} "
        f"{'DType':>6} "
        f"{'Native':>10} "
        f"{'Standalone':>11} "
        f"{'ExecCtx':>10} "
        f"{'Prebound':>10} "
        f"{'Saved':>10} "
        f"{'Saved%':>9}"
    )

    print("-" * 94)

    for result in results:
        print(
            f"{result['shape']!s:>16} "
            f"{result['dtype']:>6} "
            f"{result['native_us']:>9.2f}u "
            f"{result['current_us']:>10.2f}u "
            f"{result['execution_context_us']:>9.2f}u "
            f"{result['prebound_us']:>9.2f}u "
            f"{result['saved_us']:>9.2f}u "
            f"{result['saved_percent']:>8.2f}%"
        )

    # ========================================================
    # Remaining integration gap
    # ========================================================

    print()
    print("============================================================")

    print("Remaining Integration Gap")

    print("============================================================")

    print(f"{'Shape':>16} {'DType':>6} {'ExecCtx':>10} {'Prebound':>10} {'Remaining':>12}")

    print("-" * 62)

    for result in results:
        print(
            f"{result['shape']!s:>16} "
            f"{result['dtype']:>6} "
            f"{result['execution_context_us']:>9.2f}u "
            f"{result['prebound_us']:>9.2f}u "
            f"{result['remaining_us']:>11.2f}u"
        )

    # ========================================================
    # Notes
    # ========================================================

    print()
    print("NOTE:")

    print("  Standalone:")

    print("    triton_add() manages the LLAISYS stream per call.")

    print()
    print("  ExecCtx:")

    print("    execution_context() binds the LLAISYS stream once")

    print("    around the full benchmark region.")

    print()
    print("  Prebound:")

    print("    metadata/config/wrappers are also prepared outside")

    print("    the measured operator invocation.")

    print()
    print("  These measurements are host-visible operator latency,")

    print("  not pure GPU kernel latency.")

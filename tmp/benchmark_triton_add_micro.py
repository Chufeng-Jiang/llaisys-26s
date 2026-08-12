import statistics
import sys
import time
from pathlib import Path

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
# LLAISYS imports
# ============================================================

import llaisys

from llaisys.libllaisys import DeviceType
from llaisys.runtime import RuntimeAPI

from llaisys.triton.backends.nvidia import NvidiaTritonBackend
from llaisys.triton.kernels.add import add_kernel
from llaisys.triton.ops import add as triton_add
from llaisys.triton.tensor import as_nvidia_triton_tensor

from test_utils import random_tensor


# ============================================================
# Host-visible operator benchmark
# ============================================================
#
# Measures:
#
#     Python dispatch
#     + runtime integration
#     + Triton/native launch
#     + GPU execution
#
# Synchronization happens only at round boundaries.
#
# This is NOT pure GPU kernel latency.
# ============================================================


def benchmark_host_latency(
    func,
    synchronize,
    warmup=20,
    repeat=100,
    rounds=10,
):
    for _ in range(warmup):
        func()

    synchronize()

    samples_ms = []

    for _ in range(rounds):
        synchronize()

        start = time.perf_counter_ns()

        for _ in range(repeat):
            func()

        synchronize()

        end = time.perf_counter_ns()

        elapsed_ms = (end - start) / 1e6

        samples_ms.append(
            elapsed_ms / repeat
        )

    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


# ============================================================
# CPU-side component benchmark
# ============================================================
#
# Returned values:
#
#     microseconds / invocation
#
# Used for measuring Python / ctypes / C API overhead.
# ============================================================


def benchmark_cpu_component(
    func,
    warmup=200,
    repeat=10000,
    rounds=7,
):
    for _ in range(warmup):
        func()

    samples_us = []

    for _ in range(rounds):
        start = time.perf_counter_ns()

        for _ in range(repeat):
            func()

        end = time.perf_counter_ns()

        elapsed_us = (end - start) / 1e3

        samples_us.append(
            elapsed_us / repeat
        )

    return {
        "median_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


# ============================================================
# Full cached stream bridge benchmark
# ============================================================
#
# Measures:
#
#     get_context_stream()
#       +
#     cached ExternalStream lookup
#       +
#     torch.cuda.stream context enter/exit
#
# No Triton kernel is launched.
# ============================================================


def benchmark_stream_bridge(
    runtime,
    backend,
    device_id,
):
    def bridge_once():
        stream_ptr = runtime.get_context_stream(
            device_id
        )

        with backend.stream_context(
            stream_ptr,
            device_id,
        ):
            pass

    return benchmark_cpu_component(
        bridge_once,
    )


# ============================================================
# Benchmark one Add workload
# ============================================================


def benchmark_add(
    shape,
    dtype_name,
):
    print()
    print(
        "============================================================"
    )

    print(
        f"shape={shape} "
        f"dtype={dtype_name}"
    )

    print(
        "============================================================"
    )

    # ========================================================
    # Create tensors
    # ========================================================

    _, a = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, b = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, c_native = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, c_integrated = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    _, c_prebound = random_tensor(
        shape,
        dtype_name,
        "nvidia",
    )

    device_id = c_native.device_id()

    # ========================================================
    # Runtime / backend
    # ========================================================

    runtime = RuntimeAPI(
        DeviceType.NVIDIA
    )

    runtime.set_device(
        device_id
    )

    backend = NvidiaTritonBackend()

    stream_ptr = runtime.get_context_stream(
        device_id
    )

    # ========================================================
    # A. Native integrated Add
    # ========================================================

    def native_op():
        llaisys.Ops.add(
            c_native,
            a,
            b,
        )

    native_result = benchmark_host_latency(
        native_op,
        runtime.device_synchronize,
    )

    # ========================================================
    # B. Triton fully integrated Add
    # ========================================================
    #
    # Real current Triton path:
    #
    #     validation
    #     metadata queries
    #     numel calculation
    #     backend config
    #     TritonTensor wrappers
    #     get_context_stream
    #     stream context
    #     Triton launch
    #
    # ========================================================

    def triton_integrated_op():
        triton_add(
            c_integrated,
            a,
            b,
        )

    integrated_result = benchmark_host_latency(
        triton_integrated_op,
        runtime.device_synchronize,
    )

    # ========================================================
    # C. Prepare Triton pre-bound path
    # ========================================================

    numel = 1

    for dim in c_prebound.shape():
        numel *= dim

    config = backend.add_config(
        numel
    )

    block_size = config["BLOCK_SIZE"]

    grid = (
        triton.cdiv(
            numel,
            block_size,
        ),
    )

    c_prebound_triton = as_nvidia_triton_tensor(
        c_prebound
    )

    a_triton = as_nvidia_triton_tensor(
        a
    )

    b_triton = as_nvidia_triton_tensor(
        b
    )

    # ========================================================
    # D. Triton pre-bound stream
    # ========================================================
    #
    # The stream context is entered exactly once.
    #
    # The measured loop contains primarily:
    #
    #     Python Triton launcher
    #         ↓
    #     Triton kernel
    #
    # Still NOT pure GPU kernel latency.
    # ========================================================

    def triton_prebound_op():
        add_kernel[grid](
            c_prebound_triton,
            a_triton,
            b_triton,
            numel,
            BLOCK_SIZE=block_size,
            num_warps=config["num_warps"],
        )

    with backend.stream_context(
        stream_ptr,
        device_id,
    ):
        prebound_result = benchmark_host_latency(
            triton_prebound_op,
            runtime.device_synchronize,
        )

    # Make sure all previous GPU work has completed before
    # measuring host-only components.
    runtime.device_synchronize()

    # ========================================================
    # E. Metadata validation
    # ========================================================
    #
    # Mirrors the optimized current Triton Add path:
    #
    # each metadata field is fetched once per tensor.
    # ========================================================

    def metadata_only():
        c_shape = c_integrated.shape()
        a_shape = a.shape()
        b_shape = b.shape()

        c_dtype = c_integrated.dtype()
        a_dtype = a.dtype()
        b_dtype = b.dtype()

        c_device_type = c_integrated.device_type()
        a_device_type = a.device_type()
        b_device_type = b.device_type()

        c_device_id = c_integrated.device_id()
        a_device_id = a.device_id()
        b_device_id = b.device_id()

        if (
            c_shape != a_shape
            or c_shape != b_shape
        ):
            raise ValueError(
                "shape mismatch"
            )

        if (
            c_dtype != a_dtype
            or c_dtype != b_dtype
        ):
            raise ValueError(
                "dtype mismatch"
            )

        if (
            c_device_type != DeviceType.NVIDIA
            or a_device_type != DeviceType.NVIDIA
            or b_device_type != DeviceType.NVIDIA
        ):
            raise ValueError(
                "device type mismatch"
            )

        if (
            c_device_id != a_device_id
            or c_device_id != b_device_id
        ):
            raise ValueError(
                "device id mismatch"
            )

        local_numel = 1

        for dim in c_shape:
            local_numel *= dim

        return local_numel

    metadata_result = benchmark_cpu_component(
        metadata_only,
    )

    # ========================================================
    # F. TritonTensor wrappers
    # ========================================================
    #
    # Measures construction of wrappers for:
    #
    #     c
    #     a
    #     b
    #
    # The wrapper itself may query dtype()/data_ptr().
    # ========================================================

    def wrapper_only():
        as_nvidia_triton_tensor(
            c_integrated
        )

        as_nvidia_triton_tensor(
            a
        )

        as_nvidia_triton_tensor(
            b
        )

    wrapper_result = benchmark_cpu_component(
        wrapper_only,
    )

    # ========================================================
    # G. get_context_stream()
    # ========================================================

    def get_stream_only():
        runtime.get_context_stream(
            device_id
        )

    get_stream_result = benchmark_cpu_component(
        get_stream_only,
    )

    # ========================================================
    # H. Cached stream context enter / exit
    # ========================================================

    cached_stream_ptr = runtime.get_context_stream(
        device_id
    )

    # Warm ExternalStream cache.
    with backend.stream_context(
        cached_stream_ptr,
        device_id,
    ):
        pass

    def context_only():
        with backend.stream_context(
            cached_stream_ptr,
            device_id,
        ):
            pass

    context_result = benchmark_cpu_component(
        context_only,
    )

    # ========================================================
    # I. Full stream bridge
    # ========================================================

    bridge_result = benchmark_stream_bridge(
        runtime,
        backend,
        device_id,
    )

    # ========================================================
    # J. Tensor accessor microbenchmarks
    # ========================================================
    #
    # These isolate individual Python Tensor accessors.
    #
    # If these each cross:
    #
    #     Python
    #       ↓
    #     ctypes
    #       ↓
    #     C API
    #       ↓
    #     C++ Tensor
    #
    # even ~1 us each can accumulate significantly when every
    # operator queries multiple fields from multiple tensors.
    # ========================================================

    def shape_only():
        return c_integrated.shape()

    def dtype_only():
        return c_integrated.dtype()

    def device_type_only():
        return c_integrated.device_type()

    def device_id_only():
        return c_integrated.device_id()

    def data_ptr_only():
        return c_integrated.data_ptr()

    shape_result = benchmark_cpu_component(
        shape_only,
    )

    dtype_result = benchmark_cpu_component(
        dtype_only,
    )

    device_type_result = benchmark_cpu_component(
        device_type_only,
    )

    device_id_result = benchmark_cpu_component(
        device_id_only,
    )

    data_ptr_result = benchmark_cpu_component(
        data_ptr_only,
    )

    # ========================================================
    # Convert operator results to microseconds
    # ========================================================

    native_us = (
        native_result["median_ms"]
        * 1000.0
    )

    integrated_us = (
        integrated_result["median_ms"]
        * 1000.0
    )

    prebound_us = (
        prebound_result["median_ms"]
        * 1000.0
    )

    extra_us = (
        integrated_us
        - prebound_us
    )

    # ========================================================
    # Host integration component values
    # ========================================================

    metadata_us = metadata_result[
        "median_us"
    ]

    wrapper_us = wrapper_result[
        "median_us"
    ]

    get_stream_us = get_stream_result[
        "median_us"
    ]

    context_us = context_result[
        "median_us"
    ]

    bridge_us = bridge_result[
        "median_us"
    ]

    # ========================================================
    # Individual Tensor accessor values
    # ========================================================

    shape_us = shape_result[
        "median_us"
    ]

    dtype_us = dtype_result[
        "median_us"
    ]

    device_type_us = device_type_result[
        "median_us"
    ]

    device_id_us = device_id_result[
        "median_us"
    ]

    data_ptr_us = data_ptr_result[
        "median_us"
    ]

    # Approximate cost of accessing the four metadata fields
    # for three tensors.
    #
    # This is only a diagnostic estimate.
    accessor_metadata_estimate_us = (
        3
        * (
            shape_us
            + dtype_us
            + device_type_us
            + device_id_us
        )
    )

    # Approximate accessor work performed while creating three
    # TritonTensor wrappers if each wrapper performs:
    #
    #     dtype()
    #     data_ptr()
    #
    # This is also only diagnostic.
    wrapper_accessor_estimate_us = (
        3
        * (
            dtype_us
            + data_ptr_us
        )
    )

    # ========================================================
    # Print current workload
    # ========================================================

    print()
    print(
        "Operator-level latency:"
    )

    print(
        f"  Native integrated:       "
        f"{native_us:8.3f} us"
    )

    print(
        f"  Triton integrated:       "
        f"{integrated_us:8.3f} us"
    )

    print(
        f"  Triton pre-bound stream: "
        f"{prebound_us:8.3f} us"
    )

    print(
        f"  Integrated - prebound:   "
        f"{extra_us:8.3f} us"
    )

    print()
    print(
        "Integration component breakdown:"
    )

    print(
        f"  Metadata validation:     "
        f"{metadata_us:8.3f} us"
    )

    print(
        f"  TritonTensor wrappers:   "
        f"{wrapper_us:8.3f} us"
    )

    print(
        f"  get_context_stream:      "
        f"{get_stream_us:8.3f} us"
    )

    print(
        f"  stream context only:     "
        f"{context_us:8.3f} us"
    )

    print(
        f"  full stream bridge:      "
        f"{bridge_us:8.3f} us"
    )

    print()
    print(
        "Tensor accessor breakdown:"
    )

    print(
        f"  shape():                 "
        f"{shape_us:8.3f} us"
    )

    print(
        f"  dtype():                 "
        f"{dtype_us:8.3f} us"
    )

    print(
        f"  device_type():           "
        f"{device_type_us:8.3f} us"
    )

    print(
        f"  device_id():             "
        f"{device_id_us:8.3f} us"
    )

    print(
        f"  data_ptr():              "
        f"{data_ptr_us:8.3f} us"
    )

    print()
    print(
        "Diagnostic estimates:"
    )

    print(
        f"  3-tensor metadata "
        f"accessor estimate: "
        f"{accessor_metadata_estimate_us:8.3f} us"
    )

    print(
        f"  3-wrapper "
        f"dtype+data_ptr estimate: "
        f"{wrapper_accessor_estimate_us:8.3f} us"
    )

    # ========================================================
    # Return
    # ========================================================

    return {
        "shape": shape,
        "dtype": dtype_name,

        "native_us": native_us,
        "integrated_us": integrated_us,
        "prebound_us": prebound_us,
        "extra_us": extra_us,

        "metadata_us": metadata_us,
        "wrapper_us": wrapper_us,
        "get_stream_us": get_stream_us,
        "context_us": context_us,
        "bridge_us": bridge_us,

        "shape_us": shape_us,
        "dtype_us": dtype_us,
        "device_type_us": device_type_us,
        "device_id_us": device_id_us,
        "data_ptr_us": data_ptr_us,

        "accessor_metadata_estimate_us":
            accessor_metadata_estimate_us,

        "wrapper_accessor_estimate_us":
            wrapper_accessor_estimate_us,
    }


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print(
        "============================================================"
    )

    print(
        "LLAISYS Triton Add Integration Microbenchmark"
    )

    print(
        "============================================================"
    )

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
        result = benchmark_add(
            shape,
            dtype_name,
        )

        results.append(
            result
        )

    # ========================================================
    # Operator summary
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "Operator Summary"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Shape':>16} "
        f"{'DType':>6} "
        f"{'Native':>10} "
        f"{'Integrated':>12} "
        f"{'Prebound':>10} "
        f"{'Extra':>10}"
    )

    print(
        "-" * 72
    )

    for result in results:
        print(
            f"{str(result['shape']):>16} "
            f"{result['dtype']:>6} "
            f"{result['native_us']:>9.2f}u "
            f"{result['integrated_us']:>11.2f}u "
            f"{result['prebound_us']:>9.2f}u "
            f"{result['extra_us']:>9.2f}u"
        )

    # ========================================================
    # Integration summary
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "Integration Breakdown"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Shape':>16} "
        f"{'DType':>6} "
        f"{'Metadata':>10} "
        f"{'Wrapper':>10} "
        f"{'GetStream':>10} "
        f"{'Context':>10} "
        f"{'Bridge':>10}"
    )

    print(
        "-" * 84
    )

    for result in results:
        print(
            f"{str(result['shape']):>16} "
            f"{result['dtype']:>6} "
            f"{result['metadata_us']:>9.2f}u "
            f"{result['wrapper_us']:>9.2f}u "
            f"{result['get_stream_us']:>9.2f}u "
            f"{result['context_us']:>9.2f}u "
            f"{result['bridge_us']:>9.2f}u"
        )

    # ========================================================
    # Tensor accessor summary
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "Tensor Accessor Breakdown"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Shape':>16} "
        f"{'DType':>6} "
        f"{'Shape':>9} "
        f"{'DType()':>9} "
        f"{'DevType':>9} "
        f"{'DevId':>9} "
        f"{'DataPtr':>9}"
    )

    print(
        "-" * 80
    )

    for result in results:
        print(
            f"{str(result['shape']):>16} "
            f"{result['dtype']:>6} "
            f"{result['shape_us']:>8.2f}u "
            f"{result['dtype_us']:>8.2f}u "
            f"{result['device_type_us']:>8.2f}u "
            f"{result['device_id_us']:>8.2f}u "
            f"{result['data_ptr_us']:>8.2f}u"
        )

    # ========================================================
    # Diagnostic estimate summary
    # ========================================================

    print()
    print(
        "============================================================"
    )

    print(
        "Accessor Diagnostic Estimates"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Shape':>16} "
        f"{'DType':>6} "
        f"{'3xMetadata':>14} "
        f"{'3xWrapper':>14}"
    )

    print(
        "-" * 56
    )

    for result in results:
        print(
            f"{str(result['shape']):>16} "
            f"{result['dtype']:>6} "
            f"{result['accessor_metadata_estimate_us']:>13.2f}u "
            f"{result['wrapper_accessor_estimate_us']:>13.2f}u"
        )

    # ========================================================
    # Notes
    # ========================================================

    print()
    print(
        "NOTE:"
    )

    print(
        "  Integrated - prebound is NOT expected to equal the"
    )

    print(
        "  sum of individual component measurements."
    )

    print(
        "  GPU launches are asynchronous and host/GPU execution"
    )

    print(
        "  can overlap."
    )

    print()
    print(
        "  The accessor estimates are diagnostic only."
    )

    print(
        "  They help determine whether repeated Python -> ctypes"
    )

    print(
        "  -> C++ Tensor metadata queries are a significant"
    )

    print(
        "  source of per-operator overhead."
    )
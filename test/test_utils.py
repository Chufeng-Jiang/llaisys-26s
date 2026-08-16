import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch

import llaisys


def random_tensor(
    shape, dtype_name, device_name, device_id=0, scale=None, bias=None
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.rand(
        shape, dtype=torch_dtype(dtype_name), device=reference_torch_device(device_name, device_id)
    )

    if scale is not None:
        torch_tensor *= scale

    if bias is not None:
        torch_tensor += bias

    llaisys_tensor = llaisys.Tensor(
        shape, dtype=llaisys_dtype(dtype_name), device=llaisys_device(device_name), device_id=device_id
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    bytes_ = torch_tensor.numel() * torch_tensor.element_size()

    api.memcpy_sync(
        llaisys_tensor.data_ptr(), torch_tensor.data_ptr(), bytes_, torch_to_llaisys_memcpy_kind(device_name)
    )

    return torch_tensor, llaisys_tensor


def random_int_tensor(shape, device_name, dtype_name="i64", device_id=0, low=0, high=2):
    torch_tensor = torch.randint(
        low, high, shape, dtype=torch_dtype(dtype_name), device=reference_torch_device(device_name, device_id)
    )

    llaisys_tensor = llaisys.Tensor(
        shape, dtype=llaisys_dtype(dtype_name), device=llaisys_device(device_name), device_id=device_id
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    bytes_ = torch_tensor.numel() * torch_tensor.element_size()

    api.memcpy_sync(
        llaisys_tensor.data_ptr(), torch_tensor.data_ptr(), bytes_, torch_to_llaisys_memcpy_kind(device_name)
    )

    return torch_tensor, llaisys_tensor


def zero_tensor(shape, dtype_name, device_name, device_id=0) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.zeros(
        shape, dtype=torch_dtype(dtype_name), device=reference_torch_device(device_name, device_id)
    )

    llaisys_tensor = llaisys.Tensor(
        shape, dtype=llaisys_dtype(dtype_name), device=llaisys_device(device_name), device_id=device_id
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    bytes_ = torch_tensor.numel() * torch_tensor.element_size()

    api.memcpy_sync(
        llaisys_tensor.data_ptr(), torch_tensor.data_ptr(), bytes_, torch_to_llaisys_memcpy_kind(device_name)
    )

    return torch_tensor, llaisys_tensor


def arrange_tensor(start, end, device_name, device_id=0) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.arange(start, end, device=reference_torch_device(device_name, device_id))

    llaisys_tensor = llaisys.Tensor(
        (end - start,), dtype=llaisys_dtype("i64"), device=llaisys_device(device_name), device_id=device_id
    )

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    bytes_ = torch_tensor.numel() * torch_tensor.element_size()

    api.memcpy_sync(
        llaisys_tensor.data_ptr(), torch_tensor.data_ptr(), bytes_, torch_to_llaisys_memcpy_kind(device_name)
    )

    return torch_tensor, llaisys_tensor


def check_equal(llaisys_result: llaisys.Tensor, torch_answer: torch.Tensor, atol=1e-5, rtol=1e-5, strict=False):
    shape = llaisys_result.shape()
    strides = llaisys_result.strides()

    assert shape == torch_answer.shape

    assert torch_dtype(dtype_name(llaisys_result.dtype())) == torch_answer.dtype

    right = 0

    for i in range(len(shape)):
        if strides[i] > 0:
            right += strides[i] * (shape[i] - 1)
        else:
            raise ValueError("Negative strides are not supported yet")

    result_device_name = device_name(llaisys_result.device_type())

    tmp = torch.zeros(
        (right + 1,),
        dtype=torch_answer.dtype,
        device=reference_torch_device(result_device_name, llaisys_result.device_id()),
    )

    result = torch.as_strided(tmp, shape, strides)

    api = llaisys.RuntimeAPI(llaisys_result.device_type())

    api.memcpy_sync(
        result.data_ptr(),
        llaisys_result.data_ptr(),
        (right + 1) * tmp.element_size(),
        llaisys_to_torch_memcpy_kind(llaisys_result.device_type()),
    )

    # ========================================================
    # Correctness check
    # ========================================================

    if strict:
        mismatch_mask = result != torch_answer

        if not torch.any(mismatch_mask):
            return True

    else:
        close_mask = torch.isclose(result, torch_answer, atol=atol, rtol=rtol)

        if torch.all(close_mask):
            return True

        mismatch_mask = ~close_mask

    # ========================================================
    # Diagnostic representation
    # ========================================================

    result_f32 = result.to(torch.float32)

    answer_f32 = torch_answer.to(torch.float32)

    abs_error = torch.abs(result_f32 - answer_f32)

    denominator = torch.clamp(torch.abs(answer_f32), min=1e-12)

    rel_error = abs_error / denominator

    # ========================================================
    # Mismatch statistics
    # ========================================================

    mismatch_count = int(mismatch_mask.sum().item())

    total_count = int(result.numel())

    mismatch_ratio = mismatch_count / total_count if total_count > 0 else 0.0

    max_abs_error = float(abs_error.max().item())

    max_rel_error = float(rel_error.max().item())

    # ========================================================
    # Find worst FAILED element
    #
    # Important:
    # only search among elements that actually violate
    # torch.isclose(), rather than all elements.
    # ========================================================

    mismatch_abs_error = torch.where(mismatch_mask, abs_error, torch.zeros_like(abs_error))

    flat_error = mismatch_abs_error.reshape(-1)

    worst_flat_index = int(torch.argmax(flat_error).item())

    # ========================================================
    # Convert flat index -> tensor coordinates
    # ========================================================

    remaining = worst_flat_index
    reversed_index = []

    for dim_size in reversed(shape):
        reversed_index.append(remaining % dim_size)

        remaining //= dim_size

    worst_index = tuple(reversed(reversed_index))

    result_value = float(result_f32[worst_index].item())

    answer_value = float(answer_f32[worst_index].item())

    worst_abs_error = float(abs_error[worst_index].item())

    worst_rel_error = float(rel_error[worst_index].item())

    # ========================================================
    # Allowed error at the worst failed element
    # ========================================================

    allowed_error = float(atol) + float(rtol) * abs(answer_value)

    error_over_allowed = worst_abs_error / allowed_error if allowed_error > 0 else float("inf")

    # ========================================================
    # Print compact diagnostics
    # ========================================================

    print()
    print("========================================")
    print("Tensor mismatch diagnostics")
    print("========================================")

    print(f"shape             = {shape}")

    print(f"dtype             = {torch_answer.dtype}")

    print(f"strict            = {strict}")

    print(f"atol              = {atol:.10e}")

    print(f"rtol              = {rtol:.10e}")

    print()

    print(f"mismatch_count    = {mismatch_count}")

    print(f"total_count       = {total_count}")

    print(f"mismatch_ratio    = {mismatch_ratio:.10%}")

    print()

    print(f"max_abs_error     = {max_abs_error:.10e}")

    print(f"max_rel_error     = {max_rel_error:.10e}")

    print()
    print("Worst failed element")
    print("----------------------------------------")

    print(f"flat_index        = {worst_flat_index}")

    print(f"index             = {worst_index}")

    print(f"LLAISYS           = {result_value:.10e}")

    print(f"Torch             = {answer_value:.10e}")

    print(f"abs_error         = {worst_abs_error:.10e}")

    print(f"rel_error         = {worst_rel_error:.10e}")

    if not strict:
        print(f"allowed_error     = {allowed_error:.10e}")

        print(f"error / allowed   = {error_over_allowed:.6f}x")

    print("========================================")
    print()

    return False


def _percentile_linear(samples, percentile):
    if not samples:
        return None

    ordered = sorted(samples)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarize_benchmark_samples(samples_ms):
    mean_ms = statistics.mean(samples_ms)
    median_ms = statistics.median(samples_ms)
    std_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    cv_pct = 0.0 if mean_ms == 0 else std_ms / mean_ms * 100.0

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "cv_pct": cv_pct,
        "min_ms": min(samples_ms),
        "p25_ms": _percentile_linear(samples_ms, 0.25),
        "p75_ms": _percentile_linear(samples_ms, 0.75),
        "max_ms": max(samples_ms),
        "sample_count": len(samples_ms),
        "samples_ms": list(samples_ms),
    }


def _benchmark_function(func, synchronize, warmup=10, repeat=100, rounds=10):
    import time

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

        elapsed_ms = (end - start) / 1_000_000.0
        samples_ms.append(elapsed_ms / repeat)

    return _summarize_benchmark_samples(samples_ms)


def _benchmark_one_round(func, synchronize, repeat):
    import time

    synchronize()
    start = time.perf_counter_ns()

    for _ in range(repeat):
        func()

    synchronize()
    end = time.perf_counter_ns()

    return (end - start) / 1_000_000.0 / repeat


def _benchmark_pair_alternating(
    torch_func, torch_synchronize, llaisys_func, llaisys_synchronize, warmup=10, repeat=100, rounds=10
):
    for _ in range(warmup):
        llaisys_func()

    llaisys_synchronize()

    for _ in range(warmup):
        torch_func()

    torch_synchronize()

    torch_samples_ms = []
    llaisys_samples_ms = []

    for round_index in range(rounds):
        if round_index % 2 == 0:
            llaisys_samples_ms.append(_benchmark_one_round(llaisys_func, llaisys_synchronize, repeat))
            torch_samples_ms.append(_benchmark_one_round(torch_func, torch_synchronize, repeat))
        else:
            torch_samples_ms.append(_benchmark_one_round(torch_func, torch_synchronize, repeat))
            llaisys_samples_ms.append(_benchmark_one_round(llaisys_func, llaisys_synchronize, repeat))

    return {
        "torch": _summarize_benchmark_samples(torch_samples_ms),
        "llaisys": _summarize_benchmark_samples(llaisys_samples_ms),
    }


def _format_benchmark_stats(stats):
    return (
        f"median={stats['median_ms']:.5f} ms, "
        f"mean={stats['mean_ms']:.5f} ms, "
        f"min={stats['min_ms']:.5f} ms, "
        f"max={stats['max_ms']:.5f} ms"
    )


def benchmark_llaisys(llaisys_func, device_name, warmup=10, repeat=100, rounds=10, label=None):
    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    def synchronize():
        api.device_synchronize()

    stats = _benchmark_function(llaisys_func, synchronize, warmup=warmup, repeat=repeat, rounds=rounds)

    prefix = f"{label}: " if label is not None else ""

    print(f"        {prefix}LLAISYS {device_name}: {_format_benchmark_stats(stats)}")

    return stats


def benchmark(
    torch_func, llaisys_func, device_name, warmup=10, repeat=100, rounds=10, benchmark_order="llaisys_then_torch"
):
    valid_orders = {"llaisys_then_torch", "torch_then_llaisys", "alternating"}

    if benchmark_order not in valid_orders:
        raise ValueError(f"Unsupported benchmark order: {benchmark_order}. Expected one of {sorted(valid_orders)}.")

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    def llaisys_synchronize():
        api.device_synchronize()

    if device_name == "metax":
        llaisys_stats = _benchmark_function(
            llaisys_func, llaisys_synchronize, warmup=warmup, repeat=repeat, rounds=rounds
        )

        print(f"        LLAISYS {device_name}: {_format_benchmark_stats(llaisys_stats)}")
        print("        Torch comparison skipped: the MetaX reference tensor is currently on CPU.")

        return {"torch": None, "llaisys": llaisys_stats}

    if device_name == "cpu":

        def torch_synchronize():
            pass

    elif device_name in ("nvidia", "amd"):

        def torch_synchronize():
            torch.cuda.synchronize()

    else:
        raise ValueError(f"Unsupported benchmark device: {device_name}")

    if benchmark_order == "alternating":
        stats = _benchmark_pair_alternating(
            torch_func,
            torch_synchronize,
            llaisys_func,
            llaisys_synchronize,
            warmup=warmup,
            repeat=repeat,
            rounds=rounds,
        )

        torch_stats = stats["torch"]
        llaisys_stats = stats["llaisys"]

    elif benchmark_order == "llaisys_then_torch":
        llaisys_stats = _benchmark_function(
            llaisys_func, llaisys_synchronize, warmup=warmup, repeat=repeat, rounds=rounds
        )

        torch_stats = _benchmark_function(torch_func, torch_synchronize, warmup=warmup, repeat=repeat, rounds=rounds)

    else:
        torch_stats = _benchmark_function(torch_func, torch_synchronize, warmup=warmup, repeat=repeat, rounds=rounds)

        llaisys_stats = _benchmark_function(
            llaisys_func, llaisys_synchronize, warmup=warmup, repeat=repeat, rounds=rounds
        )

    llaisys_name = f"LLAISYS {device_name}:"
    torch_name = f"Torch {device_name}:"

    print(f"        {llaisys_name} {_format_benchmark_stats(llaisys_stats)}")
    print(f"        {torch_name:<{len(llaisys_name)}} {_format_benchmark_stats(torch_stats)}")
    print(f"        Speedup: {torch_stats['median_ms'] / llaisys_stats['median_ms']:.3f}x")

    return {"torch": torch_stats, "llaisys": llaisys_stats}


# ============================================================
# Experiment recording
# ============================================================

EXPERIMENT_SCHEMA_VERSION = "llaisys.experiment.v1"
MICROBENCHMARK_CASE_IDENTITY_VERSION = "llaisys.microbenchmark.case.v2"


_FILENAME_CONFIG_ABBREVIATIONS = {
    "BLOCK_SIZE": "bs",
    "NUM_WARPS": "w",
    "num_warps": "w",
    "NUM_STAGES": "s",
    "num_stages": "s",
    "BLOCK_M": "bm",
    "BLOCK_N": "bn",
    "BLOCK_K": "bk",
    "BLOCK_D": "bd",
    "BLOCK_V": "bv",
    "GROUP_M": "gm",
    "STAGE1_BLOCK_SIZE": "s1bs",
    "STAGE1_NUM_WARPS": "s1w",
    "STAGEN_BLOCK_SIZE": "snbs",
    "STAGEN_NUM_WARPS": "snw",
}


def _sanitize_filename_component(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return text or "unknown"


def format_config_filename_tag(config):
    """Return a stable, human-readable config tag for result filenames."""
    if not config:
        return "default"

    parts = []

    for key, value in config.items():
        short_key = _FILENAME_CONFIG_ABBREVIATIONS.get(str(key), _sanitize_filename_component(key))

        if value is None:
            value_tag = "default"
        elif isinstance(value, bool):
            value_tag = "1" if value else "0"
        else:
            value_tag = _sanitize_filename_component(value)

        parts.append(f"{short_key}{value_tag}")

    return "-".join(parts) if parts else "default"


def build_experiment_output_path(output_dir, *, op, device_name, backend, config=None, timestamp=None):
    """
    Build an automatic JSONL filename for one benchmark run.

    Example:
        add_nvidia_triton_baseline_cfg-bs256-w4_20260815T211318477120Z.jsonl

    The timestamp uses UTC with microseconds so back-to-back runs do not
    accidentally append into the same file.
    """
    output_dir = Path(output_dir)

    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    timestamp_tag = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    backend_name = backend.get("name", "unknown")
    implementation = backend.get("implementation")
    variant = backend.get("variant", "unspecified")

    if implementation and implementation != backend_name:
        backend_tag = f"{backend_name}-{implementation}"
    else:
        backend_tag = backend_name

    config_tag = format_config_filename_tag(config)

    filename = (
        "_".join(
            [
                _sanitize_filename_component(op),
                _sanitize_filename_component(device_name),
                _sanitize_filename_component(backend_tag),
                _sanitize_filename_component(variant),
                f"cfg-{config_tag}",
                timestamp_tag,
            ]
        )
        + ".jsonl"
    )

    return str(output_dir / filename)


KNOWN_DEVICE_VENDORS = {"cpu": "CPU", "nvidia": "NVIDIA", "metax": "MetaX", "amd": "AMD"}


def _optional_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(repo_root, *args):
    if repo_root is None:
        return None

    try:
        result = subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    return result.stdout.strip()


def _run_command(command, timeout=3):
    executable = command[0]

    if not os.path.isabs(executable):
        if shutil.which(executable) is None:
            return None
    elif not os.path.exists(executable):
        return None

    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None

    output_parts = []

    if result.stdout:
        output_parts.append(result.stdout.strip())

    if result.stderr:
        output_parts.append(result.stderr.strip())

    output = "\n".join(part for part in output_parts if part).strip()

    return output or None


def _first_nonempty_line(text):
    if not text:
        return None

    for line in text.splitlines():
        line = line.strip()

        if line:
            return line

    return None


def _regex_value(text, patterns):
    if not text:
        return None

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

        if match:
            value = match.group(1).strip()

            if value:
                return value

    return None


def _tool_record(name, output=None, version=None):
    if version is None:
        version = _first_nonempty_line(output)

    return {"name": name, "available": output is not None, "version": version}


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)

    return target


def _read_first_existing_file(paths):
    for path in paths:
        try:
            candidate = Path(path)

            if candidate.is_file():
                value = candidate.read_text(encoding="utf-8", errors="replace").strip()

                if value:
                    return value
        except OSError:
            pass

    return None


def collect_source_metadata(repo_root=None):
    commit = _git_output(repo_root, "rev-parse", "HEAD")

    status = _git_output(repo_root, "status", "--porcelain")

    branch = _git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")

    return {"git_commit": commit, "git_branch": branch, "git_dirty": None if status is None else bool(status)}


def collect_environment_metadata():
    selected_names = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "METAX_VISIBLE_DEVICES",
        "MACA_PATH",
        "ROCM_PATH",
        "HIP_PATH",
    ]

    selected = {name: os.environ.get(name) for name in selected_names if os.environ.get(name) is not None}

    llaisys_environment = {name: value for name, value in os.environ.items() if name.startswith("LLAISYS_")}

    return {"device_visibility": selected, "llaisys": dict(sorted(llaisys_environment.items()))}


def _collect_nvidia_stack_metadata():
    nvidia_smi_query = _run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])

    driver_version = _first_nonempty_line(nvidia_smi_query)

    nvcc_output = _run_command(["nvcc", "--version"])

    nvcc_version = _regex_value(nvcc_output, [r"release\s+([0-9.]+)", r"V([0-9.]+)"])

    ncu_output = _run_command(["ncu", "--version"])

    nsys_output = _run_command(["nsys", "--version"])

    return {
        "name": "cuda",
        "runtime_api": "cuda",
        "runtime_version": getattr(torch.version, "cuda", None),
        "driver_name": "nvidia",
        "driver_version": driver_version,
        "compiler": _tool_record("nvcc", nvcc_output, nvcc_version),
        "management_tool": _tool_record("nvidia-smi", nvidia_smi_query, None),
        "profilers": {
            "kernel": _tool_record("ncu", ncu_output, None),
            "timeline": _tool_record("nsys", nsys_output, None),
        },
    }


def _collect_metax_stack_metadata():
    mx_smi_output = _run_command(["mx-smi"], timeout=5)

    macainfo_output = _run_command(["macainfo"], timeout=5)

    mxcc_output = _run_command(["mxcc", "--version"])

    mx_smi_version = _regex_value(mx_smi_output, [r"mx-smi\s+version:\s*([^\s|]+)", r"MX-SMI\s+([^\s|]+)"])

    driver_version = _regex_value(mx_smi_output, [r"Kernel\s+Mode\s+Driver\s+Version:\s*([^\s|]+)"])

    maca_version = _regex_value(mx_smi_output, [r"MACA\s+Version:\s*([^\s|]+)"])

    runtime_version = _regex_value(macainfo_output, [r"Runtime\s+Version:\s*([^\s|]+)"])

    if runtime_version is None:
        runtime_version = maca_version

    mxcc_version = _regex_value(
        mxcc_output, [r"(?:MXCC|mxcc)[^0-9]*([0-9]+(?:\.[0-9]+)+)", r"version[^0-9]*([0-9]+(?:\.[0-9]+)+)"]
    )

    mcprofiler_output = None
    mcprofiler_name = "mcProfiler"

    for candidate in ("mcprofiler", "mcProfiler"):
        candidate_output = _run_command([candidate, "--version"])

        if candidate_output is not None:
            mcprofiler_output = candidate_output
            mcprofiler_name = candidate
            break

    return {
        "name": "maca",
        "runtime_api": "mxmaca",
        "runtime_version": runtime_version,
        "driver_name": "metax",
        "driver_version": driver_version,
        "compiler": _tool_record("mxcc", mxcc_output, mxcc_version),
        "management_tool": _tool_record("mx-smi", mx_smi_output, mx_smi_version),
        "profilers": {
            "kernel": _tool_record(mcprofiler_name, mcprofiler_output, None),
            "timeline": {"name": None, "available": False, "version": None},
        },
        "vendor_specific": {"maca_version": maca_version},
    }


def _collect_amd_stack_metadata():
    amd_smi_output = _run_command(["amd-smi", "version"], timeout=5)

    if amd_smi_output is None:
        amd_smi_output = _run_command(["amd-smi"], timeout=5)

    hipconfig_output = _run_command(["hipconfig", "--full"], timeout=5)

    hipcc_output = _run_command(["hipcc", "--version"], timeout=5)

    rocm_version = _regex_value(amd_smi_output, [r"ROCm\s+version:\s*([^|\n]+)"])

    if rocm_version is None:
        rocm_version = _read_first_existing_file(["/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"])

    driver_version = _regex_value(amd_smi_output, [r"amdgpu\s+version:\s*([^|\n]+)"])

    hip_runtime_version = getattr(torch.version, "hip", None)

    if hip_runtime_version is None:
        hip_runtime_version = _regex_value(
            hipconfig_output, [r"HIP\s+version\s*:\s*([^\s]+)", r"HIP\s+version\s*([^\s]+)"]
        )

    hipcc_version = _regex_value(hipcc_output, [r"HIP\s+version\s*:\s*([^\s]+)", r"clang\s+version\s+([^\s]+)"])

    rocprofv3_output = _run_command(["rocprofv3", "--version"])

    rocprof_output = None

    if rocprofv3_output is None:
        rocprof_output = _run_command(["rocprof", "--version"])

    profiler_name = "rocprofv3" if rocprofv3_output is not None else "rocprof"

    profiler_output = rocprofv3_output if rocprofv3_output is not None else rocprof_output

    return {
        "name": "rocm",
        "runtime_api": "hip",
        "runtime_version": hip_runtime_version,
        "stack_version": rocm_version,
        "driver_name": "amdgpu",
        "driver_version": driver_version,
        "compiler": _tool_record("hipcc", hipcc_output, hipcc_version),
        "management_tool": _tool_record(
            "amd-smi",
            amd_smi_output,
            _regex_value(amd_smi_output, [r"AMDSMI\s+Tool:\s*([^|\n]+)", r"Version:\s*([^|\n]+)"]),
        ),
        "profilers": {
            "kernel": _tool_record(profiler_name, profiler_output, None),
            "timeline": {
                "name": profiler_name,
                "available": profiler_output is not None,
                "version": _first_nonempty_line(profiler_output),
            },
        },
    }


def collect_software_metadata(device_name=None, overrides=None):
    software = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": _optional_package_version("triton"),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "llaisys_module": getattr(llaisys, "__file__", None),
        "accelerator_stack": None,
    }

    if device_name == "nvidia":
        software["accelerator_stack"] = _collect_nvidia_stack_metadata()
    elif device_name == "metax":
        software["accelerator_stack"] = _collect_metax_stack_metadata()
    elif device_name == "amd":
        software["accelerator_stack"] = _collect_amd_stack_metadata()
    elif device_name == "cpu":
        software["accelerator_stack"] = {
            "name": "cpu",
            "runtime_api": "host",
            "runtime_version": None,
            "driver_name": None,
            "driver_version": None,
            "compiler": {"name": None, "available": False, "version": None},
            "management_tool": {"name": None, "available": False, "version": None},
            "profilers": {},
        }

    if overrides:
        _deep_update(software, overrides)

    return software


def collect_host_metadata():
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
    }


def _nvidia_device_metadata(device_id):
    metadata = {}

    if torch.cuda.is_available() and getattr(torch.version, "cuda", None) is not None:
        try:
            properties = torch.cuda.get_device_properties(device_id)

            major, minor = torch.cuda.get_device_capability(device_id)

            metadata.update(
                {
                    "model": properties.name,
                    "architecture": f"sm_{major}{minor}",
                    "total_memory_bytes": int(properties.total_memory),
                    "multiprocessor_count": int(properties.multi_processor_count),
                }
            )
        except Exception:
            pass

    query_output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,pci.bus_id,uuid,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            str(device_id),
        ]
    )

    if query_output:
        values = [item.strip() for item in query_output.splitlines()[0].split(",")]

        if len(values) >= 4:
            metadata.setdefault("model", values[0] or None)
            metadata["pci_bus_id"] = values[1] or None
            metadata["uuid"] = values[2] or None

            try:
                memory_mib = float(values[3])
                metadata.setdefault("total_memory_bytes", int(memory_mib * 1024 * 1024))
            except ValueError:
                pass

    return metadata


def _metax_device_metadata(device_id):
    metadata = {}

    macainfo_output = _run_command(["macainfo"], timeout=5)

    mx_smi_output = _run_command(["mx-smi"], timeout=5)

    model = _regex_value(
        macainfo_output, [r"(?:Device|Board|Product|Marketing)\s+Name\s*:\s*(.+)", r"Device\s+Model\s*:\s*(.+)"]
    )

    if model is not None:
        metadata["model"] = model

    architecture = _regex_value(macainfo_output, [r"(?:Architecture|Arch)\s*:\s*(.+)"])

    if architecture is not None:
        metadata["architecture"] = architecture

    pci_bus_id = _regex_value(mx_smi_output, [r"Bus-id\s*[:=]\s*([0-9A-Fa-f:.]+)"])

    if pci_bus_id is not None:
        metadata["pci_bus_id"] = pci_bus_id

    metadata["vendor_specific"] = {"device_index_requested": device_id}

    return metadata


def _amd_device_metadata(device_id):
    metadata = {}

    if torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None:
        try:
            properties = torch.cuda.get_device_properties(device_id)

            metadata["model"] = properties.name
            metadata["total_memory_bytes"] = int(properties.total_memory)
            metadata["multiprocessor_count"] = int(properties.multi_processor_count)

            gcn_arch_name = getattr(properties, "gcnArchName", None)

            if gcn_arch_name:
                metadata["architecture"] = gcn_arch_name
        except Exception:
            pass

    if metadata.get("architecture") is None:
        rocminfo_output = _run_command(["rocminfo"], timeout=5)

        architecture = _regex_value(rocminfo_output, [r"Name:\s*(gfx[0-9A-Za-z:+_-]+)"])

        if architecture is not None:
            metadata["architecture"] = architecture

    return metadata


def collect_device_metadata(device_name, device_id=0, overrides=None):
    metadata = {
        "type": device_name,
        "vendor": KNOWN_DEVICE_VENDORS.get(device_name, device_name),
        "id": device_id,
        "model": None,
        "architecture": None,
        "pci_bus_id": None,
        "uuid": None,
        "total_memory_bytes": None,
        "multiprocessor_count": None,
        "partition": {
            "kind": None,
            "mode": None,
            "instance": None,
            "compute_partition": None,
            "memory_partition": None,
            "description": None,
        },
        "resource_limits": {"memory_limit_bytes": None, "compute_fraction": None, "power_limit_w": None},
        "vendor_specific": {},
    }

    if device_name == "nvidia":
        _deep_update(metadata, _nvidia_device_metadata(device_id))
    elif device_name == "metax":
        _deep_update(metadata, _metax_device_metadata(device_id))
    elif device_name == "amd":
        _deep_update(metadata, _amd_device_metadata(device_id))
    elif device_name == "cpu":
        metadata["model"] = platform.processor() or None
        metadata["architecture"] = platform.machine()

    if overrides:
        _deep_update(metadata, overrides)

    return metadata


def collect_backend_metadata(backend_name, device_name, variant="unspecified", implementation=None):
    if implementation is None:
        if backend_name == "triton":
            implementation = "triton"
        elif backend_name == "native":
            implementation = {"cpu": "cpu", "nvidia": "cuda", "metax": "maca", "amd": "hip"}.get(device_name, "native")
        else:
            implementation = backend_name

    return {"name": backend_name, "implementation": implementation, "variant": variant}


def _normalize_json_value(value):
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    return value


def make_case_id(identity):
    normalized = _normalize_json_value(identity)

    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_microbenchmark_case_identity(*, op, backend, device, workload, config):
    workload_identity = {key: value for key, value in workload.items() if key != "seed"}

    device_identity = {
        "type": device.get("type"),
        "vendor": device.get("vendor"),
        "id": device.get("id"),
        "model": device.get("model"),
        "architecture": device.get("architecture"),
        "pci_bus_id": device.get("pci_bus_id"),
        "uuid": device.get("uuid"),
        "total_memory_bytes": device.get("total_memory_bytes"),
        "partition": deepcopy(device.get("partition", {})),
        "resource_limits": deepcopy(device.get("resource_limits", {})),
    }

    return {
        "identity_version": (MICROBENCHMARK_CASE_IDENTITY_VERSION),
        "op": op,
        "backend": deepcopy(backend),
        "device": device_identity,
        "workload": workload_identity,
        "config": {"status": config.get("status"), "values": deepcopy(config.get("values", {}))},
    }


def append_jsonl_record(output_path, record):
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as handle:
        json.dump(_normalize_json_value(record), handle, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


class ExperimentRecorder:
    def __init__(self, output_path=None, repo_root=None, run_id=None, run_metadata=None, software_overrides=None):
        self.output_path = output_path
        self.run_id = run_id or uuid.uuid4().hex
        self.source = collect_source_metadata(repo_root)
        self.host = collect_host_metadata()
        self.environment = collect_environment_metadata()
        self.run_metadata = dict(run_metadata or {})
        self.software_overrides = deepcopy(software_overrides or {})
        self._software_cache = {}
        self._device_cache = {}

    @property
    def enabled(self):
        return self.output_path is not None

    def _software(self, device_name):
        if device_name not in self._software_cache:
            self._software_cache[device_name] = collect_software_metadata(
                device_name, overrides=self.software_overrides
            )

        return deepcopy(self._software_cache[device_name])

    def _device(self, device_name, device_id, overrides=None):
        cache_key = (device_name, int(device_id))

        if cache_key not in self._device_cache:
            self._device_cache[cache_key] = collect_device_metadata(device_name, device_id)

        device = deepcopy(self._device_cache[cache_key])

        if overrides:
            _deep_update(device, overrides)

        return device

    def record_microbenchmark(
        self,
        *,
        op,
        backend_name,
        backend_variant,
        backend_implementation,
        suite,
        device_name,
        device_id,
        shape,
        numel,
        dtype_name,
        seed,
        config,
        config_status,
        warmup,
        repeat,
        rounds,
        stats,
        benchmark_order="llaisys_then_torch",
        derived=None,
        workload_metadata=None,
        device_metadata=None,
    ):
        if not self.enabled:
            return None

        device = self._device(device_name, device_id, overrides=device_metadata)

        backend = collect_backend_metadata(
            backend_name, device_name, variant=backend_variant, implementation=backend_implementation
        )

        workload = {"shape": list(shape), "numel": int(numel), "dtype": dtype_name, "seed": int(seed)}

        if workload_metadata:
            workload.update(workload_metadata)

        config_record = {"status": config_status, "values": deepcopy(config)}

        case_identity = make_microbenchmark_case_identity(
            op=op, backend=backend, device=device, workload=workload, config=config_record
        )

        llaisys_stats = stats["llaisys"]

        torch_stats = stats.get("torch")

        speedup_median = None

        if torch_stats is not None:
            speedup_median = torch_stats["median_ms"] / llaisys_stats["median_ms"]

        record = {
            "schema_version": (EXPERIMENT_SCHEMA_VERSION),
            "record_type": "microbenchmark",
            "record_id": uuid.uuid4().hex,
            "run_id": self.run_id,
            "case_id": make_case_id(case_identity),
            "case_identity": case_identity,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "op": op,
            "backend": backend,
            "suite": suite,
            "device": device,
            "software": self._software(device_name),
            "source": self.source,
            "host": self.host,
            "environment": self.environment,
            "run_metadata": self.run_metadata,
            "workload": workload,
            "config": config_record,
            "protocol": {
                "warmup": int(warmup),
                "repeat": int(repeat),
                "rounds": int(rounds),
                "benchmark_order": (benchmark_order),
                "timing": ("host_wall_clock_with_device_synchronize"),
                "synchronization": ("device_synchronize_around_each_timed_round"),
            },
            "metrics": {"llaisys": llaisys_stats, "torch": torch_stats, "speedup_median": (speedup_median)},
            "derived": dict(derived or {}),
        }

        append_jsonl_record(self.output_path, record)

        return record

    def record_external(
        self, *, record_type, case_id, payload, case_identity=None, op=None, backend=None, device=None, suite=None
    ):
        if not self.enabled:
            return None

        if record_type not in {"gpu_profile", "compiler", "serving"}:
            raise ValueError(f"Unsupported external record type: {record_type}")

        record = {
            "schema_version": (EXPERIMENT_SCHEMA_VERSION),
            "record_type": record_type,
            "record_id": uuid.uuid4().hex,
            "run_id": self.run_id,
            "case_id": case_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "host": self.host,
            "environment": self.environment,
            "run_metadata": self.run_metadata,
        }

        if case_identity is not None:
            record["case_identity"] = case_identity

        if op is not None:
            record["op"] = op

        if backend is not None:
            record["backend"] = backend

        if device is not None:
            record["device"] = device

        if suite is not None:
            record["suite"] = suite

        record[record_type] = deepcopy(payload)

        append_jsonl_record(self.output_path, record)

        return record


BenchmarkRecorder = ExperimentRecorder


def torch_device(device_name: str, device_id=0):
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")

    if device_name == "amd":
        if getattr(torch.version, "hip", None) is None:
            raise RuntimeError("AMD reference execution requires a ROCm-enabled PyTorch build.")

        return torch.device(f"cuda:{device_id}")

    raise ValueError(f"Unsupported Torch device name: {device_name}")


def reference_torch_device(device_name: str, device_id=0):
    if device_name == "metax":
        return torch.device("cpu")

    return torch_device(device_name, device_id)


def torch_to_llaisys_memcpy_kind(device_name: str):
    if device_name == "metax":
        return llaisys.MemcpyKind.H2D

    return llaisys.MemcpyKind.D2D


def _optional_llaisys_device_type(name):
    return getattr(llaisys.DeviceType, name, None)


def llaisys_to_torch_memcpy_kind(device_type: llaisys.DeviceType):
    if device_type == llaisys.DeviceType.METAX:
        return llaisys.MemcpyKind.D2H

    amd_device_type = _optional_llaisys_device_type("AMD")

    if amd_device_type is not None and device_type == amd_device_type:
        return llaisys.MemcpyKind.D2D

    return llaisys.MemcpyKind.D2D


def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU

    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA

    if device_name == "metax":
        return llaisys.DeviceType.METAX

    if device_name == "amd":
        amd_device_type = _optional_llaisys_device_type("AMD")

        if amd_device_type is None:
            raise RuntimeError(
                "AMD experiment/schema support is ready, but this LLAISYS build does not yet expose DeviceType.AMD."
            )

        return amd_device_type

    raise ValueError(f"Unsupported device name: {device_name}")


def device_name(llaisys_device: llaisys.DeviceType):
    if llaisys_device == llaisys.DeviceType.CPU:
        return "cpu"

    if llaisys_device == llaisys.DeviceType.NVIDIA:
        return "nvidia"

    if llaisys_device == llaisys.DeviceType.METAX:
        return "metax"

    amd_device_type = _optional_llaisys_device_type("AMD")

    if amd_device_type is not None and llaisys_device == amd_device_type:
        return "amd"

    raise ValueError(f"Unsupported llaisys device: {llaisys_device}")


def torch_dtype(dtype_name: str):
    if dtype_name == "f16":
        return torch.float16
    elif dtype_name == "f32":
        return torch.float32
    elif dtype_name == "f64":
        return torch.float64
    elif dtype_name == "bf16":
        return torch.bfloat16
    elif dtype_name == "i32":
        return torch.int32
    elif dtype_name == "i64":
        return torch.int64
    elif dtype_name == "u32":
        return torch.uint32
    elif dtype_name == "u64":
        return torch.uint64
    elif dtype_name == "bool":
        return torch.bool
    else:
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


def llaisys_dtype(dtype_name: str):
    if dtype_name == "f16":
        return llaisys.DataType.F16
    elif dtype_name == "f32":
        return llaisys.DataType.F32
    elif dtype_name == "f64":
        return llaisys.DataType.F64
    elif dtype_name == "bf16":
        return llaisys.DataType.BF16
    elif dtype_name == "i32":
        return llaisys.DataType.I32
    elif dtype_name == "i64":
        return llaisys.DataType.I64
    elif dtype_name == "u32":
        return llaisys.DataType.U32
    elif dtype_name == "u64":
        return llaisys.DataType.U64
    elif dtype_name == "bool":
        return llaisys.DataType.BOOL
    else:
        raise ValueError(f"Unsupported llaisys dtype: {dtype_name}")


def dtype_name(llaisys_dtype: llaisys.DataType):
    if llaisys_dtype == llaisys.DataType.F16:
        return "f16"
    elif llaisys_dtype == llaisys.DataType.F32:
        return "f32"
    elif llaisys_dtype == llaisys.DataType.F64:
        return "f64"
    elif llaisys_dtype == llaisys.DataType.BF16:
        return "bf16"
    elif llaisys_dtype == llaisys.DataType.I32:
        return "i32"
    elif llaisys_dtype == llaisys.DataType.I64:
        return "i64"
    elif llaisys_dtype == llaisys.DataType.U32:
        return "u32"
    elif llaisys_dtype == llaisys.DataType.U64:
        return "u64"
    elif llaisys_dtype == llaisys.DataType.BOOL:
        return "bool"
    else:
        raise ValueError(f"Unsupported llaisys dtype: {llaisys_dtype}")

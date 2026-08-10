import llaisys
import torch


def random_tensor(
    shape, dtype_name, device_name, device_id=0, scale=None, bias=None
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.rand(
        shape,
        dtype=torch_dtype(dtype_name),
        device=reference_torch_device(device_name, device_id),
    )

    if scale is not None:
        torch_tensor *= scale

    if bias is not None:
        torch_tensor += bias

    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(
        llaisys_device(device_name)
    )

    bytes_ = (
        torch_tensor.numel()
        * torch_tensor.element_size()
    )

    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_llaisys_memcpy_kind(
            device_name
        ),
    )

    return torch_tensor, llaisys_tensor


def random_int_tensor(
    shape,
    device_name,
    dtype_name="i64",
    device_id=0,
    low=0,
    high=2,
):
    torch_tensor = torch.randint(
        low,
        high,
        shape,
        dtype=torch_dtype(dtype_name),
        device=reference_torch_device(
            device_name,
            device_id,
        ),
    )

    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(
        llaisys_device(device_name)
    )

    bytes_ = (
        torch_tensor.numel()
        * torch_tensor.element_size()
    )

    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_llaisys_memcpy_kind(
            device_name
        ),
    )

    return torch_tensor, llaisys_tensor


def zero_tensor(
    shape,
    dtype_name,
    device_name,
    device_id=0,
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.zeros(
        shape,
        dtype=torch_dtype(dtype_name),
        device=reference_torch_device(
            device_name,
            device_id,
        ),
    )

    llaisys_tensor = llaisys.Tensor(
        shape,
        dtype=llaisys_dtype(dtype_name),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(
        llaisys_device(device_name)
    )

    bytes_ = (
        torch_tensor.numel()
        * torch_tensor.element_size()
    )

    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_llaisys_memcpy_kind(
            device_name
        ),
    )

    return torch_tensor, llaisys_tensor


def arrange_tensor(
    start,
    end,
    device_name,
    device_id=0,
) -> tuple[torch.Tensor, llaisys.Tensor]:
    torch_tensor = torch.arange(
        start,
        end,
        device=reference_torch_device(
            device_name,
            device_id,
        ),
    )

    llaisys_tensor = llaisys.Tensor(
        (end - start,),
        dtype=llaisys_dtype("i64"),
        device=llaisys_device(device_name),
        device_id=device_id,
    )

    api = llaisys.RuntimeAPI(
        llaisys_device(device_name)
    )

    bytes_ = (
        torch_tensor.numel()
        * torch_tensor.element_size()
    )

    api.memcpy_sync(
        llaisys_tensor.data_ptr(),
        torch_tensor.data_ptr(),
        bytes_,
        torch_to_llaisys_memcpy_kind(
            device_name
        ),
    )

    return torch_tensor, llaisys_tensor


def check_equal(
    llaisys_result: llaisys.Tensor,
    torch_answer: torch.Tensor,
    atol=1e-5,
    rtol=1e-5,
    strict=False,
):
    shape = llaisys_result.shape()
    strides = llaisys_result.strides()

    assert shape == torch_answer.shape

    assert (
        torch_dtype(
            dtype_name(
                llaisys_result.dtype()
            )
        )
        == torch_answer.dtype
    )

    right = 0

    for i in range(len(shape)):
        if strides[i] > 0:
            right += (
                strides[i]
                * (shape[i] - 1)
            )
        else:
            raise ValueError(
                "Negative strides are not "
                "supported yet"
            )

    result_device_name = device_name(
        llaisys_result.device_type()
    )

    tmp = torch.zeros(
        (right + 1,),
        dtype=torch_answer.dtype,
        device=reference_torch_device(
            result_device_name,
            llaisys_result.device_id(),
        ),
    )

    result = torch.as_strided(
        tmp,
        shape,
        strides,
    )

    api = llaisys.RuntimeAPI(
        llaisys_result.device_type()
    )

    api.memcpy_sync(
        result.data_ptr(),
        llaisys_result.data_ptr(),
        (right + 1)
        * tmp.element_size(),
        llaisys_to_torch_memcpy_kind(
            llaisys_result.device_type()
        ),
    )

    # ========================================================
    # Correctness check
    # ========================================================

    if strict:
        mismatch_mask = (
            result != torch_answer
        )

        if not torch.any(mismatch_mask):
            return True

    else:
        close_mask = torch.isclose(
            result,
            torch_answer,
            atol=atol,
            rtol=rtol,
        )

        if torch.all(close_mask):
            return True

        mismatch_mask = ~close_mask

    # ========================================================
    # Diagnostic representation
    # ========================================================

    result_f32 = result.to(
        torch.float32
    )

    answer_f32 = torch_answer.to(
        torch.float32
    )

    abs_error = torch.abs(
        result_f32
        - answer_f32
    )

    denominator = torch.clamp(
        torch.abs(
            answer_f32
        ),
        min=1e-12,
    )

    rel_error = (
        abs_error
        / denominator
    )

    # ========================================================
    # Mismatch statistics
    # ========================================================

    mismatch_count = int(
        mismatch_mask
        .sum()
        .item()
    )

    total_count = int(
        result.numel()
    )

    mismatch_ratio = (
        mismatch_count
        / total_count
        if total_count > 0
        else 0.0
    )

    max_abs_error = float(
        abs_error
        .max()
        .item()
    )

    max_rel_error = float(
        rel_error
        .max()
        .item()
    )

    # ========================================================
    # Find worst FAILED element
    #
    # Important:
    # only search among elements that actually violate
    # torch.isclose(), rather than all elements.
    # ========================================================

    mismatch_abs_error = torch.where(
        mismatch_mask,
        abs_error,
        torch.zeros_like(
            abs_error
        ),
    )

    flat_error = (
        mismatch_abs_error
        .reshape(-1)
    )

    worst_flat_index = int(
        torch.argmax(
            flat_error
        ).item()
    )

    # ========================================================
    # Convert flat index -> tensor coordinates
    # ========================================================

    remaining = worst_flat_index
    reversed_index = []

    for dim_size in reversed(shape):
        reversed_index.append(
            remaining
            % dim_size
        )

        remaining //= dim_size

    worst_index = tuple(
        reversed(
            reversed_index
        )
    )

    result_value = float(
        result_f32[
            worst_index
        ].item()
    )

    answer_value = float(
        answer_f32[
            worst_index
        ].item()
    )

    worst_abs_error = float(
        abs_error[
            worst_index
        ].item()
    )

    worst_rel_error = float(
        rel_error[
            worst_index
        ].item()
    )

    # ========================================================
    # Allowed error at the worst failed element
    # ========================================================

    allowed_error = (
        float(atol)
        + float(rtol)
        * abs(answer_value)
    )

    error_over_allowed = (
        worst_abs_error
        / allowed_error
        if allowed_error > 0
        else float("inf")
    )

    # ========================================================
    # Print compact diagnostics
    # ========================================================

    print()
    print(
        "========================================"
    )
    print(
        "Tensor mismatch diagnostics"
    )
    print(
        "========================================"
    )

    print(
        f"shape             = {shape}"
    )

    print(
        f"dtype             = "
        f"{torch_answer.dtype}"
    )

    print(
        f"strict            = {strict}"
    )

    print(
        f"atol              = "
        f"{atol:.10e}"
    )

    print(
        f"rtol              = "
        f"{rtol:.10e}"
    )

    print()

    print(
        f"mismatch_count    = "
        f"{mismatch_count}"
    )

    print(
        f"total_count       = "
        f"{total_count}"
    )

    print(
        f"mismatch_ratio    = "
        f"{mismatch_ratio:.10%}"
    )

    print()

    print(
        f"max_abs_error     = "
        f"{max_abs_error:.10e}"
    )

    print(
        f"max_rel_error     = "
        f"{max_rel_error:.10e}"
    )

    print()
    print(
        "Worst failed element"
    )
    print(
        "----------------------------------------"
    )

    print(
        f"flat_index        = "
        f"{worst_flat_index}"
    )

    print(
        f"index             = "
        f"{worst_index}"
    )

    print(
        f"LLAISYS           = "
        f"{result_value:.10e}"
    )

    print(
        f"Torch             = "
        f"{answer_value:.10e}"
    )

    print(
        f"abs_error         = "
        f"{worst_abs_error:.10e}"
    )

    print(
        f"rel_error         = "
        f"{worst_rel_error:.10e}"
    )

    if not strict:
        print(
            f"allowed_error     = "
            f"{allowed_error:.10e}"
        )

        print(
            f"error / allowed   = "
            f"{error_over_allowed:.6f}x"
        )

    print(
        "========================================"
    )
    print()

    return False

def _benchmark_function(
    func,
    synchronize,
    warmup=10,
    repeat=100,
    rounds=10,
):
    import statistics
    import time

    # Warm up.
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

        elapsed_ms = (
            end - start
        ) / 1_000_000.0

        samples_ms.append(
            elapsed_ms / repeat
        )

    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def benchmark_llaisys(
    llaisys_func,
    device_name,
    warmup=10,
    repeat=100,
    rounds=10,
    label=None,
):
    api = llaisys.RuntimeAPI(
        llaisys_device(device_name)
    )

    def synchronize():
        api.device_synchronize()

    stats = _benchmark_function(
        llaisys_func,
        synchronize,
        warmup=warmup,
        repeat=repeat,
        rounds=rounds,
    )

    prefix = (
        f"{label}: "
        if label is not None
        else ""
    )

    print(
        f"        {prefix}"
        f"LLAISYS {device_name}: "
        f"median={stats['median_ms']:.5f} ms, "
        f"mean={stats['mean_ms']:.5f} ms, "
        f"min={stats['min_ms']:.5f} ms, "
        f"max={stats['max_ms']:.5f} ms"
    )

    return stats


def benchmark(
    torch_func,
    llaisys_func,
    device_name,
    warmup=10,
    repeat=100,
    rounds=10,
):
    llaisys_stats = benchmark_llaisys(
        llaisys_func,
        device_name,
        warmup=warmup,
        repeat=repeat,
        rounds=rounds,
    )

    if device_name == "metax":
        print(
            "        Torch comparison skipped: "
            "the reference tensor is on CPU."
        )
        return {
            "torch": None,
            "llaisys": llaisys_stats,
        }

    if device_name == "cpu":

        def torch_synchronize():
            pass

    elif device_name == "nvidia":

        def torch_synchronize():
            torch.cuda.synchronize()

    else:
        raise ValueError(
            f"Unsupported benchmark device: "
            f"{device_name}"
        )

    torch_stats = _benchmark_function(
        torch_func,
        torch_synchronize,
        warmup=warmup,
        repeat=repeat,
        rounds=rounds,
    )

    print(
        f"        Torch {device_name}: "
        f"median={torch_stats['median_ms']:.5f} ms"
    )

    print(
        f"        Speedup: "
        f"{torch_stats['median_ms'] / llaisys_stats['median_ms']:.3f}x"
    )

    return {
        "torch": torch_stats,
        "llaisys": llaisys_stats,
    }
    

def torch_device(device_name: str, device_id=0):
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "nvidia":
        return torch.device(f"cuda:{device_id}")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")

def reference_torch_device(device_name: str, device_id=0):
    if device_name == "metax":
        return torch.device("cpu")

    return torch_device(device_name, device_id)


def torch_to_llaisys_memcpy_kind(device_name: str):
    if device_name == "metax":
        return llaisys.MemcpyKind.H2D

    return llaisys.MemcpyKind.D2D


def llaisys_to_torch_memcpy_kind(device_type: llaisys.DeviceType):
    if device_type == llaisys.DeviceType.METAX:
        return llaisys.MemcpyKind.D2H

    return llaisys.MemcpyKind.D2D



def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    elif device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    elif device_name == "metax":
        return llaisys.DeviceType.METAX
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def device_name(llaisys_device: llaisys.DeviceType):
    if llaisys_device == llaisys.DeviceType.CPU:
        return "cpu"
    elif llaisys_device == llaisys.DeviceType.NVIDIA:
        return "nvidia"
    elif llaisys_device == llaisys.DeviceType.METAX:
        return "metax"
    else:
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
        raise ValueError(f"Unsupported dtype name: {dtype_name}")


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

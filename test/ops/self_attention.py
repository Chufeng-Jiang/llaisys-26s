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
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

import llaisys

from llaisys.triton import execution_context
from llaisys.triton.backends.registry import get_triton_backend
from llaisys.triton.ops import self_attention as triton_self_attention

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
# Constants
# ============================================================

DTYPE_BYTES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
}

TEST_DTYPE_PREC = [
    ("f32", 1e-5, 1e-5),
    ("f16", 1e-3, 1e-3),
    ("bf16", 1e-2, 1e-2),
]


# ============================================================
# PyTorch functional reference
# ============================================================
#
# LLAISYS layout:
#
#     Q   [S_q, H_q, D_qk]
#     K   [S_kv, H_kv, D_qk]
#     V   [S_kv, H_kv, D_v]
#     Out [S_q, H_q, D_v]
#
# F.scaled_dot_product_attention layout:
#
#     Q   [B, H_q, S_q, D_qk]
#     K   [B, H_kv, S_kv, D_qk]
#     V   [B, H_kv, S_kv, D_v]
#
# We add a batch dimension of 1 and use enable_gqa=True whenever
# H_q != H_kv.
#
# LLAISYS uses a bottom-right causal mask for prefix-KV semantics:
#
#     diagonal = S_kv - S_q
#
# torch.nn.attention.bias.causal_lower_right(S_q, S_kv)
# expresses exactly that alignment and can be passed directly to
# F.scaled_dot_product_attention.
#
# We intentionally set:
#
#     dropout_p = 0.0
#     is_causal = False
#
# because the explicit lower-right causal bias is already supplied.
# ============================================================


def make_torch_attention_bias(query_length, key_length):
    return causal_lower_right(
        query_length,
        key_length,
    )


def torch_self_attention(
    query,
    key,
    value,
    scale,
    attn_bias,
):
    query_heads = query.shape[1]
    kv_heads = key.shape[1]

    # [S, H, D] -> [1, H, S, D]
    query_sdpa = (
        query.transpose(0, 1)
        .unsqueeze(0)
    )

    key_sdpa = (
        key.transpose(0, 1)
        .unsqueeze(0)
    )

    value_sdpa = (
        value.transpose(0, 1)
        .unsqueeze(0)
    )

    result = F.scaled_dot_product_attention(
        query_sdpa,
        key_sdpa,
        value_sdpa,
        attn_mask=attn_bias,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=(query_heads != kv_heads),
    )

    # [1, H, S, D_v] -> [S, H, D_v]
    return (
        result.squeeze(0)
        .transpose(0, 1)
    )


# ============================================================
# Backend dispatch
# ============================================================


def run_llaisys_self_attention(
    attn_val,
    q,
    k,
    v,
    scale,
    backend,
):
    if backend == "native":
        llaisys.Ops.self_attention(
            attn_val,
            q,
            k,
            v,
            scale,
        )
        return

    if backend == "triton":
        triton_self_attention(
            attn_val,
            q,
            k,
            v,
            scale,
        )
        return

    raise ValueError(
        f"Unsupported Self-Attention backend: {backend}"
    )


# ============================================================
# Configuration
# ============================================================


def _parse_env_config_value(
    name,
    default="default",
):
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return value


def _device_type_from_name(device_name):
    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA

    if device_name == "metax":
        return llaisys.DeviceType.METAX

    if device_name == "amd":
        if not hasattr(
            llaisys.DeviceType,
            "AMD",
        ):
            raise ValueError(
                "This LLAISYS build does not expose DeviceType.AMD"
            )

        return llaisys.DeviceType.AMD

    raise ValueError(
        f"Unsupported Triton device: {device_name}"
    )


def get_self_attention_config(
    q,
    v,
    backend,
):
    query_length = q.shape()[0]
    qk_dim = q.shape()[2]
    value_dim = v.shape()[2]
    total_len = v.shape()[0]

    if backend == "native":
        return "backend_policy", {
            "query_length": query_length,
        }

    if backend == "triton":
        triton_backend = get_triton_backend(
            q.device_type()
        )

        config = triton_backend.self_attention_config(
            qk_dim,
            value_dim,
            total_len,
        )

        return "effective", {
            "BLOCK_M": config["BLOCK_M"],
            "BLOCK_N": config["BLOCK_N"],
            "BLOCK_D": config["BLOCK_D"],
            "BLOCK_V": config["BLOCK_V"],
            "num_warps": config["num_warps"],
            "num_stages": config["num_stages"],
        }

    raise ValueError(
        f"Unsupported Self-Attention backend: {backend}"
    )


def get_self_attention_config_label(
    q,
    v,
    backend,
):
    config_status, config = (
        get_self_attention_config(
            q,
            v,
            backend,
        )
    )

    if not config:
        return f"config[{config_status}]"

    values = ", ".join(
        f"{key}={value}"
        for key, value in config.items()
    )

    return f"config[{values}]"


def get_self_attention_output_filename_config(
    backend,
):
    if backend == "native":
        return {
            "POLICY": "backend_default",
        }

    if backend == "triton":
        return {
            "BLOCK_M": _parse_env_config_value(
                "LLAISYS_TRITON_BLOCK_M"
            ),
            "BLOCK_N": _parse_env_config_value(
                "LLAISYS_TRITON_BLOCK_N"
            ),
            "BLOCK_D": _parse_env_config_value(
                "LLAISYS_TRITON_BLOCK_D"
            ),
            "BLOCK_V": _parse_env_config_value(
                "LLAISYS_TRITON_BLOCK_V"
            ),
            "NUM_WARPS": _parse_env_config_value(
                "LLAISYS_TRITON_NUM_WARPS"
            ),
            "NUM_STAGES": _parse_env_config_value(
                "LLAISYS_TRITON_NUM_STAGES"
            ),
        }

    raise ValueError(
        f"Unsupported Self-Attention backend: {backend}"
    )


# ============================================================
# Derived performance metrics
# ============================================================
#
# We report two deliberately conservative operator-level models.
#
# 1. Minimum logical I/O:
#
#     read Q
#     read K
#     read V
#     write output
#
# The logical model does NOT expand K/V for GQA and does not count
# implementation-specific score/workspace traffic.
#
# 2. Nominal matmul FLOPs:
#
#     Q @ K^T:
#         2 * H_q * S_q * S_kv * D_qk
#
#     P @ V:
#         2 * H_q * S_q * S_kv * D_v
#
# Softmax/masking FLOPs are intentionally omitted, so the metric is
# called "nominal matmul throughput", not total attention FLOP/s.
# ============================================================


def get_attention_minimum_logical_io_bytes(
    qlen,
    kvlen,
    nh,
    nkvh,
    qk_dim,
    value_dim,
    dtype_name,
):
    element_size = DTYPE_BYTES[dtype_name]

    elements = (
        qlen * nh * qk_dim
        + kvlen * nkvh * qk_dim
        + kvlen * nkvh * value_dim
        + qlen * nh * value_dim
    )

    return elements * element_size


def get_attention_nominal_matmul_flops(
    qlen,
    kvlen,
    nh,
    qk_dim,
    value_dim,
):
    qk_flops = (
        2
        * nh
        * qlen
        * kvlen
        * qk_dim
    )

    pv_flops = (
        2
        * nh
        * qlen
        * kvlen
        * value_dim
    )

    return qk_flops + pv_flops


def get_effective_bandwidth_gbs(
    traffic_bytes,
    median_ms,
):
    return (
        traffic_bytes
        / median_ms
        / 1_000_000.0
    )


def get_throughput_tflops(
    flops,
    median_ms,
):
    return (
        flops
        / median_ms
        / 1_000_000_000.0
    )


def get_self_attention_derived_metrics(
    stats,
    qlen,
    kvlen,
    nh,
    nkvh,
    qk_dim,
    value_dim,
    dtype_name,
):
    logical_bytes = (
        get_attention_minimum_logical_io_bytes(
            qlen,
            kvlen,
            nh,
            nkvh,
            qk_dim,
            value_dim,
            dtype_name,
        )
    )

    nominal_matmul_flops = (
        get_attention_nominal_matmul_flops(
            qlen,
            kvlen,
            nh,
            qk_dim,
            value_dim,
        )
    )

    llaisys_stats = stats["llaisys"]
    torch_stats = stats.get("torch")

    derived = {
        "minimum_logical_io_traffic_bytes": (
            logical_bytes
        ),
        "nominal_matmul_flops": (
            nominal_matmul_flops
        ),
        "llaisys_effective_io_bandwidth_gbs": (
            get_effective_bandwidth_gbs(
                logical_bytes,
                llaisys_stats["median_ms"],
            )
        ),
        "llaisys_nominal_matmul_throughput_tflops": (
            get_throughput_tflops(
                nominal_matmul_flops,
                llaisys_stats["median_ms"],
            )
        ),
    }

    if torch_stats is not None:
        derived.update(
            {
                "torch_equivalent_io_bandwidth_gbs": (
                    get_effective_bandwidth_gbs(
                        logical_bytes,
                        torch_stats["median_ms"],
                    )
                ),
                "torch_nominal_matmul_throughput_tflops": (
                    get_throughput_tflops(
                        nominal_matmul_flops,
                        torch_stats["median_ms"],
                    )
                ),
            }
        )

    return derived


def print_self_attention_derived_metrics(
    derived,
    device_name,
    show_bandwidth,
    show_throughput,
):
    if show_bandwidth:
        print(
            f"        LLAISYS {device_name} "
            f"effective minimum-I/O bandwidth: "
            f"{derived['llaisys_effective_io_bandwidth_gbs']:.2f} GB/s"
        )

        torch_bandwidth = derived.get(
            "torch_equivalent_io_bandwidth_gbs"
        )

        if torch_bandwidth is not None:
            print(
                f"        Torch {device_name} "
                f"equivalent minimum-I/O bandwidth: "
                f"{torch_bandwidth:.2f} GB/s"
            )

    if show_throughput:
        print(
            f"        LLAISYS {device_name} "
            f"nominal matmul throughput: "
            f"{derived['llaisys_nominal_matmul_throughput_tflops']:.3f} TFLOP/s"
        )

        torch_throughput = derived.get(
            "torch_nominal_matmul_throughput_tflops"
        )

        if torch_throughput is not None:
            print(
                f"        Torch {device_name} "
                f"nominal matmul throughput: "
                f"{torch_throughput:.3f} TFLOP/s"
            )


# ============================================================
# Benchmark
# ============================================================


def benchmark_self_attention(
    torch_q,
    torch_k,
    torch_v,
    torch_attn_bias,
    llaisys_out,
    llaisys_q,
    llaisys_k,
    llaisys_v,
    scale,
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
    qlen = llaisys_q.shape()[0]
    kvlen = llaisys_k.shape()[0]
    nh = llaisys_q.shape()[1]
    nkvh = llaisys_k.shape()[1]
    qk_dim = llaisys_q.shape()[2]
    value_dim = llaisys_v.shape()[2]
    group_size = nh // nkvh
    prefix_len = kvlen - qlen

    config_status, config = (
        get_self_attention_config(
            llaisys_q,
            llaisys_v,
            backend,
        )
    )

    label = (
        f"Self-Attention "
        f"qlen={qlen} "
        f"kvlen={kvlen} "
        f"nh={nh} "
        f"nkvh={nkvh} "
        f"group={group_size} "
        f"qk_dim={qk_dim} "
        f"value_dim={value_dim} "
        f"prefix={prefix_len} "
        f"dtype={dtype_name} "
        f"backend={backend}"
    )

    if show_config:
        label += (
            " "
            + get_self_attention_config_label(
                llaisys_q,
                llaisys_v,
                backend,
            )
        )

    print(
        f"        {label}:"
    )

    # The lower-right causal bias is constructed outside the timed
    # region. Only the actual SDPA operator is benchmarked.
    torch_fn = lambda: torch_self_attention(
        torch_q,
        torch_k,
        torch_v,
        scale,
        torch_attn_bias,
    )

    llaisys_fn = lambda: (
        run_llaisys_self_attention(
            llaisys_out,
            llaisys_q,
            llaisys_k,
            llaisys_v,
            scale,
            backend,
        )
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
        raise ValueError(
            f"Unsupported Self-Attention backend: {backend}"
        )

    derived = (
        get_self_attention_derived_metrics(
            stats,
            qlen,
            kvlen,
            nh,
            nkvh,
            qk_dim,
            value_dim,
            dtype_name,
        )
    )

    if show_bandwidth or show_throughput:
        print_self_attention_derived_metrics(
            derived,
            device_name,
            show_bandwidth,
            show_throughput,
        )

    recorder.record_microbenchmark(
        op="self_attention",
        backend_name=backend,
        backend_variant=backend_variant,
        backend_implementation=backend_implementation,
        suite=suite,
        device_name=device_name,
        device_id=llaisys_out.device_id(),
        shape=(qlen, nh, value_dim),
        numel=qlen * nh * value_dim,
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
            "query_length": qlen,
            "kv_length": kvlen,
            "query_heads": nh,
            "kv_heads": nkvh,
            "group_size": group_size,
            "qk_dim": qk_dim,
            "value_dim": value_dim,
            "prefix_length": prefix_len,
            "scale": scale,
            "causal_alignment": "lower_right",
            "torch_reference": (
                "torch.nn.functional.scaled_dot_product_attention"
            ),
            "torch_reference_mask": (
                "torch.nn.attention.bias.causal_lower_right"
            ),
            "torch_reference_gqa": (
                "enable_gqa"
            ),
            "torch_reference_output_policy": (
                "functional_return"
            ),
            "input_distribution": "uniform[0,1)",
        },
        device_metadata=device_metadata,
    )


# ============================================================
# One correctness / performance case
# ============================================================


def test_op_self_attention(
    qlen,
    kvlen,
    nh,
    nkvh,
    qk_dim,
    value_dim,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    backend="native",
    profile=False,
    backend_variant="unspecified",
    backend_implementation=None,
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
    case_name=None,
):
    if qlen < 0:
        raise ValueError(
            "query length must not be negative"
        )

    if kvlen < qlen:
        raise ValueError(
            "Self-Attention test requires kvlen >= qlen "
            "for prefix-KV lower-right causal semantics"
        )

    if nh <= 0:
        raise ValueError(
            "query-head count must be positive"
        )

    if nkvh <= 0:
        raise ValueError(
            "KV-head count must be positive"
        )

    if nh % nkvh != 0:
        raise ValueError(
            "query-head count must be divisible by KV-head count"
        )

    if qk_dim <= 0:
        raise ValueError(
            "Q/K dimension must be positive"
        )

    if value_dim <= 0:
        raise ValueError(
            "value dimension must be positive"
        )

    case_prefix = (
        f"{case_name} "
        if case_name is not None
        else ""
    )

    print(
        f"   {case_prefix}"
        f"qlen={qlen} "
        f"kvlen={kvlen} "
        f"nh={nh} "
        f"nkvh={nkvh} "
        f"qk_dim={qk_dim} "
        f"value_dim={value_dim} "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    # ========================================================
    # Q / K / V
    # ========================================================

    torch_q, llaisys_q = random_tensor(
        (qlen, nh, qk_dim),
        dtype_name,
        device_name,
    )

    torch_k, llaisys_k = random_tensor(
        (kvlen, nkvh, qk_dim),
        dtype_name,
        device_name,
    )

    torch_v, llaisys_v = random_tensor(
        (kvlen, nkvh, value_dim),
        dtype_name,
        device_name,
    )

    scale = (
        1.0
        / math.sqrt(qk_dim)
    )

    # ========================================================
    # PyTorch functional reference
    #
    # Build the causal bias outside the operator call so the
    # same object is reused for correctness and benchmarking.
    # ========================================================

    torch_attn_bias = (
        make_torch_attention_bias(
            qlen,
            kvlen,
        )
    )

    torch_out = torch_self_attention(
        torch_q,
        torch_k,
        torch_v,
        scale,
        torch_attn_bias,
    )

    # ========================================================
    # LLAISYS output
    # ========================================================

    _, llaisys_out = zero_tensor(
        (qlen, nh, value_dim),
        dtype_name,
        device_name,
    )

    run_llaisys_self_attention(
        llaisys_out,
        llaisys_q,
        llaisys_k,
        llaisys_v,
        scale,
        backend,
    )

    # ========================================================
    # Correctness
    #
    # Keep the existing tolerance contract. Do not widen the
    # tolerance merely because a backend fails.
    # ========================================================

    assert check_equal(
        llaisys_out,
        torch_out,
        atol=atol,
        rtol=rtol,
    ), (
        "Self-Attention mismatch: "
        f"case={case_name}, "
        f"qlen={qlen}, "
        f"kvlen={kvlen}, "
        f"nh={nh}, "
        f"nkvh={nkvh}, "
        f"qk_dim={qk_dim}, "
        f"value_dim={value_dim}, "
        f"dtype={dtype_name}, "
        f"device={device_name}, "
        f"backend={backend}"
    )

    if not profile:
        return

    if recorder is None:
        recorder = BenchmarkRecorder()

    benchmark_self_attention(
        torch_q,
        torch_k,
        torch_v,
        torch_attn_bias,
        llaisys_out,
        llaisys_q,
        llaisys_k,
        llaisys_v,
        scale,
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
        device_metadata or {},
    )


# ============================================================
# Correctness cases
# ============================================================


def get_fixed_correctness_cases():
    return [
        # ----------------------------------------------------
        # Empty-query fast path.
        # ----------------------------------------------------
        (
            "empty_query",
            0,
            4,
            2,
            1,
            8,
            8,
        ),

        # ----------------------------------------------------
        # Scalar / tiny MHA.
        # ----------------------------------------------------
        (
            "scalar_mha",
            1,
            1,
            1,
            1,
            1,
            1,
        ),
        (
            "tiny_mha",
            2,
            2,
            1,
            1,
            4,
            4,
        ),

        # ----------------------------------------------------
        # Original GQA + prefix-KV test.
        # group_size = 2
        # prefix = 6
        # ----------------------------------------------------
        (
            "tiny_gqa_prefix",
            5,
            11,
            4,
            2,
            8,
            8,
        ),

        # ----------------------------------------------------
        # MQA.
        # group_size = 8
        # ----------------------------------------------------
        (
            "mqa_prefix",
            3,
            7,
            8,
            1,
            16,
            16,
        ),

        # ----------------------------------------------------
        # Query BLOCK_M boundary around the current baseline 16.
        # Keep KV long enough that lower-right prefix alignment
        # is exercised.
        # ----------------------------------------------------
        (
            "qlen15",
            15,
            47,
            4,
            2,
            32,
            64,
        ),
        (
            "qlen16",
            16,
            48,
            4,
            2,
            32,
            64,
        ),
        (
            "qlen17",
            17,
            49,
            4,
            2,
            32,
            64,
        ),

        # ----------------------------------------------------
        # KV BLOCK_N boundary around 32.
        # ----------------------------------------------------
        (
            "kvlen31",
            7,
            31,
            4,
            2,
            32,
            64,
        ),
        (
            "kvlen32",
            7,
            32,
            4,
            2,
            32,
            64,
        ),
        (
            "kvlen33",
            7,
            33,
            4,
            2,
            32,
            64,
        ),

        # ----------------------------------------------------
        # Q/K BLOCK_D boundary around 32.
        # ----------------------------------------------------
        (
            "qk_dim31",
            7,
            33,
            4,
            2,
            31,
            64,
        ),
        (
            "qk_dim32",
            7,
            33,
            4,
            2,
            32,
            64,
        ),
        (
            "qk_dim33",
            7,
            33,
            4,
            2,
            33,
            64,
        ),

        # ----------------------------------------------------
        # V BLOCK_V boundary around 64.
        # ----------------------------------------------------
        (
            "value_dim63",
            7,
            33,
            4,
            2,
            32,
            63,
        ),
        (
            "value_dim64",
            7,
            33,
            4,
            2,
            32,
            64,
        ),
        (
            "value_dim65",
            7,
            33,
            4,
            2,
            32,
            65,
        ),

        # ----------------------------------------------------
        # Irregular dimensions:
        # qk_dim != value_dim.
        # ----------------------------------------------------
        (
            "irregular_qk31_v37",
            17,
            65,
            4,
            2,
            31,
            37,
        ),

        # ----------------------------------------------------
        # Different GQA group sizes.
        # ----------------------------------------------------
        (
            "mha_group1",
            4,
            12,
            4,
            4,
            16,
            16,
        ),
        (
            "gqa_group2",
            4,
            12,
            4,
            2,
            16,
            16,
        ),
        (
            "gqa_group4",
            4,
            12,
            8,
            2,
            16,
            16,
        ),
        (
            "mqa_group8",
            4,
            12,
            8,
            1,
            16,
            16,
        ),

        # ----------------------------------------------------
        # Decode-like.
        # ----------------------------------------------------
        (
            "decode_qwen_like",
            1,
            513,
            12,
            2,
            128,
            128,
        ),

        # ----------------------------------------------------
        # Prefill-like + prefix KV.
        # ----------------------------------------------------
        (
            "prefill_qwen_like",
            64,
            128,
            12,
            2,
            128,
            128,
        ),
    ]


def get_triton_dynamic_boundary_cases(
    device_name,
):
    device_type = _device_type_from_name(
        device_name
    )

    triton_backend = get_triton_backend(
        device_type
    )

    config = (
        triton_backend.self_attention_config(
            128,
            128,
            128,
        )
    )

    block_m = int(config["BLOCK_M"])
    block_n = int(config["BLOCK_N"])
    block_d = int(config["BLOCK_D"])
    block_v = int(config["BLOCK_V"])

    cases = []

    # --------------------------------------------------------
    # Query-length boundary: BLOCK_M - 1 / exact / +1.
    # Keep a prefix of BLOCK_N so kvlen >= qlen.
    # --------------------------------------------------------

    for delta, label in (
        (-1, "minus_one"),
        (0, "exact"),
        (1, "plus_one"),
    ):
        qlen = block_m + delta

        if qlen < 0:
            continue

        kvlen = (
            qlen
            + max(block_n, 1)
        )

        cases.append(
            (
                f"dynamic_block_m_{label}",
                qlen,
                kvlen,
                4,
                2,
                max(block_d, 1),
                max(block_v, 1),
            )
        )

    # --------------------------------------------------------
    # KV-length boundary: BLOCK_N - 1 / exact / +1.
    # qlen=1 keeps every generated case valid.
    # --------------------------------------------------------

    for delta, label in (
        (-1, "minus_one"),
        (0, "exact"),
        (1, "plus_one"),
    ):
        kvlen = block_n + delta

        if kvlen < 1:
            continue

        cases.append(
            (
                f"dynamic_block_n_{label}",
                1,
                kvlen,
                4,
                2,
                max(block_d, 1),
                max(block_v, 1),
            )
        )

    # --------------------------------------------------------
    # Q/K feature boundary: BLOCK_D - 1 / exact / +1.
    # --------------------------------------------------------

    for delta, label in (
        (-1, "minus_one"),
        (0, "exact"),
        (1, "plus_one"),
    ):
        qk_dim = block_d + delta

        if qk_dim <= 0:
            continue

        cases.append(
            (
                f"dynamic_block_d_{label}",
                min(7, max(block_m, 1)),
                max(
                    block_n + 1,
                    min(
                        7,
                        max(block_m, 1),
                    ),
                ),
                4,
                2,
                qk_dim,
                max(block_v, 1),
            )
        )

    # --------------------------------------------------------
    # Value feature boundary: BLOCK_V - 1 / exact / +1.
    # --------------------------------------------------------

    for delta, label in (
        (-1, "minus_one"),
        (0, "exact"),
        (1, "plus_one"),
    ):
        value_dim = block_v + delta

        if value_dim <= 0:
            continue

        cases.append(
            (
                f"dynamic_block_v_{label}",
                min(7, max(block_m, 1)),
                max(
                    block_n + 1,
                    min(
                        7,
                        max(block_m, 1),
                    ),
                ),
                4,
                2,
                max(block_d, 1),
                value_dim,
            )
        )

    # Remove exact duplicates while preserving order.
    result = []
    seen = set()

    for case in cases:
        key = case[1:]

        if key in seen:
            continue

        seen.add(key)
        result.append(case)

    return result


# ============================================================
# Performance suites
# ============================================================


def get_profile_cases(
    profile_suite,
):
    sweep = [
        (
            "decode_short",
            1,
            129,
            12,
            2,
            128,
            128,
        ),
        (
            "decode_medium",
            1,
            513,
            12,
            2,
            128,
            128,
        ),
        (
            "query16",
            16,
            528,
            12,
            2,
            128,
            128,
        ),
        (
            "query32",
            32,
            544,
            12,
            2,
            128,
            128,
        ),
        (
            "query64",
            64,
            576,
            12,
            2,
            128,
            128,
        ),
    ]

    llm = [
        (
            "decode_qwen_like",
            1,
            513,
            12,
            2,
            128,
            128,
        ),
        (
            "small_batch_qwen_like",
            32,
            544,
            12,
            2,
            128,
            128,
        ),
        (
            "prefill_qwen_like",
            64,
            128,
            12,
            2,
            128,
            128,
        ),
        (
            "decode_32h",
            1,
            513,
            32,
            4,
            128,
            128,
        ),
        (
            "prefill_32h",
            64,
            128,
            32,
            4,
            128,
            128,
        ),
    ]

    if profile_suite == "sweep":
        return [
            ("sweep", *case)
            for case in sweep
        ]

    if profile_suite == "llm":
        return [
            ("llm", *case)
            for case in llm
        ]

    return (
        [
            ("sweep", *case)
            for case in sweep
        ]
        + [
            ("llm", *case)
            for case in llm
        ]
    )


# ============================================================
# Profiler helpers
# ============================================================


def parse_attention_case(
    text,
):
    try:
        values = tuple(
            int(part.strip())
            for part in text.split(",")
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Attention case must contain six integers: "
            "qlen,kvlen,nh,nkvh,qk_dim,value_dim"
        ) from exc

    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "Attention case must contain exactly six integers: "
            "qlen,kvlen,nh,nkvh,qk_dim,value_dim"
        )

    return values


def _torch_profiler_synchronize(
    device_name,
):
    if device_name in (
        "nvidia",
        "amd",
    ):
        torch.cuda.synchronize()


def _begin_profiler_range(
    label,
    device_name,
):
    if device_name not in (
        "nvidia",
        "amd",
    ):
        return False

    if not torch.cuda.is_available():
        return False

    try:
        torch.cuda.nvtx.range_push(
            label
        )
        return True
    except Exception:
        return False


def _end_profiler_range(
    range_pushed,
):
    if not range_pushed:
        return

    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


def run_self_attention_profiler_case(
    *,
    case_shape,
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
    (
        qlen,
        kvlen,
        nh,
        nkvh,
        qk_dim,
        value_dim,
    ) = case_shape

    if kvlen < qlen:
        raise ValueError(
            "Profiler case requires kvlen >= qlen"
        )

    if nh <= 0 or nkvh <= 0:
        raise ValueError(
            "Profiler head counts must be positive"
        )

    if nh % nkvh != 0:
        raise ValueError(
            "Profiler query heads must be divisible by KV heads"
        )

    if qk_dim <= 0 or value_dim <= 0:
        raise ValueError(
            "Profiler feature dimensions must be positive"
        )

    print()
    print(
        "=== Profiler single case ==="
    )

    print(
        f"   target <{profiler_target}> "
        f"qlen={qlen} "
        f"kvlen={kvlen} "
        f"nh={nh} "
        f"nkvh={nkvh} "
        f"qk_dim={qk_dim} "
        f"value_dim={value_dim} "
        f"dtype <{dtype_name}> "
        f"device <{device_name}> "
        f"backend <{backend}>"
    )

    torch_q, llaisys_q = random_tensor(
        (qlen, nh, qk_dim),
        dtype_name,
        device_name,
    )

    torch_k, llaisys_k = random_tensor(
        (kvlen, nkvh, qk_dim),
        dtype_name,
        device_name,
    )

    torch_v, llaisys_v = random_tensor(
        (kvlen, nkvh, value_dim),
        dtype_name,
        device_name,
    )

    _, llaisys_out = zero_tensor(
        (qlen, nh, value_dim),
        dtype_name,
        device_name,
    )

    scale = (
        1.0
        / math.sqrt(qk_dim)
    )

    torch_attn_bias = (
        make_torch_attention_bias(
            qlen,
            kvlen,
        )
    )

    if profiler_target == "torch":
        if device_name == "metax":
            raise ValueError(
                "Torch profiler target is unavailable for MetaX "
                "because the current MetaX reference tensor is "
                "hosted on CPU."
            )

        target_fn = lambda: torch_self_attention(
            torch_q,
            torch_k,
            torch_v,
            scale,
            torch_attn_bias,
        )

        synchronize = lambda: (
            _torch_profiler_synchronize(
                device_name
            )
        )

        config_status = "reference"
        config = {}

        target_label = (
            f"LLAISYS_PROFILE:self_attention:"
            f"torch:{device_name}:"
            f"qlen={qlen}:"
            f"kvlen={kvlen}:"
            f"nh={nh}:"
            f"nkvh={nkvh}:"
            f"qk_dim={qk_dim}:"
            f"value_dim={value_dim}:"
            f"dtype={dtype_name}"
        )

        for _ in range(
            profiler_warmup
        ):
            target_fn()

        synchronize()

        range_pushed = (
            _begin_profiler_range(
                target_label,
                device_name,
            )
        )

        try:
            for _ in range(
                profiler_launches
            ):
                target_fn()

            synchronize()
        finally:
            _end_profiler_range(
                range_pushed
            )

        if profiler_check:
            torch_out = torch_self_attention(
                torch_q,
                torch_k,
                torch_v,
                scale,
                torch_attn_bias,
            )

            run_llaisys_self_attention(
                llaisys_out,
                llaisys_q,
                llaisys_k,
                llaisys_v,
                scale,
                backend,
            )

            assert check_equal(
                llaisys_out,
                torch_out,
                atol=atol,
                rtol=rtol,
            ), (
                "Self-Attention profiler correctness mismatch: "
                f"case={case_shape}, "
                f"dtype={dtype_name}, "
                f"device={device_name}, "
                f"backend={backend}"
            )
    else:
        config_status, config = (
            get_self_attention_config(
                llaisys_q,
                llaisys_v,
                backend,
            )
        )

        if show_config:
            print(
                "        "
                + get_self_attention_config_label(
                    llaisys_q,
                    llaisys_v,
                    backend,
                )
            )

        target_fn = lambda: (
            run_llaisys_self_attention(
                llaisys_out,
                llaisys_q,
                llaisys_k,
                llaisys_v,
                scale,
                backend,
            )
        )

        api = llaisys.RuntimeAPI(
            llaisys_out.device_type()
        )

        synchronize = (
            api.device_synchronize
        )

        config_tag = ",".join(
            f"{key}={value}"
            for key, value in config.items()
        )

        target_label = (
            f"LLAISYS_PROFILE:self_attention:"
            f"{backend}:{backend_variant}:"
            f"{device_name}:"
            f"qlen={qlen}:"
            f"kvlen={kvlen}:"
            f"nh={nh}:"
            f"nkvh={nkvh}:"
            f"qk_dim={qk_dim}:"
            f"value_dim={value_dim}:"
            f"dtype={dtype_name}:"
            f"{config_tag}"
        )

        def execute_target():
            for _ in range(
                profiler_warmup
            ):
                target_fn()

            synchronize()

            range_pushed = (
                _begin_profiler_range(
                    target_label,
                    device_name,
                )
            )

            try:
                for _ in range(
                    profiler_launches
                ):
                    target_fn()

                synchronize()
            finally:
                _end_profiler_range(
                    range_pushed
                )

        if backend == "triton":
            with execution_context(
                llaisys_out.device_type(),
                llaisys_out.device_id(),
            ):
                execute_target()
        else:
            execute_target()

        if profiler_check:
            torch_out = torch_self_attention(
                torch_q,
                torch_k,
                torch_v,
                scale,
                torch_attn_bias,
            )

            _torch_profiler_synchronize(
                device_name
            )

            assert check_equal(
                llaisys_out,
                torch_out,
                atol=atol,
                rtol=rtol,
            ), (
                "Self-Attention profiler correctness mismatch: "
                f"case={case_shape}, "
                f"dtype={dtype_name}, "
                f"device={device_name}, "
                f"backend={backend}"
            )

    print(
        f"Profiler target range: "
        f"{target_label}"
    )

    print(
        f"Profiler launches: "
        f"warmup={profiler_warmup}, "
        f"target={profiler_launches}"
    )

    if profiler_target == "torch":
        print(
            "Profiler note: PyTorch SDPA may dispatch to a fused "
            "SDPA backend on supported GPU inputs."
        )

    if profiler_check:
        print(
            "Profiler post-check: passed"
        )

    return {
        "target": profiler_target,
        "case_shape": list(case_shape),
        "dtype": dtype_name,
        "config_status": config_status,
        "config": config,
        "warmup": profiler_warmup,
        "launches": profiler_launches,
        "range": target_label,
    }


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cpu",
        choices=[
            "cpu",
            "nvidia",
            "metax",
            "amd",
        ],
        type=str,
    )

    parser.add_argument(
        "--backend",
        default="native",
        choices=[
            "native",
            "triton",
        ],
        type=str,
    )

    parser.add_argument(
        "--backend-variant",
        default="unspecified",
        type=str,
        help=(
            "Experiment variant label, for example baseline, "
            "tuned, autotuned, or vendor-specific."
        ),
    )

    parser.add_argument(
        "--backend-implementation",
        default=None,
        type=str,
        help=(
            "Optional implementation override. Native normally "
            "maps to cpu/cuda/maca/hip and Triton maps to triton."
        ),
    )

    execution_mode = (
        parser.add_mutually_exclusive_group()
    )

    execution_mode.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Run the paper-oriented Self-Attention "
            "microbenchmark suite."
        ),
    )

    execution_mode.add_argument(
        "--profiler-mode",
        action="store_true",
        help=(
            "Run one controlled Self-Attention workload for "
            "ncu/nsys/mcProfiler/rocprof."
        ),
    )

    parser.add_argument(
        "--case-shape",
        default=None,
        type=parse_attention_case,
        help=(
            "Profiler case: "
            "qlen,kvlen,nh,nkvh,qk_dim,value_dim"
        ),
    )

    parser.add_argument(
        "--case-dtype",
        default="f16",
        choices=[
            "f32",
            "f16",
            "bf16",
        ],
        type=str,
    )

    parser.add_argument(
        "--profiler-target",
        default="llaisys",
        choices=[
            "llaisys",
            "torch",
        ],
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
    )

    parser.add_argument(
        "--show-bandwidth",
        action="store_true",
    )

    parser.add_argument(
        "--show-throughput",
        action="store_true",
    )

    parser.add_argument(
        "--skip-correctness",
        action="store_true",
    )

    parser.add_argument(
        "--skip-dynamic-boundaries",
        action="store_true",
        help=(
            "Skip Triton effective-config boundary "
            "correctness cases."
        ),
    )

    parser.add_argument(
        "--profile-suite",
        default="all",
        choices=[
            "sweep",
            "llm",
            "all",
        ],
        type=str,
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

    args = parser.parse_args()

    # ========================================================
    # Validation
    # ========================================================

    if (
        args.backend == "triton"
        and args.device == "cpu"
    ):
        raise ValueError(
            "Triton Self-Attention requires a GPU device"
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup must be non-negative"
        )

    if args.repeat <= 0:
        raise ValueError(
            "--repeat must be greater than zero"
        )

    if args.rounds <= 0:
        raise ValueError(
            "--rounds must be greater than zero"
        )

    if args.profiler_warmup < 0:
        raise ValueError(
            "--profiler-warmup must be non-negative"
        )

    if args.profiler_launches <= 0:
        raise ValueError(
            "--profiler-launches must be greater than zero"
        )

    if (
        args.profiler_mode
        and args.case_shape is None
    ):
        raise ValueError(
            "--case-shape is required with --profiler-mode"
        )

    torch.manual_seed(
        args.seed
    )

    if (
        args.device in ("nvidia", "amd")
        and torch.cuda.is_available()
    ):
        torch.cuda.manual_seed_all(
            args.seed
        )

    backend_metadata = (
        collect_backend_metadata(
            args.backend,
            args.device,
            variant=args.backend_variant,
            implementation=(
                args.backend_implementation
            ),
        )
    )

    filename_config = (
        get_self_attention_output_filename_config(
            args.backend
        )
    )

    if args.output is not None:
        output_path = args.output
    elif (
        args.profile
        and not args.no_record
    ):
        output_path = (
            build_experiment_output_path(
                args.output_dir,
                op="self_attention",
                device_name=args.device,
                backend=backend_metadata,
                config=filename_config,
            )
        )
    else:
        output_path = None

    run_metadata = {
        "profile_suite": args.profile_suite,
        "benchmark_order": args.benchmark_order,
        "note": args.run_note,
        "reference": {
            "torch": (
                "torch.nn.functional."
                "scaled_dot_product_attention"
            ),
            "execution": "eager_api_call",
            "output_policy": "functional_return",
            "causal_alignment": "lower_right",
            "causal_bias": (
                "torch.nn.attention.bias."
                "causal_lower_right"
            ),
            "gqa": "enable_gqa",
            "dropout_p": 0.0,
        },
        "input_distribution": {
            "q": "uniform[0,1)",
            "k": "uniform[0,1)",
            "v": "uniform[0,1)",
        },
        "profiler_mode": args.profiler_mode,
        "profiler_case": {
            "shape": (
                list(args.case_shape)
                if args.case_shape is not None
                else None
            ),
            "dtype": args.case_dtype,
            "target": args.profiler_target,
            "warmup": args.profiler_warmup,
            "launches": args.profiler_launches,
        },
        "output": {
            "automatic": (
                args.output is None
            ),
            "directory": (
                args.output_dir
            ),
            "filename_config": (
                filename_config
            ),
        },
    }

    recorder = BenchmarkRecorder(
        output_path=output_path,
        repo_root=REPO_ROOT,
        run_id=args.run_id,
        run_metadata=run_metadata,
    )

    print(
        f"Testing Ops.self_attention "
        f"on {args.device} "
        f"with {args.backend} backend"
    )

    print(
        f"Backend identity: "
        f"name={backend_metadata['name']}, "
        f"implementation="
        f"{backend_metadata['implementation']}, "
        f"variant={backend_metadata['variant']}"
    )

    print(
        f"Random seed: {args.seed}"
    )

    print(
        f"Benchmark protocol: "
        f"warmup={args.warmup}, "
        f"repeat={args.repeat}, "
        f"rounds={args.rounds}, "
        f"order={args.benchmark_order}"
    )

    print(
        f"Using llaisys from: "
        f"{llaisys.__file__}"
    )

    if output_path is not None:
        print(
            f"Recording JSONL: "
            f"{output_path}"
        )

        print(
            f"Run ID: "
            f"{recorder.run_id}"
        )

    # ========================================================
    # Profiler mode
    # ========================================================

    if args.profiler_mode:
        tolerance = {
            "f32": (1e-5, 1e-5),
            "f16": (1e-3, 1e-3),
            "bf16": (1e-2, 1e-2),
        }[args.case_dtype]

        run_self_attention_profiler_case(
            case_shape=args.case_shape,
            dtype_name=args.case_dtype,
            atol=tolerance[0],
            rtol=tolerance[1],
            device_name=args.device,
            backend=args.backend,
            backend_variant=(
                args.backend_variant
            ),
            profiler_target=(
                args.profiler_target
            ),
            profiler_warmup=(
                args.profiler_warmup
            ),
            profiler_launches=(
                args.profiler_launches
            ),
            profiler_check=(
                args.profiler_check
            ),
            show_config=(
                args.show_config
            ),
        )

        print()
        print(
            "\033[92mProfiler run completed!\033[0m"
        )

        raise SystemExit(0)

    # ========================================================
    # Fixed correctness
    # ========================================================

    if not args.skip_correctness:
        print()
        print(
            "=== Correctness: fixed semantic / "
            "boundary / workload cases ==="
        )

        for (
            case_name,
            qlen,
            kvlen,
            nh,
            nkvh,
            qk_dim,
            value_dim,
        ) in get_fixed_correctness_cases():
            for (
                dtype_name,
                atol,
                rtol,
            ) in TEST_DTYPE_PREC:
                test_op_self_attention(
                    qlen,
                    kvlen,
                    nh,
                    nkvh,
                    qk_dim,
                    value_dim,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    device_name=args.device,
                    backend=args.backend,
                    profile=False,
                    case_name=case_name,
                )

    # ========================================================
    # Dynamic Triton boundaries
    # ========================================================

    if (
        args.backend == "triton"
        and not args.skip_correctness
        and not args.skip_dynamic_boundaries
    ):
        print()
        print(
            "=== Correctness: effective Triton "
            "configuration boundaries ==="
        )

        dynamic_cases = (
            get_triton_dynamic_boundary_cases(
                args.device
            )
        )

        for (
            case_name,
            qlen,
            kvlen,
            nh,
            nkvh,
            qk_dim,
            value_dim,
        ) in dynamic_cases:
            for (
                dtype_name,
                atol,
                rtol,
            ) in TEST_DTYPE_PREC:
                test_op_self_attention(
                    qlen,
                    kvlen,
                    nh,
                    nkvh,
                    qk_dim,
                    value_dim,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    device_name=args.device,
                    backend=args.backend,
                    profile=False,
                    case_name=case_name,
                )

    # ========================================================
    # Performance
    # ========================================================

    if args.profile:
        print()
        print(
            "=== Performance ==="
        )

        for (
            suite,
            case_name,
            qlen,
            kvlen,
            nh,
            nkvh,
            qk_dim,
            value_dim,
        ) in get_profile_cases(
            args.profile_suite
        ):
            for (
                dtype_name,
                atol,
                rtol,
            ) in TEST_DTYPE_PREC:
                test_op_self_attention(
                    qlen,
                    kvlen,
                    nh,
                    nkvh,
                    qk_dim,
                    value_dim,
                    dtype_name=dtype_name,
                    atol=atol,
                    rtol=rtol,
                    device_name=args.device,
                    backend=args.backend,
                    profile=True,
                    backend_variant=(
                        args.backend_variant
                    ),
                    backend_implementation=(
                        args.backend_implementation
                    ),
                    suite=suite,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    rounds=args.rounds,
                    benchmark_order=(
                        args.benchmark_order
                    ),
                    show_config=(
                        args.show_config
                    ),
                    show_bandwidth=(
                        args.show_bandwidth
                    ),
                    show_throughput=(
                        args.show_throughput
                    ),
                    recorder=recorder,
                    device_metadata={},
                    case_name=case_name,
                )

    print()
    print(
        "\033[92mTest passed!\033[0m"
    )
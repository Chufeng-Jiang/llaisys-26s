import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
from test_utils import arrange_tensor, benchmark_llaisys, check_equal, random_tensor

import llaisys


def torch_rope(y: torch.Tensor, x: torch.Tensor, pos_ids: torch.Tensor, theta: float):
    assert y.dim() == 3

    seq_len, n_heads, head_dim = y.shape

    assert head_dim % 2 == 0, "Head dimension must be even for RoPE."

    # ========================================================
    # Split dimension into paired halves
    # ========================================================

    x_a = x[..., : head_dim // 2]
    x_b = x[..., head_dim // 2 :]

    # ========================================================
    # Position IDs in FP32
    # ========================================================

    positions = pos_ids.to(torch.float32).unsqueeze(1)

    # ========================================================
    # Match LLAISYS RoPE angle semantics
    #
    # exponent    = 2 * i / d
    # denominator = theta ** exponent
    # angle       = position / denominator
    # ========================================================

    i = torch.arange(0, head_dim // 2, dtype=torch.float32, device=y.device)

    exponent = 2.0 * i / head_dim

    denominator = theta**exponent

    freqs = positions / denominator

    sin = freqs.sin().unsqueeze(1)

    cos = freqs.cos().unsqueeze(1)

    # ========================================================
    # Rotation
    # ========================================================

    y[..., : head_dim // 2] = x_a * cos - x_b * sin

    y[..., head_dim // 2 :] = x_b * cos + x_a * sin


def test_op_rope(shape, start_end, dtype_name="f32", atol=1e-5, rtol=1e-5, device_name="cpu", profile=False):
    print(f"   shape {shape} range {start_end} dtype <{dtype_name}>")

    x, x_ = random_tensor(shape, dtype_name, device_name)

    pos_ids, pos_ids_ = arrange_tensor(start_end[0], start_end[1], device_name)

    theta = 10000.0

    y, y_ = random_tensor(shape, dtype_name, device_name)

    # ========================================================
    # Correctness reference
    # ========================================================

    torch_rope(y, x, pos_ids, theta)

    llaisys.Ops.rope(y_, x_, pos_ids_, theta)

    assert check_equal(y_, y, atol=atol, rtol=rtol)

    # ========================================================
    # LLAISYS profiling
    # ========================================================

    if profile:
        benchmark_llaisys(
            lambda: llaisys.Ops.rope(y_, x_, pos_ids_, theta),
            device_name,
            label=(f"RoPE shape={shape} range={start_end} dtype={dtype_name}"),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "metax"], type=str)

    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()

    # ========================================================
    # Diagnostic FP32 cases
    # ========================================================
    #
    # Case 1:
    #     Small baseline already known to pass.
    #
    # Case 2:
    #     Large tensor / large dimension,
    #     but positions begin at zero.
    #
    # Case 3:
    #     Large positions,
    #     but much smaller head dimension.
    #
    # Case 4:
    #     Original failing configuration.
    #
    # These cases help separate:
    #
    #     position-dependent trig error
    #     vs
    #     head-dimension / frequency error
    #
    # ========================================================

    test_shapes = [
        # Existing tiny direct sanity check
        ((2, 1, 4), (0, 2)),
        # Decode-like: small head dimension, multi-head
        ((1, 12, 128), (512, 513)),
        # Prefill-like: small head dimension, multi-head
        ((512, 12, 128), (512, 1024)),
        # Intermediate head dimension
        ((512, 12, 256), (512, 1024)),
        # Stress case: already known cache regression
        ((512, 4, 4096), (512, 1024)),
    ]

    # Temporarily test FP32 only while diagnosing
    # the large-shape numerical mismatch.
    test_dtype_prec = [
        ("f32", 2e-4, 1e-4),  # ("f32", 1e-4, 1e-4),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]

    print(f"Testing Ops.rope on {args.device}")

    for shape, start_end in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            test_op_rope(shape, start_end, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")

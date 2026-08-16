import argparse

import torch
import triton
import triton.language as tl

from triton.language.extra import libdevice


@triton.jit
def rope_math_debug_kernel(
    exponent_out,
    denominator_out,
    angle_out,
    sine_out,
    cosine_out,
    position,
    theta,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair = tl.arange(0, BLOCK_SIZE)
    half_dim: tl.constexpr = HEAD_DIM // 2
    mask = pair < half_dim

    pair_f32 = pair.to(tl.float32)
    head_dim_f32 = tl.full((BLOCK_SIZE,), HEAD_DIM, tl.float32)

    exponent = libdevice.div_rn(
        pair_f32 * 2.0,
        head_dim_f32,
    )

    theta_f32 = tl.full((BLOCK_SIZE,), theta, tl.float32)

    denominator = libdevice.pow(
        theta_f32,
        exponent,
    )

    position_f32 = tl.full(
        (BLOCK_SIZE,),
        position,
        tl.float32,
    )

    angle = libdevice.div_rn(
        position_f32,
        denominator,
    )

    sine = libdevice.sin(angle)
    cosine = libdevice.cos(angle)

    tl.store(exponent_out + pair, exponent, mask=mask)
    tl.store(denominator_out + pair, denominator, mask=mask)
    tl.store(angle_out + pair, angle, mask=mask)
    tl.store(sine_out + pair, sine, mask=mask)
    tl.store(cosine_out + pair, cosine, mask=mask)


def ordered_int32(x: torch.Tensor) -> torch.Tensor:
    bits = x.contiguous().view(torch.int32)
    sign = bits >> 31
    return bits ^ (sign & 0x7FFFFFFF)


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_i = ordered_int32(a).to(torch.int64)
    b_i = ordered_int32(b).to(torch.int64)
    return (a_i - b_i).abs()


def summarize_stage(name, triton_value, torch_value):
    abs_error = (triton_value - torch_value).abs()
    ulp = ulp_distance(triton_value, torch_value)

    max_abs, max_abs_idx = abs_error.max(dim=0)
    max_ulp, max_ulp_idx = ulp.max(dim=0)

    print()
    print(f"=== {name} ===")
    print(f"max_abs_error = {max_abs.item():.10e} at pair {max_abs_idx.item()}")
    print(f"max_ulp       = {max_ulp.item()} at pair {max_ulp_idx.item()}")


def print_pair(
    pair,
    exponent_triton,
    denominator_triton,
    angle_triton,
    sine_triton,
    cosine_triton,
    exponent_torch,
    denominator_torch,
    angle_torch,
    sine_torch,
    cosine_torch,
):
    print()
    print("=" * 72)
    print(f"PAIR {pair}")
    print("=" * 72)

    rows = [
        ("exponent", exponent_triton[pair], exponent_torch[pair]),
        ("denominator", denominator_triton[pair], denominator_torch[pair]),
        ("angle", angle_triton[pair], angle_torch[pair]),
        ("sine", sine_triton[pair], sine_torch[pair]),
        ("cosine", cosine_triton[pair], cosine_torch[pair]),
    ]

    for name, tv, pv in rows:
        tv1 = tv.reshape(1)
        pv1 = pv.reshape(1)
        abs_error = (tv1 - pv1).abs().item()
        ulp = ulp_distance(tv1, pv1).item()

        print(
            f"{name:12s} "
            f"Triton={tv.item(): .10e} "
            f"Torch={pv.item(): .10e} "
            f"abs={abs_error:.10e} "
            f"ulp={ulp}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--position", type=int, default=8193)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--pair", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.head_dim <= 0 or args.head_dim % 2 != 0:
        raise ValueError("--head-dim must be positive and even")

    half_dim = args.head_dim // 2
    block_size = triton.next_power_of_2(half_dim)

    if args.pair < 0 or args.pair >= half_dim:
        raise ValueError(
            f"--pair must be within [0, {half_dim - 1}]"
        )

    device = "cuda"

    exponent_triton = torch.empty(
        half_dim,
        dtype=torch.float32,
        device=device,
    )
    denominator_triton = torch.empty_like(exponent_triton)
    angle_triton = torch.empty_like(exponent_triton)
    sine_triton = torch.empty_like(exponent_triton)
    cosine_triton = torch.empty_like(exponent_triton)

    rope_math_debug_kernel[(1,)](
        exponent_triton,
        denominator_triton,
        angle_triton,
        sine_triton,
        cosine_triton,
        float(args.position),
        float(args.theta),
        HEAD_DIM=args.head_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    torch.cuda.synchronize()

    # Match the current test/ops/rope.py Torch reference as closely as possible.
    pair_torch = torch.arange(
        half_dim,
        dtype=torch.float32,
        device=device,
    )

    exponent_torch = (
        2.0 * pair_torch / float(args.head_dim)
    )

    # Match test/ops/rope.py exactly:
    #
    #     denominator = theta ** exponent
    #
    # where theta is a Python float and exponent is a CUDA FloatTensor.
    denominator_torch = args.theta ** exponent_torch

    position_torch = torch.tensor(
        float(args.position),
        dtype=torch.float32,
        device=device,
    )

    angle_torch = position_torch / denominator_torch
    sine_torch = torch.sin(angle_torch)
    cosine_torch = torch.cos(angle_torch)

    print()
    print("RoPE Triton-vs-Torch intermediate diagnostic")
    print(f"head_dim  = {args.head_dim}")
    print(f"half_dim  = {half_dim}")
    print(f"position  = {args.position}")
    print(f"theta     = {args.theta}")
    print(f"block     = {block_size}")
    print(f"pair      = {args.pair}")

    summarize_stage(
        "exponent",
        exponent_triton,
        exponent_torch,
    )
    summarize_stage(
        "denominator",
        denominator_triton,
        denominator_torch,
    )
    summarize_stage(
        "angle",
        angle_triton,
        angle_torch,
    )
    summarize_stage(
        "sine",
        sine_triton,
        sine_torch,
    )
    summarize_stage(
        "cosine",
        cosine_triton,
        cosine_torch,
    )

    print_pair(
        args.pair,
        exponent_triton,
        denominator_triton,
        angle_triton,
        sine_triton,
        cosine_triton,
        exponent_torch,
        denominator_torch,
        angle_torch,
        sine_torch,
        cosine_torch,
    )

    first_divergence = None

    stages = [
        ("exponent", exponent_triton, exponent_torch),
        ("denominator", denominator_triton, denominator_torch),
        ("angle", angle_triton, angle_torch),
        ("sine", sine_triton, sine_torch),
        ("cosine", cosine_triton, cosine_torch),
    ]

    for name, triton_value, torch_value in stages:
        if not torch.equal(triton_value, torch_value):
            first_divergence = name
            break

    print()
    print("=" * 72)

    if first_divergence is None:
        print("First exact divergence: none")
    else:
        print(f"First exact divergence: {first_divergence}")

    print("=" * 72)


if __name__ == "__main__":
    main()
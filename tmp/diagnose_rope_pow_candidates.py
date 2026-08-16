import argparse

import torch
import triton
import triton.language as tl

from triton.language.extra import libdevice


@triton.jit
def pow_candidate_kernel(
    pow32_out,
    pow64_round_f32_out,
    angle32_out,
    angle64pow_round_f32_out,
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

    # Match the Torch reference's FP32 exponent construction.
    exponent_f32 = libdevice.div_rn(
        pair_f32 * 2.0,
        head_dim_f32,
    )

    theta_f32 = tl.full((BLOCK_SIZE,), theta, tl.float32)

    # Current Triton path.
    denominator_pow32 = libdevice.pow(
        theta_f32,
        exponent_f32,
    )

    # Candidate:
    # evaluate pow in FP64 using the SAME already-rounded FP32 exponent,
    # then round the denominator once back to FP32.
    theta_f64 = theta_f32.to(tl.float64)
    exponent_f64 = exponent_f32.to(tl.float64)

    denominator_pow64_round_f32 = libdevice.pow(
        theta_f64,
        exponent_f64,
    ).to(tl.float32)

    position_f32 = tl.full(
        (BLOCK_SIZE,),
        position,
        tl.float32,
    )

    angle32 = libdevice.div_rn(
        position_f32,
        denominator_pow32,
    )

    angle64pow_round_f32 = libdevice.div_rn(
        position_f32,
        denominator_pow64_round_f32,
    )

    tl.store(
        pow32_out + pair,
        denominator_pow32,
        mask=mask,
    )
    tl.store(
        pow64_round_f32_out + pair,
        denominator_pow64_round_f32,
        mask=mask,
    )
    tl.store(
        angle32_out + pair,
        angle32,
        mask=mask,
    )
    tl.store(
        angle64pow_round_f32_out + pair,
        angle64pow_round_f32,
        mask=mask,
    )


def ordered_int32(x):
    bits = x.contiguous().view(torch.int32)
    sign = bits >> 31
    return bits ^ (sign & 0x7FFFFFFF)


def ulp_distance(a, b):
    return (
        ordered_int32(a).to(torch.int64)
        - ordered_int32(b).to(torch.int64)
    ).abs()


def summarize(name, candidate, reference):
    diff = candidate != reference
    mismatch_count = int(diff.sum().item())
    abs_error = (candidate - reference).abs()
    ulp = ulp_distance(candidate, reference)

    print()
    print(f"=== {name} ===")
    print(f"mismatch_count = {mismatch_count} / {candidate.numel()}")
    print(f"max_abs_error  = {abs_error.max().item():.10e}")
    print(f"max_ulp        = {ulp.max().item()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--position", type=int, default=8193)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--pair", type=int, default=1)
    args = parser.parse_args()

    if args.head_dim <= 0 or args.head_dim % 2 != 0:
        raise ValueError("--head-dim must be positive and even")

    half_dim = args.head_dim // 2
    block_size = triton.next_power_of_2(half_dim)

    pow32 = torch.empty(half_dim, device="cuda", dtype=torch.float32)
    pow64_round = torch.empty_like(pow32)
    angle32 = torch.empty_like(pow32)
    angle64pow_round = torch.empty_like(pow32)

    pow_candidate_kernel[(1,)](
        pow32,
        pow64_round,
        angle32,
        angle64pow_round,
        float(args.position),
        float(args.theta),
        HEAD_DIM=args.head_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    torch.cuda.synchronize()

    pair_torch = torch.arange(
        half_dim,
        dtype=torch.float32,
        device="cuda",
    )

    exponent_torch = (
        2.0 * pair_torch / float(args.head_dim)
    )

    denominator_torch = args.theta ** exponent_torch

    position_torch = torch.tensor(
        float(args.position),
        dtype=torch.float32,
        device="cuda",
    )

    angle_torch = position_torch / denominator_torch

    print()
    print("RoPE denominator candidate diagnostic")
    print(f"head_dim = {args.head_dim}")
    print(f"position = {args.position}")
    print(f"theta    = {args.theta}")

    summarize(
        "current libdevice.pow FP32 denominator",
        pow32,
        denominator_torch,
    )

    summarize(
        "FP64 pow -> FP32 rounded denominator",
        pow64_round,
        denominator_torch,
    )

    summarize(
        "current FP32 angle",
        angle32,
        angle_torch,
    )

    summarize(
        "FP64-pow-rounded-to-FP32 angle",
        angle64pow_round,
        angle_torch,
    )

    p = args.pair

    print()
    print("=" * 72)
    print(f"PAIR {p}")
    print("=" * 72)

    for name, value, reference in [
        ("pow32", pow32[p], denominator_torch[p]),
        ("pow64->f32", pow64_round[p], denominator_torch[p]),
        ("angle32", angle32[p], angle_torch[p]),
        ("angle64pow->f32", angle64pow_round[p], angle_torch[p]),
    ]:
        value1 = value.reshape(1)
        reference1 = reference.reshape(1)

        print(
            f"{name:18s} "
            f"Triton={value.item(): .10e} "
            f"Torch={reference.item(): .10e} "
            f"abs={(value-reference).abs().item():.10e} "
            f"ulp={ulp_distance(value1, reference1).item()}"
        )


if __name__ == "__main__":
    main()
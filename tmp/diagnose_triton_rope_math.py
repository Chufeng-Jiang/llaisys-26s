import argparse

import torch
import triton
import triton.language as tl

from triton.language.extra import libdevice


# ============================================================
# Diagnostic kernel 1
#
# Reproduce the exact mathematical path used by the current
# Triton RoPE kernel:
#
#     exponent
#         ↓
#     pow(theta, exponent)
#         ↓
#     position / denominator
#         ↓
#     sin(angle), cos(angle)
#
# Store every intermediate so that we can compare each stage
# independently against PyTorch FP32.
# ============================================================


@triton.jit
def rope_math_diagnostic_kernel(
    pos_ids,
    denominator_out,
    angle_out,
    sine_out,
    cosine_out,
    theta,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # ========================================================
    # Grid
    #
    # axis 0:
    #     token
    #
    # axis 1:
    #     pair tile
    # ========================================================

    token = tl.program_id(axis=0)

    pair_block = tl.program_id(axis=1)

    # ========================================================
    # Pair index
    # ========================================================

    pair = pair_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = pair < HALF_DIM

    # ========================================================
    # Position ID
    # ========================================================

    position = tl.load(pos_ids + token).to(tl.float32)

    # ========================================================
    # Exponent
    #
    # Match:
    #
    #     2 * pair / head_dim
    # ========================================================

    pair_f32 = pair.to(tl.float32)

    exponent = 2.0 * pair_f32 / float(HEAD_DIM)

    # ========================================================
    # Denominator
    #
    #     theta ** exponent
    # ========================================================

    theta_value = tl.full((BLOCK_SIZE,), theta, tl.float32)

    denominator = libdevice.pow(theta_value, exponent)

    # ========================================================
    # Angle
    #
    #     position / denominator
    # ========================================================

    angle = position / denominator

    # ========================================================
    # Trig
    # ========================================================

    sine = libdevice.sin(angle)

    cosine = libdevice.cos(angle)

    # ========================================================
    # Flatten [token, pair]
    # ========================================================

    offset = token * HALF_DIM + pair

    # ========================================================
    # Store intermediates
    # ========================================================

    tl.store(denominator_out + offset, denominator, mask=mask)

    tl.store(angle_out + offset, angle, mask=mask)

    tl.store(sine_out + offset, sine, mask=mask)

    tl.store(cosine_out + offset, cosine, mask=mask)


# ============================================================
# Diagnostic kernel 2
#
# This kernel receives the PyTorch-computed FP32 angle directly.
#
# Therefore:
#
#     PyTorch angle
#          ↓
#     Triton libdevice.sin/cos
#
# If this differs from:
#
#     PyTorch angle
#          ↓
#     torch.sin/cos
#
# then the difference is specifically in the trig stage,
# independent of pow/division.
# ============================================================


@triton.jit
def trig_from_reference_angle_kernel(angle, sine_out, cosine_out, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < numel

    value = tl.load(angle + offsets, mask=mask, other=0.0).to(tl.float32)

    sine = libdevice.sin(value)

    cosine = libdevice.cos(value)

    tl.store(sine_out + offsets, sine, mask=mask)

    tl.store(cosine_out + offsets, cosine, mask=mask)


# ============================================================
# Report one comparison
# ============================================================


def report_difference(name, actual, reference, positions=None):
    actual_f32 = actual.float()
    reference_f32 = reference.float()

    abs_error = torch.abs(actual_f32 - reference_f32)

    denominator = torch.clamp(torch.abs(reference_f32), min=1e-12)

    rel_error = abs_error / denominator

    mismatch = actual_f32 != reference_f32

    mismatch_count = int(mismatch.sum().item())

    total_count = actual_f32.numel()

    max_abs_error = float(abs_error.max().item())

    max_rel_error = float(rel_error.max().item())

    flat_index = int(abs_error.reshape(-1).argmax().item())

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(f"shape                  = {tuple(actual.shape)}")

    print(f"exact_mismatch_count   = {mismatch_count}")

    print(f"total_count            = {total_count}")

    print(f"exact_mismatch_ratio   = {100.0 * mismatch_count / total_count:.10f}%")

    print(f"max_abs_error          = {max_abs_error:.10e}")

    print(f"max_rel_error          = {max_rel_error:.10e}")

    # ========================================================
    # 1D tensor
    # ========================================================

    if actual.ndim == 1:
        pair = flat_index

        print()
        print("Worst element")

        print("-" * 40)

        print(f"pair                   = {pair}")

        print(f"actual                 = {float(actual_f32[pair].item()):.10e}")

        print(f"reference              = {float(reference_f32[pair].item()):.10e}")

        print(f"abs_error              = {float(abs_error[pair].item()):.10e}")

        return

    # ========================================================
    # 2D [token, pair]
    # ========================================================

    if actual.ndim == 2:
        half_dim = actual.shape[1]

        token = flat_index // half_dim

        pair = flat_index % half_dim

        print()
        print("Worst element")

        print("-" * 40)

        print(f"token                  = {token}")

        if positions is not None:
            print(f"position               = {int(positions[token].item())}")

        print(f"pair                   = {pair}")

        print(f"actual                 = {float(actual_f32[token, pair].item()):.10e}")

        print(f"reference              = {float(reference_f32[token, pair].item()):.10e}")

        print(f"abs_error              = {float(abs_error[token, pair].item()):.10e}")

        print(f"rel_error              = {float(rel_error[token, pair].item()):.10e}")


# ============================================================
# Print the exact element that failed the current formal test
# ============================================================


def report_target(
    positions,
    target_position,
    target_pair,
    denominator_triton,
    denominator_ref,
    angle_triton,
    angle_ref,
    sine_triton,
    sine_ref,
    cosine_triton,
    cosine_ref,
    sine_same_angle,
    cosine_same_angle,
):
    matches = (positions == target_position).nonzero(as_tuple=False)

    if matches.numel() == 0:
        print()
        print(f"Target position {target_position} is outside the current range.")

        return

    token = int(matches[0].item())

    pair = target_pair

    print()
    print("=" * 72)

    print("CURRENT FORMAL-TEST FAILURE LOCATION")

    print("=" * 72)

    print(f"token                  = {token}")

    print(f"position               = {target_position}")

    print(f"pair                   = {pair}")

    print()

    print("denominator")

    print(f"  Triton               = {denominator_triton[token, pair].item():.10e}")

    print(f"  PyTorch              = {denominator_ref[pair].item():.10e}")

    print(f"  abs error            = {abs(denominator_triton[token, pair].item() - denominator_ref[pair].item()):.10e}")

    print()

    print("angle")

    print(f"  Triton               = {angle_triton[token, pair].item():.10e}")

    print(f"  PyTorch              = {angle_ref[token, pair].item():.10e}")

    print(f"  abs error            = {abs(angle_triton[token, pair].item() - angle_ref[token, pair].item()):.10e}")

    print()

    print("sin")

    print(f"  Triton full path     = {sine_triton[token, pair].item():.10e}")

    print(f"  Triton same angle    = {sine_same_angle[token, pair].item():.10e}")

    print(f"  PyTorch              = {sine_ref[token, pair].item():.10e}")

    print()

    print("cos")

    print(f"  Triton full path     = {cosine_triton[token, pair].item():.10e}")

    print(f"  Triton same angle    = {cosine_same_angle[token, pair].item():.10e}")

    print(f"  PyTorch              = {cosine_ref[token, pair].item():.10e}")


# ============================================================
# Main diagnostic
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=int, default=512)

    parser.add_argument("--end", type=int, default=1024)

    parser.add_argument("--head-dim", type=int, default=4096)

    parser.add_argument("--theta", type=float, default=10000.0)

    parser.add_argument("--block-size", type=int, default=128)

    parser.add_argument("--target-position", type=int, default=858)

    parser.add_argument("--target-pair", type=int, default=4)

    args = parser.parse_args()

    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")

    if args.head_dim <= 0:
        raise ValueError("--head-dim must be positive")

    if args.head_dim % 2 != 0:
        raise ValueError("--head-dim must be even")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")

    head_dim = args.head_dim

    half_dim = head_dim // 2

    sequence_length = args.end - args.start

    theta = float(args.theta)

    block_size = args.block_size

    # ========================================================
    # Positions
    # ========================================================

    positions = torch.arange(args.start, args.end, dtype=torch.int64, device=device)

    # ========================================================
    # Triton outputs
    # ========================================================

    shape = (sequence_length, half_dim)

    denominator_triton = torch.empty(shape, dtype=torch.float32, device=device)

    angle_triton = torch.empty(shape, dtype=torch.float32, device=device)

    sine_triton = torch.empty(shape, dtype=torch.float32, device=device)

    cosine_triton = torch.empty(shape, dtype=torch.float32, device=device)

    # ========================================================
    # Launch current Triton mathematical path
    # ========================================================

    grid = (sequence_length, triton.cdiv(half_dim, block_size))

    rope_math_diagnostic_kernel[grid](
        positions,
        denominator_triton,
        angle_triton,
        sine_triton,
        cosine_triton,
        theta,
        HEAD_DIM=head_dim,
        HALF_DIM=half_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    torch.cuda.synchronize()

    # ========================================================
    # PyTorch FP32 reference
    #
    # This matches the reference currently used by:
    #
    #     test/ops/rope.py
    # ========================================================

    pair = torch.arange(0, half_dim, dtype=torch.float32, device=device)

    exponent_ref = 2.0 * pair / head_dim

    denominator_ref = theta**exponent_ref

    angle_ref = positions.to(torch.float32).unsqueeze(1) / denominator_ref.unsqueeze(0)

    sine_ref = torch.sin(angle_ref)

    cosine_ref = torch.cos(angle_ref)

    # ========================================================
    # Isolate trig only
    #
    # Feed the exact PyTorch-computed FP32 angle into Triton.
    # ========================================================

    sine_same_angle = torch.empty_like(angle_ref)

    cosine_same_angle = torch.empty_like(angle_ref)

    numel = angle_ref.numel()

    trig_block_size = 256

    trig_grid = (triton.cdiv(numel, trig_block_size),)

    trig_from_reference_angle_kernel[trig_grid](
        angle_ref, sine_same_angle, cosine_same_angle, numel, BLOCK_SIZE=trig_block_size, num_warps=4
    )

    torch.cuda.synchronize()

    # ========================================================
    # Stage 1
    #
    # denominator is position-independent.
    #
    # Compare one row against PyTorch.
    # ========================================================

    report_difference("STAGE 1: denominator = theta ** exponent", denominator_triton[0], denominator_ref)

    # ========================================================
    # Stage 2
    # ========================================================

    report_difference("STAGE 2: angle = position / denominator", angle_triton, angle_ref, positions)

    # ========================================================
    # Stage 3
    #
    # Full Triton path.
    # ========================================================

    report_difference("STAGE 3A: sin - full Triton path", sine_triton, sine_ref, positions)

    report_difference("STAGE 3B: cos - full Triton path", cosine_triton, cosine_ref, positions)

    # ========================================================
    # Stage 4
    #
    # Same input angle.
    #
    # This removes pow and division from the comparison.
    # ========================================================

    report_difference("STAGE 4A: sin - SAME PyTorch angle", sine_same_angle, sine_ref, positions)

    report_difference("STAGE 4B: cos - SAME PyTorch angle", cosine_same_angle, cosine_ref, positions)

    # ========================================================
    # Exact element observed in the failed formal test
    #
    # Formal failure:
    #
    #     token = 346
    #     start = 512
    #
    # therefore:
    #
    #     position = 512 + 346 = 858
    #
    # flat dimension index = 4
    # therefore:
    #
    #     pair = 4
    # ========================================================

    report_target(
        positions,
        args.target_position,
        args.target_pair,
        denominator_triton,
        denominator_ref,
        angle_triton,
        angle_ref,
        sine_triton,
        sine_ref,
        cosine_triton,
        cosine_ref,
        sine_same_angle,
        cosine_same_angle,
    )

    print()
    print("=" * 72)

    print("INTERPRETATION")

    print("=" * 72)

    print(
        """
Read the results in this order:

1. STAGE 1 differs
   -> pow / exponent path is already different.

2. STAGE 1 matches, but STAGE 2 differs
   -> FP32 division / type promotion is different.

3. STAGE 2 matches, STAGE 3 differs,
   and STAGE 4 also differs
   -> the difference is specifically in sin/cos.

4. STAGE 3 differs, but STAGE 4 matches
   -> trig itself is fine; the upstream angle difference
      is being amplified by sin/cos.

5. Everything above is exact or extremely close
   -> next inspect the final low*cos-high*sin /
      high*cos+low*sin arithmetic and possible FMA behavior.

Do NOT change the RoPE tolerance yet.
"""
    )


if __name__ == "__main__":
    main()

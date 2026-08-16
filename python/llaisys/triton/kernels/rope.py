import triton
import triton.language as tl

from triton.language.extra import libdevice


@triton.jit
def rope_kernel(
    out,
    x,
    pos_ids,
    theta,
    head_count,
    stride_xs,
    stride_xh,
    stride_xd,
    stride_os,
    stride_oh,
    stride_od,
    stride_pos,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # ============================================================
    # Program decomposition
    #
    # One Triton program handles:
    #
    #     one token
    #     one RoPE pair tile
    #     all attention heads
    #
    # Grid:
    #
    #     axis 0 = token
    #     axis 1 = pair tile
    #
    # The expensive frequency / pow / sin / cos path therefore runs
    # once per (token, pair), instead of once per (token, head, pair).
    # ============================================================

    token = tl.program_id(0)
    pair_block = tl.program_id(1)

    HALF_DIM: tl.constexpr = HEAD_DIM // 2

    pair = pair_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair < HALF_DIM

    # ============================================================
    # Accurate RoPE frequency path
    #
    # Keep exponent construction in FP32.
    #
    # Evaluate only pow in FP64, then round the denominator back to
    # FP32. This is the numerical repair already validated by the
    # full RoPE correctness suite.
    # ============================================================

    position = tl.load(
        pos_ids + token * stride_pos
    ).to(tl.float32)

    pair_f32 = pair.to(tl.float32)

    head_dim_f32 = tl.full(
        (BLOCK_SIZE,),
        HEAD_DIM,
        tl.float32,
    )

    exponent_f32 = libdevice.div_rn(
        pair_f32 * 2.0,
        head_dim_f32,
    )

    theta_f32 = tl.full(
        (BLOCK_SIZE,),
        theta,
        tl.float32,
    )

    denominator = libdevice.pow(
        theta_f32.to(tl.float64),
        exponent_f32.to(tl.float64),
    ).to(tl.float32)

    position_f32 = position + tl.zeros(
        (BLOCK_SIZE,),
        tl.float32,
    )

    angle = libdevice.div_rn(
        position_f32,
        denominator,
    )

    sine = libdevice.sin(angle)
    cosine = libdevice.cos(angle)

    # ============================================================
    # Reuse across heads
    #
    # sine/cosine stay live while the program walks all heads.
    # Each iteration touches disjoint head storage, so exact
    # out == x aliasing remains safe: both values of a RoPE pair are
    # loaded before either result is stored.
    # ============================================================

    for head in tl.range(0, head_count):
        x_base = (
            token * stride_xs
            + head * stride_xh
        )

        low_x_ptrs = (
            x
            + x_base
            + pair * stride_xd
        )

        high_x_ptrs = (
            x
            + x_base
            + (pair + HALF_DIM) * stride_xd
        )

        low = tl.load(
            low_x_ptrs,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)

        high = tl.load(
            high_x_ptrs,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32)

        rotated_low = (
            low * cosine
            - high * sine
        )

        rotated_high = (
            high * cosine
            + low * sine
        )

        out_base = (
            token * stride_os
            + head * stride_oh
        )

        low_out_ptrs = (
            out
            + out_base
            + pair * stride_od
        )

        high_out_ptrs = (
            out
            + out_base
            + (pair + HALF_DIM) * stride_od
        )

        tl.store(
            low_out_ptrs,
            rotated_low,
            mask=pair_mask,
        )

        tl.store(
            high_out_ptrs,
            rotated_high,
            mask=pair_mask,
        )
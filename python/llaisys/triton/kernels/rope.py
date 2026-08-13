import triton
import triton.language as tl

from triton.language.extra import libdevice


@triton.jit
def rope_kernel(
    out,
    x,
    pos_ids,
    theta,
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
    token = tl.program_id(0)
    head = tl.program_id(1)
    pair_block = tl.program_id(2)

    HALF_DIM: tl.constexpr = HEAD_DIM // 2

    pair = pair_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair < HALF_DIM

    position = tl.load(pos_ids + token * stride_pos).to(tl.float32)

    pair_f32 = pair.to(tl.float32)

    head_dim_f32 = tl.full((BLOCK_SIZE,), HEAD_DIM, tl.float32)

    exponent = libdevice.div_rn(pair_f32 * 2.0, head_dim_f32)

    theta_f32 = tl.full((BLOCK_SIZE,), theta, tl.float32)

    denominator = libdevice.pow(theta_f32, exponent)

    position_f32 = position + tl.zeros((BLOCK_SIZE,), tl.float32)

    angle = libdevice.div_rn(position_f32, denominator)

    sine = libdevice.sin(angle)
    cosine = libdevice.cos(angle)

    x_base = token * stride_xs + head * stride_xh

    low_x_ptrs = x + x_base + pair * stride_xd
    high_x_ptrs = x + x_base + (pair + HALF_DIM) * stride_xd

    low = tl.load(low_x_ptrs, mask=pair_mask, other=0.0).to(tl.float32)

    high = tl.load(high_x_ptrs, mask=pair_mask, other=0.0).to(tl.float32)

    rotated_low = low * cosine - high * sine
    rotated_high = high * cosine + low * sine

    out_base = token * stride_os + head * stride_oh

    low_out_ptrs = out + out_base + pair * stride_od
    high_out_ptrs = out + out_base + (pair + HALF_DIM) * stride_od

    tl.store(low_out_ptrs, rotated_low, mask=pair_mask)

    tl.store(high_out_ptrs, rotated_high, mask=pair_mask)

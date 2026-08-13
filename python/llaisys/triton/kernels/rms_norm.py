import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    out, x, weight, eps, ncol, stride_xm, stride_xn, stride_om, stride_on, stride_w, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < ncol

    x_offsets = row * stride_xm + cols * stride_xn
    out_offsets = row * stride_om + cols * stride_on

    x_values = tl.load(x + x_offsets, mask=mask, other=0.0).to(tl.float32)

    square_sum = tl.sum(x_values * x_values, axis=0)

    mean_square = square_sum / ncol
    inverse_rms = tl.rsqrt(mean_square + eps)

    weight_values = tl.load(weight + cols * stride_w, mask=mask, other=0.0).to(tl.float32)

    result = x_values * inverse_rms * weight_values

    tl.store(out + out_offsets, result, mask=mask)

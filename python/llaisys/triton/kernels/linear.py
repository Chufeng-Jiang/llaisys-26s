import triton
import triton.language as tl


@triton.jit
def linear_kernel(
    out,
    x,
    weight,
    bias,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_bias,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group

    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = x + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    weight_ptrs = weight + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k_block * BLOCK_K

        x_values = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        weight_values = tl.load(weight_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

        accumulator = tl.dot(x_values, weight_values, acc=accumulator, input_precision="ieee")

        x_ptrs += BLOCK_K * stride_xk
        weight_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias_values = tl.load(bias + offs_n * stride_bias, mask=mask_n, other=0.0).to(tl.float32)

        accumulator += bias_values[None, :]

    output_ptrs = out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    output_mask = mask_m[:, None] & mask_n[None, :]

    tl.store(output_ptrs, accumulator, mask=output_mask)


@triton.jit
def linear_zero_k_kernel(
    out, bias, M, N, stride_om, stride_on, stride_bias, HAS_BIAS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    numel = M * N
    mask = offsets < numel

    rows = offsets // N
    cols = offsets % N

    values = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    if HAS_BIAS:
        values = tl.load(bias + cols * stride_bias, mask=mask, other=0.0).to(tl.float32)

    output_offsets = rows * stride_om + cols * stride_on

    tl.store(out + output_offsets, values, mask=mask)

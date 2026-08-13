import triton
import triton.language as tl


@triton.jit
def add_kernel(c, a, b, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    a_values = tl.load(a + offsets, mask=mask, other=0.0)
    b_values = tl.load(b + offsets, mask=mask, other=0.0)

    tl.store(c + offsets, a_values + b_values, mask=mask)

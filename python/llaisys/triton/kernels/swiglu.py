import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(out, gate, up, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    gate_value = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)

    up_value = tl.load(up + offsets, mask=mask, other=0.0).to(tl.float32)

    denominator = 1.0 + tl.exp(-gate_value)
    result = up_value * gate_value / denominator

    tl.store(out + offsets, result, mask=mask)

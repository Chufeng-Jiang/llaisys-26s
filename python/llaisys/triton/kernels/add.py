import triton
import triton.language as tl


@triton.jit
def add_kernel(
    c,
    a,
    b,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < numel

    a_values = tl.load(
        a + offsets,
        mask=mask,
    )

    b_values = tl.load(
        b + offsets,
        mask=mask,
    )

    tl.store(
        c + offsets,
        a_values + b_values,
        mask=mask,
    )

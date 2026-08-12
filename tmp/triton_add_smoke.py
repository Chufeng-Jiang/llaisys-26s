import torch

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x, y):
    assert x.is_cuda
    assert y.is_cuda
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.is_contiguous()
    assert y.is_contiguous()

    output = torch.empty_like(x)

    n_elements = output.numel()

    grid = (triton.cdiv(n_elements, 256),)

    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=256,
    )

    return output


def check(shape, dtype):
    print(f"Testing shape={shape}, dtype={dtype}")

    x = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
    )

    y = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
    )

    actual = triton_add(x, y)
    expected = x + y

    torch.cuda.synchronize()

    if dtype == torch.float32:
        atol = 1e-5
        rtol = 1e-5
    else:
        atol = 1e-2
        rtol = 1e-2

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
    )

    print("PASS")


def main():
    print("Triton version:", triton.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    shapes = [
        (2, 3),
        (1027,),
        (512, 4096),
    ]

    dtypes = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]

    for shape in shapes:
        for dtype in dtypes:
            check(shape, dtype)

    print()
    print("All Triton Add tests PASSED")


if __name__ == "__main__":
    main()

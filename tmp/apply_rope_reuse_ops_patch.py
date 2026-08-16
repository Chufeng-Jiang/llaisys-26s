from pathlib import Path


OPS_PATH = Path("python/llaisys/triton/ops.py")


OLD = """    grid = (
        sequence_length,
        head_count,
        triton.cdiv(half_dim, block_size),
    )

    out_triton = as_triton_tensor(out, backend)
    x_triton = as_triton_tensor(x, backend)
    pos_triton = as_triton_tensor(pos_ids, backend)

    def launch():
        rope_kernel[grid](
            out_triton,
            x_triton,
            pos_triton,
            theta,
            x_strides[0],
            x_strides[1],
            x_strides[2],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            pos_strides[0],
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            num_warps=config["num_warps"],
        )
"""


NEW = """    # One program handles one token and one pair tile, then reuses
    # the computed RoPE sine/cosine values across all heads.
    grid = (
        sequence_length,
        triton.cdiv(half_dim, block_size),
    )

    out_triton = as_triton_tensor(out, backend)
    x_triton = as_triton_tensor(x, backend)
    pos_triton = as_triton_tensor(pos_ids, backend)

    def launch():
        rope_kernel[grid](
            out_triton,
            x_triton,
            pos_triton,
            theta,
            head_count,
            x_strides[0],
            x_strides[1],
            x_strides[2],
            out_strides[0],
            out_strides[1],
            out_strides[2],
            pos_strides[0],
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            num_warps=config["num_warps"],
        )
"""


def main():
    if not OPS_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {OPS_PATH}. Run this script from the repository root."
        )

    text = OPS_PATH.read_text(encoding="utf-8")

    count = text.count(OLD)

    if count != 1:
        raise RuntimeError(
            "Expected exactly one current RoPE launch block in "
            f"{OPS_PATH}, found {count}. No changes were made."
        )

    updated = text.replace(OLD, NEW, 1)

    backup = OPS_PATH.with_suffix(".py.rope_head_backup")
    backup.write_text(text, encoding="utf-8")
    OPS_PATH.write_text(updated, encoding="utf-8")

    compile(updated, str(OPS_PATH), "exec")

    print(f"Patched: {OPS_PATH}")
    print(f"Backup:  {backup}")
    print("Python syntax check: passed")


if __name__ == "__main__":
    main()
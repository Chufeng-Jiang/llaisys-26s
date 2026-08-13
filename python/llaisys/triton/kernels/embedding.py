import triton
import triton.language as tl


@triton.jit
def embedding_kernel(out_ptr, idx_ptr, weight_ptr, vocabulary_size, embedding_dim, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    column_block = tl.program_id(1)

    columns = column_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    column_mask = columns < embedding_dim

    embedding_index = tl.load(idx_ptr + row).to(tl.int64)

    valid_index = (embedding_index >= 0) & (embedding_index < vocabulary_size)
    safe_index = tl.where(valid_index, embedding_index, 0)

    embedding_dim_i64 = embedding_dim.to(tl.int64)
    columns_i64 = columns.to(tl.int64)
    row_i64 = row.to(tl.int64)

    weight_offsets = safe_index * embedding_dim_i64 + columns_i64
    output_offsets = row_i64 * embedding_dim_i64 + columns_i64

    copy_mask = column_mask & valid_index

    values = tl.load(weight_ptr + weight_offsets, mask=copy_mask, other=0.0)

    tl.store(out_ptr + output_offsets, values, mask=copy_mask)

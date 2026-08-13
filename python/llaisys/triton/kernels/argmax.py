import triton
import triton.language as tl


@triton.jit
def argmax_stage1_kernel(x_ptr, out_val_ptr, out_idx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)

    lanes = tl.arange(0, BLOCK_SIZE)
    lanes_i64 = lanes.to(tl.int64)

    block_start = block_id.to(tl.int64) * BLOCK_SIZE
    offsets = block_start + lanes_i64
    mask = offsets < n_elements

    values = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))

    is_nan = values != values
    nan_candidates = mask & is_nan

    first_nan_lane = tl.min(tl.where(nan_candidates, lanes, BLOCK_SIZE), axis=0)
    has_nan = first_nan_lane < BLOCK_SIZE

    numeric_values = tl.where(mask & ~is_nan, values, -float("inf"))

    numeric_max, numeric_lane = tl.max(numeric_values, axis=0, return_indices=True, return_indices_tie_break_left=True)

    winner_lane = tl.where(has_nan, first_nan_lane, numeric_lane).to(tl.int64)

    winner_offset = block_start + winner_lane
    winner_value = tl.load(x_ptr + winner_offset)

    tl.store(out_val_ptr + block_id, winner_value)
    tl.store(out_idx_ptr + block_id, winner_offset)


@triton.jit
def argmax_stage_n_kernel(val_ptr, idx_ptr, out_val_ptr, out_idx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)

    lanes = tl.arange(0, BLOCK_SIZE)
    lanes_i64 = lanes.to(tl.int64)

    block_start = block_id.to(tl.int64) * BLOCK_SIZE
    offsets = block_start + lanes_i64
    mask = offsets < n_elements

    values = tl.load(val_ptr + offsets, mask=mask, other=-float("inf"))

    is_nan = values != values
    nan_candidates = mask & is_nan

    first_nan_lane = tl.min(tl.where(nan_candidates, lanes, BLOCK_SIZE), axis=0)
    has_nan = first_nan_lane < BLOCK_SIZE

    numeric_values = tl.where(mask & ~is_nan, values, -float("inf"))

    numeric_max, numeric_lane = tl.max(numeric_values, axis=0, return_indices=True, return_indices_tie_break_left=True)

    winner_lane = tl.where(has_nan, first_nan_lane, numeric_lane).to(tl.int64)

    winner_offset = block_start + winner_lane
    winner_value = tl.load(val_ptr + winner_offset)
    winner_index = tl.load(idx_ptr + winner_offset)

    tl.store(out_val_ptr + block_id, winner_value)
    tl.store(out_idx_ptr + block_id, winner_index)

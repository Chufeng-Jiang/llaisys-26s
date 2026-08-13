import triton
import triton.language as tl

from triton.language.extra import libdevice


@triton.jit
def self_attention_kernel(
    out,
    q,
    k,
    v,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    v_stride_s,
    v_stride_h,
    v_stride_d,
    out_stride_s,
    out_stride_h,
    out_stride_d,
    seqlen,
    total_len,
    scale,
    GROUP_SIZE: tl.constexpr,
    QK_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    query_block = tl.program_id(0)
    query_head = tl.program_id(1)
    value_block = tl.program_id(2)

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_tile_offsets = tl.arange(0, BLOCK_N)
    dim_tile_offsets = tl.arange(0, BLOCK_D)
    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)

    query_mask = query_offsets < seqlen
    value_mask = value_offsets < V_DIM

    kv_head = query_head // GROUP_SIZE
    prefix_length = total_len - seqlen

    running_max = tl.where(query_mask, -float("inf"), 0.0).to(tl.float32)

    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    accumulator = tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)

    scale_f32 = tl.full((), scale, tl.float32)

    for key_start in tl.range(0, total_len, BLOCK_N):
        key_offsets = key_start + key_tile_offsets
        key_mask = key_offsets < total_len

        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for dim_start in range(0, QK_DIM, BLOCK_D):
            dim_offsets = dim_start + dim_tile_offsets
            dim_mask = dim_offsets < QK_DIM

            q_ptrs = (
                q + query_offsets[:, None] * q_stride_s + query_head * q_stride_h + dim_offsets[None, :] * q_stride_d
            )

            k_ptrs = k + key_offsets[:, None] * k_stride_s + kv_head * k_stride_h + dim_offsets[None, :] * k_stride_d

            q_values = tl.load(q_ptrs, mask=query_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)

            k_values = tl.load(k_ptrs, mask=key_mask[:, None] & dim_mask[None, :], other=0.0).to(tl.float32)

            scores = tl.dot(q_values, tl.trans(k_values), acc=scores, out_dtype=tl.float32, input_precision="ieee")

        scores *= scale_f32

        causal_limit = prefix_length + query_offsets
        causal_mask = key_offsets[None, :] <= causal_limit[:, None]

        valid = query_mask[:, None] & key_mask[None, :] & causal_mask

        scores = tl.where(valid, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)

        new_max = tl.maximum(running_max, tile_max)

        previous_scale = libdevice.exp(running_max - new_max)

        probabilities = libdevice.exp(scores - new_max[:, None])

        probabilities = tl.where(valid, probabilities, 0.0)

        tile_sum = tl.sum(probabilities, axis=1)

        v_ptrs = v + key_offsets[:, None] * v_stride_s + kv_head * v_stride_h + value_offsets[None, :] * v_stride_d

        v_values = tl.load(v_ptrs, mask=key_mask[:, None] & value_mask[None, :], other=0.0).to(tl.float32)

        accumulator *= previous_scale[:, None]

        accumulator = tl.dot(probabilities, v_values, acc=accumulator, out_dtype=tl.float32, input_precision="ieee")

        running_sum = running_sum * previous_scale + tile_sum

        running_max = new_max

    running_sum = tl.where(query_mask, running_sum, 1.0)

    result = accumulator / running_sum[:, None]

    out_ptrs = (
        out + query_offsets[:, None] * out_stride_s + query_head * out_stride_h + value_offsets[None, :] * out_stride_d
    )

    output_mask = query_mask[:, None] & value_mask[None, :]

    tl.store(out_ptrs, result, mask=output_mask)

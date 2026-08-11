#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::metax {

// Compute causal grouped-query self-attention:
//
//     attn_val = causal_softmax(Q * K^T * scale) * V
//
// Tensor layouts are contiguous and batch-free:
//
//     attn_val: [seqlen, nhead, dv]
//     q:        [seqlen, nhead, d]
//     k:        [total_len, nkvhead, d]
//     v:        [total_len, nkvhead, dv]
//
// When total_len > seqlen, the causal diagonal is aligned to the
// bottom right so that query i can attend through:
//
//     key = total_len - seqlen + i
//
// This MetaX backend is the portability baseline. The attention
// algorithm itself is shared through:
//
//     ../cuda_compat/self_attention_cuda_compat.cuh
//
// The MetaX adapter owns:
//
//     dtype dispatch
//     launch policy
//     stream conversion
//     launch-error handling
//
void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    float scale,
    llaisysDataType_t type,
    std::size_t seqlen,
    std::size_t nhead,
    std::size_t dv,
    std::size_t total_len,
    std::size_t nkvhead,
    std::size_t d,
    llaisysStream_t stream);

} // namespace llaisys::ops::metax
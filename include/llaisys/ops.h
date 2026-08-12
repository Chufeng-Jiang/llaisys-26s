#ifndef LLAISYS_OPS_H
#define LLAISYS_OPS_H

#include "tensor.h"

__C {
    __export int llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);

    __export int llaisysArgmax(
        llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals);

    __export int llaisysEmbedding(
        llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight);

    __export int llaisysLinear(
        llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias);

    __export int llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in);

    __export int llaisysRmsNorm(
        llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps);

    __export int llaisysROPE(
        llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);

    __export int llaisysSelfAttention(
        llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v,
        float scale);

    __export int llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);
}

#endif
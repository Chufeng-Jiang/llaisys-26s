#include "llaisys/ops.h"

#include "error.hpp"
#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

__C int llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
    return llaisys::c_api::guard([&]() { llaisys::ops::add(c->tensor, a->tensor, b->tensor); })
             ? 0
             : -1;
}

__C int llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
    return llaisys::c_api::guard(
               [&]() { llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor); })
             ? 0
             : -1;
}

__C int llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
    return llaisys::c_api::guard(
               [&]() { llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor); })
             ? 0
             : -1;
}

__C int llaisysLinear(
    llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
    return llaisys::c_api::guard([&]() {
        llaisys::ops::linear(
            out->tensor, in->tensor, weight->tensor, bias == nullptr ? nullptr : bias->tensor);
    })
             ? 0
             : -1;
}

__C int llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
    return llaisys::c_api::guard([&]() { llaisys::ops::rearrange(out->tensor, in->tensor); }) ? 0
                                                                                              : -1;
}

__C int llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
    return llaisys::c_api::guard(
               [&]() { llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps); })
             ? 0
             : -1;
}

__C int llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
    return llaisys::c_api::guard(
               [&]() { llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta); })
             ? 0
             : -1;
}

__C int llaisysSelfAttention(
    llaisysTensor_t attn_val,
    llaisysTensor_t q,
    llaisysTensor_t k,
    llaisysTensor_t v,
    float scale) {
    return llaisys::c_api::guard([&]() {
        llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    })
             ? 0
             : -1;
}

__C int llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
    return llaisys::c_api::guard(
               [&]() { llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor); })
             ? 0
             : -1;
}
import ctypes

from llaisys.libllaisys import (
    LIB_LLAISYS,
    DataType,
    DeviceType,
    LlaisysQwen2Meta,
)


meta = LlaisysQwen2Meta(
    dtype=int(DataType.F32),
    nlayer=2,
    hs=8,
    nh=2,
    nkvh=1,
    dh=4,
    di=16,
    maxseq=32,
    voc=64,
    epsilon=1e-5,
    theta=10000.0,
    end_token=2,
)

device_ids = (ctypes.c_int * 1)(0)

model = LIB_LLAISYS.llaisysQwen2ModelCreate(
	ctypes.byref(meta),
	int(DeviceType.CPU),
	device_ids,
	1,
)

if not model:
	raise RuntimeError("Model creation returned null.")

model_address = ctypes.cast(
	model,
	ctypes.c_void_p,
).value

print("Model created:", hex(model_address))

weights_pointer = LIB_LLAISYS.llaisysQwen2ModelWeights(
	model
)

if not weights_pointer:
	LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
	raise RuntimeError("Weights pointer is null.")

weights = weights_pointer.contents

print("Weights pointer retrieved.")
print("in_embed:", weights.in_embed)
print("layer 0 q weight:", weights.attn_q_w[0])
print("layer 1 q weight:", weights.attn_q_w[1])

tokens = (ctypes.c_int64 * 1)(42)

result = LIB_LLAISYS.llaisysQwen2ModelInfer(
    model,
    tokens,
    1,
)

print("Temporary infer result:", result)

if result != meta.end_token:
    LIB_LLAISYS.llaisysQwen2ModelDestroy(model)
    raise RuntimeError(
        f"Expected {meta.end_token}, got {result}."
    )

LIB_LLAISYS.llaisysQwen2ModelDestroy(model)

print("Model destroyed.")
print("Qwen2 ctypes binding test passed.")
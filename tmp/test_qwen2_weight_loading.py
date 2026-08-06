import ctypes
from pathlib import Path

from llaisys.libllaisys import (
	DeviceType,
	LIB_LLAISYS,
)
from llaisys.models.qwen2 import Qwen2


model_path = Path(
	Path("tmp/model_path.txt").read_text(
		encoding="utf-8"
	).strip()
)

with Qwen2(
	model_path,
	DeviceType.CPU,
) as model:
	print("\n===== Weight loading result =====")
	print("Weights loaded:", model._weights_loaded)
	print(
		"Owned tensor count:",
		len(model._weight_tensors),
	)

	if not model._weights_loaded:
		raise RuntimeError(
			"Model weights were not loaded."
		)

	if len(model._weight_tensors) != 339:
		raise RuntimeError(
			"Expected 339 loaded weights, got "
			f"{len(model._weight_tensors)}."
		)

	global_weights = {
		"embedding": model._weights.in_embed,
		"output": model._weights.out_embed,
		"final norm": model._weights.out_norm_w,
		"layer 0 q": model._weights.attn_q_w[0],
		"layer 27 down": model._weights.mlp_down_w[27],
	}

	for name, tensor in global_weights.items():
		if not tensor:
			raise RuntimeError(
				f"Null tensor handle: {name}"
			)

		ndim = LIB_LLAISYS.tensorGetNdim(
			tensor
		)

		ShapeArray = ctypes.c_size_t * ndim
		shape = ShapeArray()

		LIB_LLAISYS.tensorGetShape(
			tensor,
			shape,
		)

		print(
			f"{name}: "
			f"shape={list(shape)}, "
			f"address={hex(int(tensor))}"
		)

print("\nQwen2 full weight loading test passed.")

import argparse
import gc
import os
import sys
from pathlib import Path


ROOT = Path(
	"/data/llaisys-26s"
)

sys.path.insert(
	0,
	str(ROOT / "python"),
)

sys.path.insert(
	0,
	str(ROOT / "test"),
)


import llaisys
import torch

from test_utils import (
	benchmark_llaisys,
	check_equal,
	random_tensor,
)


# ============================================================
# DeepSeek-R1-Distill-Qwen-1.5B / Qwen2 real biased Linear
# shapes.
#
# hidden_size = 1536
# num_attention_heads = 12
# num_key_value_heads = 2
# head_dim = 128
#
# q_proj:
#     in  = hidden_size = 1536
#     out = 12 * 128 = 1536
#
# k_proj / v_proj:
#     in  = hidden_size = 1536
#     out = 2 * 128 = 256
#
# These are the two unique biased Linear shapes.
# ============================================================

SHAPES = [
	(
		"q_proj",
		1536,
		1536,
	),
	(
		"kv_proj",
		256,
		1536,
	),
]

M_VALUES = [
	1,
	2,
	4,
	8,
	16,
	32,
	64,
	128,
	256,
	512,
]

DTYPE_PRECISION = {
	"f32": (
		1e-5,
		1e-5,
	),
	"f16": (
		1e-3,
		1e-3,
	),
	"bf16": (
		1e-2,
		1e-2,
	),
}

VALID_IMPLS = [
	"portable",
	"fused",
]


def torch_linear(
	out,
	x,
	w,
	bias,
):
	torch.nn.functional.linear(
		x,
		w,
		bias,
		out=out,
	)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--device",
		default="metax",
		choices=[
			"metax",
		],
	)

	parser.add_argument(
		"--impl",
		required=True,
		choices=VALID_IMPLS,
	)

	parser.add_argument(
		"--order",
		default="asc",
		choices=[
			"asc",
			"desc",
		],
	)

	parser.add_argument(
		"--shape-order",
		default="q_proj,kv_proj",
	)

	parser.add_argument(
		"--dtype-order",
		default="f32,f16,bf16",
	)

	parser.add_argument(
		"--skip-correctness",
		action="store_true",
	)

	args = parser.parse_args()

	os.environ[
		"LLAISYS_METAX_LINEAR_IMPL"
	] = args.impl

	shape_lookup = {
		name: (
			n,
			k,
		)
		for (
			name,
			n,
			k,
		) in SHAPES
	}

	shape_order = [
		item.strip()
		for item in args.shape_order.split(",")
		if item.strip()
	]

	for shape_name in shape_order:
		if shape_name not in shape_lookup:
			raise ValueError(
				f"Unsupported shape: "
				f"{shape_name}"
			)

	dtype_order = [
		item.strip()
		for item in args.dtype_order.split(",")
		if item.strip()
	]

	for dtype_name in dtype_order:
		if dtype_name not in DTYPE_PRECISION:
			raise ValueError(
				f"Unsupported dtype: "
				f"{dtype_name}"
			)

	m_values = list(
		M_VALUES
	)

	if args.order == "desc":
		m_values.reverse()

	print(
		"============================================================"
	)

	print(
		"Qwen2 real Q/K/V Linear fused-policy characterization"
	)

	print(
		f"device={args.device}"
	)

	print(
		f"impl={args.impl}"
	)

	print(
		f"M_order={args.order}"
	)

	print(
		f"shape_order={','.join(shape_order)}"
	)

	print(
		f"dtype_order={','.join(dtype_order)}"
	)

	print(
		"bias=True"
	)

	print(
		"============================================================"
	)

	for shape_name in shape_order:

		n, k = shape_lookup[
			shape_name
		]

		for dtype_name in dtype_order:

			atol, rtol = (
				DTYPE_PRECISION[
					dtype_name
				]
			)

			# ------------------------------------------------
			# Reuse the same weight and bias across all M for
			# this shape/dtype.
			# ------------------------------------------------

			w, w_ = random_tensor(
				(
					n,
					k,
				),
				dtype_name,
				args.device,
				scale=0.01,
			)

			bias, bias_ = random_tensor(
				(
					n,
				),
				dtype_name,
				args.device,
				scale=0.1,
			)

			for m in m_values:

				x, x_ = random_tensor(
					(
						m,
						k,
					),
					dtype_name,
					args.device,
					scale=0.1,
				)

				out, out_ = random_tensor(
					(
						m,
						n,
					),
					dtype_name,
					args.device,
				)

				if not args.skip_correctness:

					torch_linear(
						out,
						x,
						w,
						bias,
					)

					llaisys.Ops.linear(
						out_,
						x_,
						w_,
						bias_,
					)

					assert check_equal(
						out_,
						out,
						atol=atol,
						rtol=rtol,
					), (
						f"Linear correctness failed: "
						f"impl={args.impl}, "
						f"shape={shape_name}, "
						f"M={m}, "
						f"N={n}, "
						f"K={k}, "
						f"dtype={dtype_name}"
					)

				# --------------------------------------------
				# Warm cached mcBLAS/mcBLASLt state before
				# timing.
				# --------------------------------------------

				llaisys.Ops.linear(
					out_,
					x_,
					w_,
					bias_,
				)

				print(
					f"PROFILE QKVLinearFusedAB "
					f"impl={args.impl} "
					f"shape={shape_name} "
					f"M={m} "
					f"N={n} "
					f"K={k} "
					f"dtype={dtype_name}"
				)

				benchmark_llaisys(
					lambda: llaisys.Ops.linear(
						out_,
						x_,
						w_,
						bias_,
					),
					args.device,
				)

				del x
				del x_
				del out
				del out_

				gc.collect()

			del w
			del w_
			del bias
			del bias_

			gc.collect()

	print()

	print(
		f"QKV Linear impl={args.impl} "
		f"round PASSED"
	)


if __name__ == "__main__":
	main()
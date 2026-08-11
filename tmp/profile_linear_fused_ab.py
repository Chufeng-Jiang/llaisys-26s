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


N = 4096
K = 4096

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

VALID_IMPLS = [
	"portable",
	"fused",
	"auto",
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
		"--dtype-order",
		default="f32,f16,bf16",
	)

	parser.add_argument(
		"--skip-correctness",
		action="store_true",
	)

	args = parser.parse_args()

	# Implementation is read once by the native backend.
	os.environ[
		"LLAISYS_METAX_LINEAR_IMPL"
	] = args.impl

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
		"Linear portable-vs-fused"
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
		f"dtype_order={','.join(dtype_order)}"
	)

	print(
		f"N={N} K={K} bias=True"
	)

	print(
		"============================================================"
	)

	resources = {}

	for dtype_name in dtype_order:

		w, w_ = random_tensor(
			(
				N,
				K,
			),
			dtype_name,
			args.device,
			scale=0.01,
		)

		bias, bias_ = random_tensor(
			(
				N,
			),
			dtype_name,
			args.device,
			scale=0.1,
		)

		resources[
			dtype_name
		] = (
			w,
			w_,
			bias,
			bias_,
		)

	for m in m_values:

		for dtype_name in dtype_order:

			atol, rtol = (
				DTYPE_PRECISION[
					dtype_name
				]
			)

			(
				w,
				w_,
				bias,
				bias_,
			) = resources[
				dtype_name
			]

			x, x_ = random_tensor(
				(
					m,
					K,
				),
				dtype_name,
				args.device,
				scale=0.1,
			)

			out, out_ = random_tensor(
				(
					m,
					N,
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
					f"M={m}, "
					f"dtype={dtype_name}"
				)

			# Warm the plan before the timed helper.
			llaisys.Ops.linear(
				out_,
				x_,
				w_,
				bias_,
			)

			print(
				f"PROFILE LinearFusedAB "
				f"impl={args.impl} "
				f"M={m} "
				f"N={N} "
				f"K={K} "
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

	print()

	print(
		f"Linear impl={args.impl} round PASSED"
	)


if __name__ == "__main__":
	main()
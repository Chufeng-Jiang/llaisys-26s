import argparse
import gc
import os
import sys
from pathlib import Path

ROOT = Path("/data/llaisys-26s")

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
	benchmark,
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


def run_profile(
	path_name,
	out_,
	x_,
	w_,
	bias_,
	out,
	x,
	w,
	bias,
	device_name,
	m,
	dtype_name,
):
	print(
		f"PROFILE LinearBias "
		f"path={path_name} "
		f"M={m} "
		f"N={N} "
		f"K={K} "
		f"dtype={dtype_name}"
	)

	benchmark(
		lambda: torch_linear(
			out,
			x,
			w,
			bias,
		),
		lambda: llaisys.Ops.linear(
			out_,
			x_,
			w_,
			bias_,
		),
		device_name,
	)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--device",
		default="metax",
		choices=[
			"metax",
			"nvidia",
		],
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
		"--pair-order",
		default="nobias-first",
		choices=[
			"nobias-first",
			"bias-first",
		],
	)

	parser.add_argument(
		"--dtype-order",
		default="f32,f16,bf16",
	)

	args = parser.parse_args()

	dtypes = [
		x.strip()
		for x in args.dtype_order.split(",")
		if x.strip()
	]

	m_values = list(
		M_VALUES
	)

	if args.order == "desc":
		m_values.reverse()

	print(
		"============================================================"
	)

	print(
		"Linear bias A/B"
	)

	print(
		f"device={args.device}"
	)

	print(
		f"M_order={args.order}"
	)

	print(
		f"pair_order={args.pair_order}"
	)

	print(
		f"dtype_order={','.join(dtypes)}"
	)

	print(
		"N=4096 K=4096"
	)

	print(
		"============================================================"
	)

	# ========================================================
	# One weight + bias vector per dtype.
	# ========================================================

	resources = {}

	for dtype_name in dtypes:

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

		resources[dtype_name] = (
			w,
			w_,
			bias,
			bias_,
		)

	# ========================================================
	# A/B
	# ========================================================

	for m in m_values:

		for dtype_name in dtypes:

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

			# ------------------------------------------------
			# Separate outputs.
			# ------------------------------------------------

			out_nobias, out_nobias_ = (
				random_tensor(
					(
						m,
						N,
					),
					dtype_name,
					args.device,
				)
			)

			out_bias, out_bias_ = (
				random_tensor(
					(
						m,
						N,
					),
					dtype_name,
					args.device,
				)
			)

			# ------------------------------------------------
			# Correctness: no bias
			# ------------------------------------------------

			torch_linear(
				out_nobias,
				x,
				w,
				None,
			)

			llaisys.Ops.linear(
				out_nobias_,
				x_,
				w_,
				None,
			)

			assert check_equal(
				out_nobias_,
				out_nobias,
				atol=atol,
				rtol=rtol,
			)

			# ------------------------------------------------
			# Correctness: bias
			# ------------------------------------------------

			torch_linear(
				out_bias,
				x,
				w,
				bias,
			)

			llaisys.Ops.linear(
				out_bias_,
				x_,
				w_,
				bias_,
			)

			assert check_equal(
				out_bias_,
				out_bias,
				atol=atol,
				rtol=rtol,
			)

			def profile_nobias():
				run_profile(
					"nobias",
					out_nobias_,
					x_,
					w_,
					None,
					out_nobias,
					x,
					w,
					None,
					args.device,
					m,
					dtype_name,
				)

			def profile_bias():
				run_profile(
					"bias",
					out_bias_,
					x_,
					w_,
					bias_,
					out_bias,
					x,
					w,
					bias,
					args.device,
					m,
					dtype_name,
				)

			if (
				args.pair_order
				== "nobias-first"
			):
				profile_nobias()
				profile_bias()

			else:
				profile_bias()
				profile_nobias()

			del x
			del x_

			del out_nobias
			del out_nobias_

			del out_bias
			del out_bias_

			gc.collect()

	print()
	print(
		"Linear bias A/B round PASSED"
	)


if __name__ == "__main__":
	main()

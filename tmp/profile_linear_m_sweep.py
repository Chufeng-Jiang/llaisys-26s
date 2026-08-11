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
):
	torch.nn.functional.linear(
		x,
		w,
		None,
		out=out,
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
		"--dtype-order",
		default="f32,f16,bf16",
		type=str,
	)

	args = parser.parse_args()

	dtype_order = [
		item.strip()
		for item in args.dtype_order.split(",")
		if item.strip()
	]

	for dtype_name in dtype_order:
		if dtype_name not in DTYPE_PRECISION:
			raise ValueError(
				f"Unsupported dtype: {dtype_name}"
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
		"Linear MetaX M-sweep"
	)

	print(
		f"device={args.device}"
	)

	print(
		f"order={args.order}"
	)

	print(
		f"dtype_order={','.join(dtype_order)}"
	)

	print(
		f"N={N} K={K} bias=False"
	)

	print(
		"============================================================"
	)

	# ========================================================
	# Keep one 4096x4096 weight per dtype.
	#
	# This removes repeated weight generation/copy from the
	# experiment setup while keeping all timed work unchanged.
	# ========================================================

	weights = {}

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

		weights[dtype_name] = (
			w,
			w_,
		)

	# ========================================================
	# Sweep M.
	# ========================================================

	for m in m_values:

		for dtype_name in dtype_order:

			atol, rtol = (
				DTYPE_PRECISION[
					dtype_name
				]
			)

			w, w_ = (
				weights[
					dtype_name
				]
			)

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

			# ------------------------------------------------
			# Correctness guard before timing.
			# ------------------------------------------------

			torch_linear(
				out,
				x,
				w,
			)

			llaisys.Ops.linear(
				out_,
				x_,
				w_,
				None,
			)

			assert check_equal(
				out_,
				out,
				atol=atol,
				rtol=rtol,
			), (
				f"Linear correctness failed: "
				f"M={m}, N={N}, K={K}, "
				f"dtype={dtype_name}"
			)

			# ------------------------------------------------
			# Machine-readable workload marker.
			#
			# The summary parser associates the next
			# "LLAISYS metax: median=..." record with this.
			# ------------------------------------------------

			print(
				f"PROFILE Linear "
				f"M={m} "
				f"N={N} "
				f"K={K} "
				f"dtype={dtype_name} "
				f"bias=False"
			)

			benchmark(
				lambda: torch_linear(
					out,
					x,
					w,
				),
				lambda: llaisys.Ops.linear(
					out_,
					x_,
					w_,
					None,
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
		"Linear M-sweep round PASSED"
	)


if __name__ == "__main__":
	main()

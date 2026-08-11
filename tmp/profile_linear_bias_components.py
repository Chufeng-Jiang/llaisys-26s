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

from test_utils import (
	benchmark_llaisys,
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

VALID_PATHS = [
	"nobias",
	"broadcast_beta0",
	"normal",
]

VALID_DTYPES = [
	"f32",
	"f16",
	"bf16",
]


def configure_path(
	path_name,
):
	if path_name == "nobias":
		os.environ.pop(
			"LLAISYS_METAX_LINEAR_BIAS_MODE",
			None,
		)

		return

	if path_name == "broadcast_beta0":
		os.environ[
			"LLAISYS_METAX_LINEAR_BIAS_MODE"
		] = "broadcast_beta0"

		return

	if path_name == "normal":
		os.environ[
			"LLAISYS_METAX_LINEAR_BIAS_MODE"
		] = "normal"

		return

	raise ValueError(
		f"Unsupported path: {path_name}"
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
		"--path",
		required=True,
		choices=VALID_PATHS,
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

	args = parser.parse_args()

	# ========================================================
	# IMPORTANT:
	#
	# MetaX Linear caches LLAISYS_METAX_LINEAR_BIAS_MODE on
	# the first biased Linear invocation in this process.
	#
	# Therefore each process profiles exactly ONE path.
	# ========================================================

	configure_path(
		args.path
	)

	dtype_order = [
		item.strip()
		for item in args.dtype_order.split(",")
		if item.strip()
	]

	for dtype_name in dtype_order:
		if dtype_name not in VALID_DTYPES:
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
		"Linear bias-component profiling"
	)

	print(
		f"device={args.device}"
	)

	print(
		f"path={args.path}"
	)

	print(
		f"M_order={args.order}"
	)

	print(
		f"dtype_order={','.join(dtype_order)}"
	)

	print(
		f"N={N} K={K}"
	)

	print(
		"============================================================"
	)

	# ========================================================
	# Keep one device-side weight and bias per dtype.
	# ========================================================

	resources = {}

	for dtype_name in dtype_order:

		_, w_ = random_tensor(
			(
				N,
				K,
			),
			dtype_name,
			args.device,
			scale=0.01,
		)

		_, bias_ = random_tensor(
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
			w_,
			bias_,
		)

	# ========================================================
	# M sweep.
	# ========================================================

	for m in m_values:

		for dtype_name in dtype_order:

			(
				w_,
				bias_,
			) = resources[
				dtype_name
			]

			_, x_ = random_tensor(
				(
					m,
					K,
				),
				dtype_name,
				args.device,
				scale=0.1,
			)

			_, out_ = random_tensor(
				(
					m,
					N,
				),
				dtype_name,
				args.device,
			)

			if args.path == "nobias":
				active_bias_ = None
			else:
				active_bias_ = bias_

			print(
				f"PROFILE LinearBiasComponent "
				f"path={args.path} "
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
					active_bias_,
				),
				args.device,
			)

			del x_
			del out_

			gc.collect()

	print()

	print(
		f"Linear bias-component "
		f"path={args.path} PASSED"
	)


if __name__ == "__main__":
	main()
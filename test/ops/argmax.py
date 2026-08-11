import os
import sys
from ctypes import c_void_p

import torch

parent_dir = os.path.abspath(
	os.path.join(
		os.path.dirname(__file__),
		"..",
	)
)
sys.path.insert(0, parent_dir)

import llaisys

from test_utils import (
	benchmark_llaisys,
	check_equal,
	random_tensor,
	zero_tensor,
)


# ============================================================
# PyTorch reference
# ============================================================


def torch_argmax(
	max_idx,
	max_val,
	vals,
):
	torch.max(
		vals,
		keepdim=True,
		dim=-1,
		out=(
			max_val,
			max_idx,
		),
	)


# ============================================================
# DType helpers
# ============================================================


TORCH_DTYPES = {
	"f32": torch.float32,
	"f16": torch.float16,
	"bf16": torch.bfloat16,
}


# ============================================================
# Random differential correctness + optional profiling
# ============================================================


def test_op_argmax(
	shape,
	dtype_name="f32",
	device_name="cpu",
	profile=False,
):
	print(
		f"   shape {shape} "
		f"dtype <{dtype_name}>"
	)

	vals, vals_ = random_tensor(
		shape,
		dtype_name,
		device_name,
	)

	max_idx, max_idx_ = zero_tensor(
		(1,),
		"i64",
		device_name,
	)

	max_val, max_val_ = zero_tensor(
		(1,),
		dtype_name,
		device_name,
	)

	# --------------------------------------------------------
	# Correctness reference
	# --------------------------------------------------------

	torch_argmax(
		max_idx,
		max_val,
		vals,
	)

	# --------------------------------------------------------
	# LLAISYS
	# --------------------------------------------------------

	llaisys.Ops.argmax(
		max_idx_,
		max_val_,
		vals_,
	)

	assert check_equal(
		max_val_,
		max_val,
		strict=True,
	), (
		f"Argmax value mismatch: "
		f"shape={shape}, "
		f"dtype={dtype_name}, "
		f"device={device_name}"
	)

	assert check_equal(
		max_idx_,
		max_idx,
		strict=True,
	), (
		f"Argmax index mismatch: "
		f"shape={shape}, "
		f"dtype={dtype_name}, "
		f"device={device_name}"
	)

	# --------------------------------------------------------
	# LLAISYS-only profiling
	#
	# PyTorch remains the correctness oracle only.
	# --------------------------------------------------------

	if profile:
		print(
			f"        Argmax shape={shape} "
			f"dtype={dtype_name}:",
			end=" ",
		)

		benchmark_llaisys(
			lambda: llaisys.Ops.argmax(
				max_idx_,
				max_val_,
				vals_,
			),
			device_name,
		)


# ============================================================
# Exact semantic correctness
# ============================================================


def test_semantic_case(
	name,
	values,
	expected_index,
	dtype_name,
	device_name,
):
	print(
		f"   semantic {name} "
		f"dtype <{dtype_name}>"
	)

	torch_dtype = TORCH_DTYPES[
		dtype_name
	]

	vals = torch.tensor(
		values,
		dtype=torch_dtype,
		device="cpu",
	).contiguous()

	shape = tuple(
		vals.shape
	)

	# --------------------------------------------------------
	# Allocate LLAISYS input then overwrite with exact data.
	# --------------------------------------------------------

	_, vals_ = random_tensor(
		shape,
		dtype_name,
		device_name,
	)

	vals_.load(
		c_void_p(
			vals.data_ptr()
		)
	)

	# --------------------------------------------------------
	# Reference outputs
	# --------------------------------------------------------

	max_idx = torch.zeros(
		(1,),
		dtype=torch.int64,
		device="cpu",
	)

	max_val = torch.zeros(
		(1,),
		dtype=torch_dtype,
		device="cpu",
	)

	# --------------------------------------------------------
	# LLAISYS outputs
	# --------------------------------------------------------

	_, max_idx_ = zero_tensor(
		(1,),
		"i64",
		device_name,
	)

	_, max_val_ = zero_tensor(
		(1,),
		dtype_name,
		device_name,
	)

	torch_argmax(
		max_idx,
		max_val,
		vals,
	)

	reference_index = int(
		max_idx.item()
	)

	assert (
		reference_index
		== expected_index
	), (
		f"Incorrect semantic-test expectation: "
		f"case={name}, "
		f"dtype={dtype_name}, "
		f"expected={expected_index}, "
		f"torch={reference_index}"
	)

	llaisys.Ops.argmax(
		max_idx_,
		max_val_,
		vals_,
	)

	# --------------------------------------------------------
	# Index semantics
	# --------------------------------------------------------

	assert check_equal(
		max_idx_,
		max_idx,
		strict=True,
	), (
		f"Argmax semantic index mismatch: "
		f"case={name}, "
		f"dtype={dtype_name}, "
		f"device={device_name}"
	)

	# --------------------------------------------------------
	# Value semantics
	#
	# check_equal cannot directly treat NaN == NaN.
	# For NaN cases we verify the selected NaN index.
	# --------------------------------------------------------

	reference_is_nan = bool(
		torch.isnan(
			max_val
		).item()
	)

	if reference_is_nan:
		print(
			f"      NaN reference selected "
			f"at index {reference_index}"
		)
	else:
		assert check_equal(
			max_val_,
			max_val,
			strict=True,
		), (
			f"Argmax semantic value mismatch: "
			f"case={name}, "
			f"dtype={dtype_name}, "
			f"device={device_name}"
		)


# ============================================================
# Semantic suite
# ============================================================


def run_semantic_tests(
	device_name,
	dtype_name,
):
	cases = [
		(
			"single_element",
			[
				5.0,
			],
			0,
		),
		(
			"normal",
			[
				1.0,
				4.0,
				2.0,
				3.0,
			],
			1,
		),
		(
			"maximum_first",
			[
				9.0,
				1.0,
				2.0,
				3.0,
			],
			0,
		),
		(
			"maximum_last",
			[
				1.0,
				2.0,
				3.0,
				9.0,
			],
			3,
		),
		(
			"duplicate_maximum",
			[
				1.0,
				9.0,
				3.0,
				9.0,
			],
			1,
		),
		(
			"all_negative",
			[
				-9.0,
				-3.0,
				-7.0,
			],
			1,
		),
		(
			"positive_infinity",
			[
				1.0,
				float("inf"),
				100.0,
			],
			1,
		),
		(
			"negative_infinity",
			[
				float("-inf"),
				-2.0,
				-3.0,
			],
			1,
		),
		(
			"wide_duplicate_maximum",
			[
				1.0,
				2.0,
				99.0,
				4.0,
				5.0,
				6.0,
				7.0,
				99.0,
			],
			2,
		),
		(
			"single_nan",
			[
				1.0,
				float("nan"),
				100.0,
			],
			1,
		),
		(
			"multiple_nan",
			[
				float("nan"),
				1.0,
				float("nan"),
			],
			0,
		),
		(
			"nan_after_numeric_max",
			[
				1000.0,
				999.0,
				float("nan"),
			],
			2,
		),
		(
			"nan_before_numeric_max",
			[
				float("nan"),
				999.0,
				1000.0,
			],
			0,
		),
		(
			"wide_multiple_nan",
			[
				1.0,
				2.0,
				float("nan"),
				4.0,
				5.0,
				6.0,
				float("nan"),
				1000.0,
			],
			2,
		),
	]

	for (
		name,
		values,
		expected_index,
	) in cases:
		test_semantic_case(
			name,
			values,
			expected_index,
			dtype_name,
			device_name,
		)


# ============================================================
# Profile shape sets
# ============================================================


STANDARD_PROFILE_SHAPES = [
	(256,),
	(4096,),
	(32000,),
	(151936,),
	(512 * 4096,),
]


# ============================================================
# Crossover characterization
#
# Important:
#
# All shapes are multiples of 256.
#
# This avoids introducing a reduction-tail variable while
# measuring the portable -> multiblock policy crossover.
# ============================================================


CROSSOVER_PROFILE_SHAPES = [
	(4096,),
	(5120,),
	(6144,),
	(7168,),
	(8192,),
	(10240,),
	(12288,),
	(16384,),
	(20480,),
	(24576,),
	(28672,),
	(32000,),
]


# ============================================================
# Reduction characterization
#
# Used for controlled:
#
#     multiblock + tree
#         vs
#     multiblock + CUB
#
# Keep execution policy fixed while changing only the
# block-wide reduction primitive.
# ============================================================


REDUCTION_PROFILE_SHAPES = [
	(6144,),
	(8192,),
	(32000,),
	(151936,),
	(512 * 4096,),
]


CORRECTNESS_SHAPES = [
	(1,),
	(4,),

	(31,),
	(32,),
	(33,),

	(63,),
	(64,),
	(65,),

	(127,),
	(128,),
	(129,),

	(255,),
	(256,),
	(257,),

	# NVIDIA policy boundary.
	(4095,),
	(4096,),
	(4097,),
 
 	# MetaX final AUTO boundary:
	# portable -> multiblock + CUB
	(6143,),
	(6144,),
	(6145,),

	# MetaX measured AUTO policy boundary.
	(8191,),
	(8192,),
	(8193,),

	(32767,),
	(32768,),
	(32769,),

	(32000,),
	(151936,),

	(512 * 4096,),
]

TEST_DTYPES = [
	"f32",
	"f16",
	"bf16",
]


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--device",
		default="cpu",
		choices=[
			"cpu",
			"nvidia",
			"metax",
		],
		type=str,
	)

	parser.add_argument(
		"--profile",
		action="store_true",
	)

	parser.add_argument(
		"--profile-only",
		action="store_true",
		help=(
			"Skip the full correctness suite and "
			"run only the selected profiling set."
		),
	)

	parser.add_argument(
		"--profile-set",
		default="standard",
		choices=[
			"standard",
			"crossover",
			"reduction",
		],
		type=str,
	)

	parser.add_argument(
		"--skip-semantic",
		action="store_true",
	)

	args = parser.parse_args()

	print(
		f"Testing Ops.argmax on "
		f"{args.device}"
	)

	# --------------------------------------------------------
	# Select profile set.
	# --------------------------------------------------------

	if args.profile_set == "crossover":
		profile_shapes = CROSSOVER_PROFILE_SHAPES
	elif args.profile_set == "reduction":
		profile_shapes = REDUCTION_PROFILE_SHAPES
	else:
		profile_shapes = STANDARD_PROFILE_SHAPES

	# ========================================================
	# Performance-only mode
	# ========================================================

	if args.profile_only:
		print()

		print(
			"=== Performance benchmark ==="
		)

		print(
			f"profile-set = "
			f"{args.profile_set}"
		)

		for shape in profile_shapes:
			for dtype_name in TEST_DTYPES:
				test_op_argmax(
					shape,
					dtype_name,
					args.device,
					profile=True,
				)

		print()

		print(
			"\033[92mTest passed!\033[0m"
		)

		sys.exit(0)

	# ========================================================
	# Random correctness
	# ========================================================

	print()

	print(
		"=== Random differential correctness ==="
	)

	for shape in CORRECTNESS_SHAPES:
		for dtype_name in TEST_DTYPES:
			test_op_argmax(
				shape,
				dtype_name,
				args.device,
				profile=False,
			)

	# ========================================================
	# Semantic correctness
	# ========================================================

	if not args.skip_semantic:
		print()

		print(
			"=== Deterministic semantic correctness ==="
		)

		for dtype_name in TEST_DTYPES:
			run_semantic_tests(
				args.device,
				dtype_name,
			)

	# ========================================================
	# Optional profile
	# ========================================================

	if args.profile:
		print()

		print(
			"=== Performance benchmark ==="
		)

		print(
			f"profile-set = "
			f"{args.profile_set}"
		)

		for shape in profile_shapes:
			for dtype_name in TEST_DTYPES:
				test_op_argmax(
					shape,
					dtype_name,
					args.device,
					profile=True,
				)

	print()

	print(
		"\033[92mTest passed!\033[0m"
	)
import os
import sys

project_dir = os.path.abspath(
	os.path.join(
		os.path.dirname(__file__),
		"..",
	)
)

test_dir = os.path.join(
	project_dir,
	"test",
)

sys.path.insert(0, project_dir)
sys.path.insert(0, test_dir)

import llaisys

from test_utils import llaisys_dtype, llaisys_device


def test_invalid_view():
	print("=== Test invalid view ===")

	tensor = llaisys.Tensor(
		(2, 3),
		dtype=llaisys_dtype("i64"),
		device=llaisys_device("cpu"),
	)

	try:
		# Original numel = 6, requested numel = 8.
		tensor.view(2, 4)

	except RuntimeError as error:
		print("Caught RuntimeError:")
		print(error)
		print("Invalid view error propagation passed.")
		return

	raise RuntimeError(
		"Expected invalid view to raise RuntimeError."
	)


def test_invalid_slice():
	print("\n=== Test invalid slice ===")

	tensor = llaisys.Tensor(
		(3, 4, 5),
		dtype=llaisys_dtype("i64"),
		device=llaisys_device("cpu"),
	)

	try:
		# ndim == 3, therefore dim 10 is invalid.
		tensor.slice(
			10,
			0,
			1,
		)

	except RuntimeError as error:
		print("Caught RuntimeError:")
		print(error)
		print("Invalid slice error propagation passed.")
		return

	raise RuntimeError(
		"Expected invalid slice to raise RuntimeError."
	)


def test_invalid_permute():
	print("\n=== Test invalid permute ===")

	tensor = llaisys.Tensor(
		(3, 4, 5),
		dtype=llaisys_dtype("i64"),
		device=llaisys_device("cpu"),
	)

	try:
		# Duplicate dimension 0, so this is not a valid permutation.
		tensor.permute(
			0,
			0,
			2,
		)

	except RuntimeError as error:
		print("Caught RuntimeError:")
		print(error)
		print("Invalid permute error propagation passed.")
		return

	raise RuntimeError(
		"Expected invalid permute to raise RuntimeError."
	)


if __name__ == "__main__":
	print("===== Tensor C API Error Boundary Test =====")

	test_invalid_view()
	test_invalid_slice()
	test_invalid_permute()

	print(
		"\nTensor C API error propagation test passed."
	)
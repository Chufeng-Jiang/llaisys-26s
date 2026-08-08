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

from test_utils import random_tensor



def main():
	print("===== Python C API Error Test =====")

	# Deliberately use incompatible shapes.
	_, a = random_tensor(
		(2, 3),
		"f32",
		"cpu",
	)

	_, b = random_tensor(
		(2, 4),
		"f32",
		"cpu",
	)

	_, out = random_tensor(
		(2, 3),
		"f32",
		"cpu",
	)

	try:
		llaisys.Ops.add(
			out,
			a,
			b,
		)

	except RuntimeError as error:
		print(
			"Caught Python RuntimeError:"
		)

		print(error)

		print(
			"Python C API error propagation test passed."
		)

		return

	raise RuntimeError(
		"Expected LLAISYS Ops.add to fail, "
		"but no exception was raised."
	)


if __name__ == "__main__":
	main()
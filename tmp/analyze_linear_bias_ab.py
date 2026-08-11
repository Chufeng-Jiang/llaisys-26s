import re
import statistics
from collections import defaultdict
from pathlib import Path


LOG_DIR = Path(
	"/data/llaisys-26s/tmp/linear_bias_ab"
)

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

DTYPES = [
	"f32",
	"f16",
	"bf16",
]

PATHS = [
	"nobias",
	"bias",
]


profile_re = re.compile(
	r"PROFILE LinearBias "
	r"path=(nobias|bias) "
	r"M=(\d+) "
	r"N=(\d+) "
	r"K=(\d+) "
	r"dtype=(f32|f16|bf16)"
)

result_re = re.compile(
	r"LLAISYS metax:\s+"
	r"median=([0-9.]+)\s+ms"
)


records = defaultdict(
	list
)


for log_path in sorted(
	LOG_DIR.glob("r*.log")
):

	current = None

	for line in log_path.read_text(
		errors="replace"
	).splitlines():

		profile_match = (
			profile_re.search(
				line
			)
		)

		if profile_match:

			current = (
				profile_match.group(1),
				int(
					profile_match.group(2)
				),
				profile_match.group(5),
			)

			continue

		result_match = (
			result_re.search(
				line
			)
		)

		if (
			result_match
			and current is not None
		):

			records[
				current
			].append(
				float(
					result_match.group(1)
				)
			)

			current = None


expected = (
	len(M_VALUES)
	* len(DTYPES)
	* len(PATHS)
	* 6
)

actual = sum(
	len(values)
	for values in records.values()
)


print(
	"============================================================"
)

print(
	"Linear bias A/B analysis"
)

print(
	"============================================================"
)

print(
	f"records={actual} "
	f"expected={expected}"
)

print()


if actual != expected:
	raise RuntimeError(
		f"Expected {expected} records, "
		f"found {actual}."
	)


summary = {}


for path in PATHS:

	for dtype in DTYPES:

		for m in M_VALUES:

			key = (
				path,
				m,
				dtype,
			)

			values = records[
				key
			]

			if len(values) != 6:
				raise RuntimeError(
					f"{key}: expected 6 "
					f"records, got "
					f"{len(values)}"
				)

			summary[key] = (
				statistics.median(
					values
				)
			)


print(
	f"{'M':>5} "
	f"{'dtype':>6} "
	f"{'NoBias us':>12} "
	f"{'Bias us':>12} "
	f"{'Extra us':>12} "
	f"{'Overhead %':>12} "
	f"{'Bias share %':>13}"
)

print(
	"-" * 83
)


for dtype in DTYPES:

	for m in M_VALUES:

		nobias = summary[
			(
				"nobias",
				m,
				dtype,
			)
		]

		bias = summary[
			(
				"bias",
				m,
				dtype,
			)
		]

		extra = (
			bias
			- nobias
		)

		overhead_pct = (
			extra
			/ nobias
			* 100.0
		)

		bias_share_pct = (
			extra
			/ bias
			* 100.0
		)

		print(
			f"{m:5d} "
			f"{dtype:>6} "
			f"{nobias * 1000:12.3f} "
			f"{bias * 1000:12.3f} "
			f"{extra * 1000:12.3f} "
			f"{overhead_pct:12.2f} "
			f"{bias_share_pct:13.2f}"
		)

	print()


print(
	"============================================================"
)

print(
	"Per-round pair deltas"
)

print(
	"============================================================"
)


for dtype in DTYPES:

	print()
	print(
		f"[{dtype}]"
	)

	for m in M_VALUES:

		nobias_values = records[
			(
				"nobias",
				m,
				dtype,
			)
		]

		bias_values = records[
			(
				"bias",
				m,
				dtype,
			)
		]

		deltas_us = [
			(
				bias
				- nobias
			)
			* 1000
			for (
				nobias,
				bias,
			) in zip(
				nobias_values,
				bias_values,
			)
		]

		print(
			f"M={m:3d}: "
			+ ", ".join(
				f"{x:.3f}"
				for x in deltas_us
			)
			+ " us"
		)

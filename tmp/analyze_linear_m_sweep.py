import re
import statistics
from collections import defaultdict
from pathlib import Path


LOG_DIR = Path(
	"/data/llaisys-26s/tmp/linear_m_sweep"
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


profile_re = re.compile(
	r"PROFILE Linear "
	r"M=(\d+) "
	r"N=(\d+) "
	r"K=(\d+) "
	r"dtype=(f32|f16|bf16) "
	r"bias=False"
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
			current = {
				"m": int(
					profile_match.group(1)
				),
				"n": int(
					profile_match.group(2)
				),
				"k": int(
					profile_match.group(3)
				),
				"dtype": (
					profile_match.group(4)
				),
			}

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
			median_ms = float(
				result_match.group(1)
			)

			key = (
				current["m"],
				current["dtype"],
			)

			records[key].append(
				median_ms
			)

			current = None


expected_records = (
	len(M_VALUES)
	* len(DTYPES)
	* 6
)

actual_records = sum(
	len(values)
	for values in records.values()
)


print(
	"============================================================"
)

print(
	"Linear M-sweep analysis"
)

print(
	"============================================================"
)

print(
	f"records={actual_records} "
	f"expected={expected_records}"
)

print()


if actual_records != expected_records:
	raise RuntimeError(
		f"Expected {expected_records} records, "
		f"found {actual_records}."
	)


summary = {}


for dtype in DTYPES:
	for m in M_VALUES:

		key = (
			m,
			dtype,
		)

		values_ms = (
			records[
				key
			]
		)

		if len(values_ms) != 6:
			raise RuntimeError(
				f"M={m}, dtype={dtype}: "
				f"expected 6 rounds, "
				f"found {len(values_ms)}"
			)

		mom_ms = statistics.median(
			values_ms
		)

		summary[key] = mom_ms


print(
	"Median-of-medians"
)

print()

print(
	f"{'M':>5} "
	f"{'dtype':>6} "
	f"{'MoM us':>12} "
	f"{'min us':>12} "
	f"{'max us':>12} "
	f"{'TFLOPS':>12} "
	f"{'lat/M1':>10} "
	f"{'TP/M1':>10}"
)

print(
	"-" * 86
)


for dtype in DTYPES:

	m1_ms = summary[
		(
			1,
			dtype,
		)
	]

	for m in M_VALUES:

		values_ms = records[
			(
				m,
				dtype,
			)
		]

		mom_ms = summary[
			(
				m,
				dtype,
			)
		]

		# GEMM:
		#
		#     [M,K] x [K,N]
		#
		# FLOPs = 2*M*N*K
		#
		# latency_ms -> seconds:
		#     latency_ms * 1e-3
		#
		# TFLOPS:
		#     FLOPs / seconds / 1e12
		#
		tflops = (
			2.0
			* m
			* 4096
			* 4096
			/ (
				mom_ms
				* 1e9
			)
		)

		latency_ratio = (
			mom_ms
			/ m1_ms
		)

		# Rows processed per millisecond.
		throughput = (
			m
			/ mom_ms
		)

		m1_throughput = (
			1
			/ m1_ms
		)

		throughput_ratio = (
			throughput
			/ m1_throughput
		)

		print(
			f"{m:5d} "
			f"{dtype:>6} "
			f"{mom_ms * 1000:12.3f} "
			f"{min(values_ms) * 1000:12.3f} "
			f"{max(values_ms) * 1000:12.3f} "
			f"{tflops:12.3f} "
			f"{latency_ratio:10.3f} "
			f"{throughput_ratio:10.3f}"
		)

	print()


print(
	"============================================================"
)

print(
	"Per-round medians"
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

		values_us = [
			value * 1000
			for value in records[
				(
					m,
					dtype,
				)
			]
		]

		formatted = ", ".join(
			f"{value:.3f}"
			for value in values_us
		)

		print(
			f"M={m:3d}: "
			f"{formatted} us"
		)

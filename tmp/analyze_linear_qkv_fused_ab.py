import re
import statistics
from collections import defaultdict
from pathlib import Path


LOG_DIR = Path(
	"/data/llaisys-26s/tmp/linear_qkv_fused_ab"
)

SHAPES = [
	"q_proj",
	"kv_proj",
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

DTYPES = [
	"f32",
	"f16",
	"bf16",
]

IMPLS = [
	"portable",
	"fused",
]


profile_re = re.compile(
	r"PROFILE QKVLinearFusedAB "
	r"impl=(portable|fused) "
	r"shape=(q_proj|kv_proj) "
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


for round_id in range(
	1,
	7,
):

	for impl in IMPLS:

		log_path = (
			LOG_DIR
			/ f"r{round_id}_{impl}.log"
		)

		if not log_path.exists():
			raise RuntimeError(
				f"Missing log: "
					f"{log_path}"
			)

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
					profile_match.group(2),
					int(
						profile_match.group(3)
					),
					profile_match.group(6),
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
	len(SHAPES)
	* len(M_VALUES)
	* len(DTYPES)
	* len(IMPLS)
	* 6
)

actual = sum(
	len(values)
	for values in records.values()
)


print(
	"# Qwen2 Q/K/V Linear portable-vs-fused analysis"
)

print()

print(
	f"records={actual} expected={expected}"
)

print()


if actual != expected:
	raise RuntimeError(
		f"Expected {expected}, "
		f"found {actual}."
	)


summary = {}


for impl in IMPLS:

	for shape_name in SHAPES:

		for dtype_name in DTYPES:

			for m in M_VALUES:

				key = (
					impl,
					shape_name,
					m,
					dtype_name,
				)

				values = records[
					key
				]

				if len(values) != 6:
					raise RuntimeError(
						f"{key}: expected 6, "
						f"found {len(values)}."
					)

				summary[
					key
				] = statistics.median(
					values
				)


print(
	"Median-of-medians"
)

print()

print(
	f"{'shape':>8} "
	f"{'M':>5} "
	f"{'dtype':>6} "
	f"{'Portable us':>13} "
	f"{'Fused us':>13} "
	f"{'Saved us':>11} "
	f"{'Speedup':>10} "
	f"{'Reduction %':>12} "
	f"{'Winner':>10}"
)

print(
	"-" * 101
)


for shape_name in SHAPES:

	for dtype_name in DTYPES:

		for m in M_VALUES:

			portable = summary[
				(
					"portable",
					shape_name,
					m,
					dtype_name,
				)
			]

			fused = summary[
				(
					"fused",
					shape_name,
					m,
					dtype_name,
				)
			]

			saved = (
				portable
				- fused
			)

			speedup = (
				portable
				/ fused
			)

			reduction = (
				saved
				/ portable
				* 100.0
			)

			winner = (
				"fused"
				if fused < portable
				else "portable"
			)

			print(
				f"{shape_name:>8} "
				f"{m:5d} "
				f"{dtype_name:>6} "
				f"{portable * 1000:13.3f} "
				f"{fused * 1000:13.3f} "
				f"{saved * 1000:11.3f} "
				f"{speedup:10.3f} "
				f"{reduction:12.2f} "
				f"{winner:>10}"
			)

		print()

	print()


print()
print(
	"Paired-median policy summary"
)

print()

print(
	f"{'shape':>8} "
	f"{'M':>5} "
	f"{'dtype':>6} "
	f"{'median saved us':>16} "
	f"{'median speedup':>16} "
	f"{'winner':>10}"
)

print(
	"-" * 75
)


for shape_name in SHAPES:

	for dtype_name in DTYPES:

		for m in M_VALUES:

			portable_values = records[
				(
					"portable",
					shape_name,
					m,
					dtype_name,
				)
			]

			fused_values = records[
				(
					"fused",
					shape_name,
					m,
					dtype_name,
				)
			]

			deltas = [
				portable
				- fused
				for (
					portable,
					fused,
				) in zip(
					portable_values,
					fused_values,
				)
			]

			speedups = [
				portable
				/ fused
				for (
					portable,
					fused,
				) in zip(
					portable_values,
					fused_values,
				)
			]

			median_saved = (
				statistics.median(
					deltas
				)
			)

			median_speedup = (
				statistics.median(
					speedups
				)
			)

			winner = (
				"fused"
				if median_saved > 0
				else "portable"
			)

			print(
				f"{shape_name:>8} "
				f"{m:5d} "
				f"{dtype_name:>6} "
				f"{median_saved * 1000:16.3f} "
				f"{median_speedup:16.3f} "
				f"{winner:>10}"
			)

		print()

	print()
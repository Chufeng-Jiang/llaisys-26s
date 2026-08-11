import re
import statistics
from collections import defaultdict
from pathlib import Path


LOG_DIR = Path(
	"/data/llaisys-26s/tmp/linear_bias_components"
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
	"broadcast_beta0",
	"normal",
]


profile_re = re.compile(
	r"PROFILE LinearBiasComponent "
	r"path=(nobias|broadcast_beta0|normal) "
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


# ============================================================
# Parse all logs.
# ============================================================

for round_id in range(
	1,
	7,
):

	for path_name in PATHS:

		log_path = (
			LOG_DIR
			/ f"r{round_id}_{path_name}.log"
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
				median_ms = float(
					result_match.group(1)
				)

				records[
					current
				].append(
					median_ms
				)

				current = None


expected_records = (
	len(M_VALUES)
	* len(DTYPES)
	* len(PATHS)
	* 6
)

actual_records = sum(
	len(values)
	for values in records.values()
)


print(
	"# Linear bias-component A/B/C analysis"
)

print()

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


# ============================================================
# Median-of-medians.
# ============================================================

summary = {}


for path_name in PATHS:

	for dtype_name in DTYPES:

		for m in M_VALUES:

			key = (
				path_name,
				m,
				dtype_name,
			)

			values = records[
				key
			]

			if len(values) != 6:
				raise RuntimeError(
					f"{key}: expected 6 rounds, "
					f"found {len(values)}."
				)

			summary[
				key
			] = statistics.median(
				values
			)


# ============================================================
# Main decomposition table.
#
# A = nobias
# B = broadcast_beta0
# C = normal
#
# B - A:
#     approximate incremental shared bias-broadcast path cost
#
# C - B:
#     approximate beta=1 vs beta=0 GEMM-path contribution
#
# C - A:
#     total observed bias-path overhead
# ============================================================

print(
	"Median-of-medians"
)

print()

print(
	f"{'M':>5} "
	f"{'dtype':>6} "
	f"{'A nobias us':>13} "
	f"{'B beta0 us':>13} "
	f"{'C normal us':>13} "
	f"{'B-A us':>10} "
	f"{'C-B us':>10} "
	f"{'C-A us':>10} "
	f"{'Broadcast %':>12} "
	f"{'Beta %':>9}"
)

print(
	"-" * 116
)


for dtype_name in DTYPES:

	for m in M_VALUES:

		a = summary[
			(
				"nobias",
				m,
				dtype_name,
			)
		]

		b = summary[
			(
				"broadcast_beta0",
				m,
				dtype_name,
			)
		]

		c = summary[
			(
				"normal",
				m,
				dtype_name,
			)
		]

		broadcast_delta = (
			b - a
		)

		beta_delta = (
			c - b
		)

		total_delta = (
			c - a
		)

		if total_delta != 0.0:
			broadcast_share = (
				broadcast_delta
				/ total_delta
				* 100.0
			)

			beta_share = (
				beta_delta
				/ total_delta
				* 100.0
			)
		else:
			broadcast_share = float(
				"nan"
			)

			beta_share = float(
				"nan"
			)

		print(
			f"{m:5d} "
			f"{dtype_name:>6} "
			f"{a * 1000:13.3f} "
			f"{b * 1000:13.3f} "
			f"{c * 1000:13.3f} "
			f"{broadcast_delta * 1000:10.3f} "
			f"{beta_delta * 1000:10.3f} "
			f"{total_delta * 1000:10.3f} "
			f"{broadcast_share:12.2f} "
			f"{beta_share:9.2f}"
		)

	print()


# ============================================================
# Paired round deltas.
#
# Because every key has one value per round and logs were read
# round-by-round, zip() aligns the same round across A/B/C.
# ============================================================

print()
print(
	"Per-round paired deltas"
)

print()


for dtype_name in DTYPES:

	print(
		f"[{dtype_name}]"
	)

	for m in M_VALUES:

		a_values = records[
			(
				"nobias",
				m,
				dtype_name,
			)
		]

		b_values = records[
			(
				"broadcast_beta0",
				m,
				dtype_name,
			)
		]

		c_values = records[
			(
				"normal",
				m,
				dtype_name,
			)
		]

		broadcast_deltas_us = [
			(
				b - a
			)
			* 1000
			for (
				a,
				b,
			) in zip(
				a_values,
				b_values,
			)
		]

		beta_deltas_us = [
			(
				c - b
			)
			* 1000
			for (
				b,
				c,
			) in zip(
				b_values,
				c_values,
			)
		]

		total_deltas_us = [
			(
				c - a
			)
			* 1000
			for (
				a,
				c,
			) in zip(
				a_values,
				c_values,
			)
		]

		print(
			f"M={m:3d}"
		)

		print(
			"  B-A broadcast: "
			+ ", ".join(
				f"{value:.3f}"
				for value in broadcast_deltas_us
			)
			+ " us"
		)

		print(
			"  C-B beta:      "
			+ ", ".join(
				f"{value:.3f}"
				for value in beta_deltas_us
			)
			+ " us"
		)

		print(
			"  C-A total:     "
			+ ", ".join(
				f"{value:.3f}"
				for value in total_deltas_us
			)
			+ " us"
		)

	print()


# ============================================================
# Compact paired-median summary.
# ============================================================

print()
print(
	"Paired-median decomposition"
)

print()

print(
	f"{'M':>5} "
	f"{'dtype':>6} "
	f"{'median B-A us':>15} "
	f"{'median C-B us':>15} "
	f"{'median C-A us':>15}"
)

print(
	"-" * 66
)


for dtype_name in DTYPES:

	for m in M_VALUES:

		a_values = records[
			(
				"nobias",
				m,
				dtype_name,
			)
		]

		b_values = records[
			(
				"broadcast_beta0",
				m,
				dtype_name,
			)
		]

		c_values = records[
			(
				"normal",
				m,
				dtype_name,
			)
		]

		broadcast_deltas = [
			b - a
			for (
				a,
				b,
			) in zip(
				a_values,
				b_values,
			)
		]

		beta_deltas = [
			c - b
			for (
				b,
				c,
			) in zip(
				b_values,
				c_values,
			)
		]

		total_deltas = [
			c - a
			for (
				a,
				c,
			) in zip(
				a_values,
				c_values,
			)
		]

		print(
			f"{m:5d} "
			f"{dtype_name:>6} "
			f"{statistics.median(broadcast_deltas) * 1000:15.3f} "
			f"{statistics.median(beta_deltas) * 1000:15.3f} "
			f"{statistics.median(total_deltas) * 1000:15.3f}"
		)

	print()
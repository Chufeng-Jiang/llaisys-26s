#!/usr/bin/env bash

set -u
set -o pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/linear_bias_components"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"


run_path() {
	local round="$1"
	local path_name="$2"
	local order="$3"
	local dtype_order="$4"

	local log="${LOG_DIR}/r${round}_${path_name}.log"

	echo
	echo "============================================================"
	echo "Linear bias components"
	echo "round=${round}"
	echo "path=${path_name}"
	echo "M_order=${order}"
	echo "dtype_order=${dtype_order}"
	echo "============================================================"

	if ! python \
		/data/llaisys-26s/tmp/profile_linear_bias_components.py \
		--device metax \
		--path "${path_name}" \
		--order "${order}" \
		--dtype-order "${dtype_order}" \
		> "${log}" 2>&1
	then
		echo
		echo "FAILED:"
		echo "${log}"
		echo
		cat "${log}"
		exit 1
	fi

	echo "PASS: round=${round} path=${path_name}"

	grep -E \
		"PROFILE LinearBiasComponent|LLAISYS metax:" \
		"${log}" \
		|| true
}


# ============================================================
# Six balanced rounds.
#
# The path order changes each round.
# The M direction alternates.
# The dtype order rotates.
# ============================================================

# Round 1
run_path 1 nobias          asc  f32,f16,bf16
run_path 1 broadcast_beta0 asc  f32,f16,bf16
run_path 1 normal          asc  f32,f16,bf16

# Round 2
run_path 2 normal          desc bf16,f16,f32
run_path 2 broadcast_beta0 desc bf16,f16,f32
run_path 2 nobias          desc bf16,f16,f32

# Round 3
run_path 3 broadcast_beta0 asc  f16,bf16,f32
run_path 3 normal          asc  f16,bf16,f32
run_path 3 nobias          asc  f16,bf16,f32

# Round 4
run_path 4 nobias          desc f32,bf16,f16
run_path 4 normal          desc f32,bf16,f16
run_path 4 broadcast_beta0 desc f32,bf16,f16

# Round 5
run_path 5 normal          asc  bf16,f32,f16
run_path 5 nobias          asc  bf16,f32,f16
run_path 5 broadcast_beta0 asc  bf16,f32,f16

# Round 6
run_path 6 broadcast_beta0 desc f16,f32,bf16
run_path 6 nobias          desc f16,f32,bf16
run_path 6 normal          desc f16,f32,bf16


echo
echo "============================================================"
echo "Linear bias component A/B/C COMPLETE"
echo "============================================================"

echo
echo "Expected profile records:"
echo "10 M x 3 dtype x 3 paths x 6 rounds = 540"

echo
echo "Logs:"
echo "${LOG_DIR}"
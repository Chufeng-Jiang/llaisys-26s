#!/usr/bin/env bash

set -u
set -o pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/linear_bias_ab"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"


run_round() {
	local round="$1"
	local order="$2"
	local pair_order="$3"
	local dtype_order="$4"

	local log="${LOG_DIR}/r${round}.log"

	echo
	echo "============================================================"
	echo "Linear bias A/B"
	echo "round=${round}"
	echo "M_order=${order}"
	echo "pair_order=${pair_order}"
	echo "dtype_order=${dtype_order}"
	echo "============================================================"

	if ! python \
		/data/llaisys-26s/tmp/profile_linear_bias_ab.py \
		--device metax \
		--order "${order}" \
		--pair-order "${pair_order}" \
		--dtype-order "${dtype_order}" \
		> "${log}" 2>&1
	then
		echo
		echo "FAILED: ${log}"
		cat "${log}"
		exit 1
	fi

	echo "PASS: round=${round}"

	grep -E \
		"PROFILE LinearBias|LLAISYS metax:" \
		"${log}" \
		|| true
}


run_round \
	1 \
	asc \
	nobias-first \
	f32,f16,bf16

run_round \
	2 \
	desc \
	bias-first \
	bf16,f16,f32

run_round \
	3 \
	asc \
	bias-first \
	f16,bf16,f32

run_round \
	4 \
	desc \
	nobias-first \
	f32,bf16,f16

run_round \
	5 \
	asc \
	nobias-first \
	bf16,f32,f16

run_round \
	6 \
	desc \
	bias-first \
	f16,f32,bf16


echo
echo "============================================================"
echo "Linear bias A/B COMPLETE"
echo "============================================================"
echo
echo "${LOG_DIR}"

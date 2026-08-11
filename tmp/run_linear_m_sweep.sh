#!/usr/bin/env bash

set -u
set -o pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/linear_m_sweep"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"


run_round() {
	local round="$1"
	local order="$2"
	local dtype_order="$3"

	local log="${LOG_DIR}/r${round}.log"

	echo
	echo "============================================================"
	echo "Linear M-sweep"
	echo "round=${round}"
	echo "order=${order}"
	echo "dtype_order=${dtype_order}"
	echo "============================================================"

	if ! python \
		/data/llaisys-26s/tmp/profile_linear_m_sweep.py \
		--device metax \
		--order "${order}" \
		--dtype-order "${dtype_order}" \
		> "${log}" 2>&1
	then
		echo
		echo "ERROR:"
		echo "round=${round}"
		echo "log=${log}"
		echo
		cat "${log}"
		exit 1
	fi

	echo "PASS: round=${round}"

	grep -E \
		"PROFILE Linear|LLAISYS metax:" \
		"${log}" \
		|| true
}


# ============================================================
# Six balanced rounds
# ============================================================

run_round \
	1 \
	asc \
	f32,f16,bf16

run_round \
	2 \
	desc \
	bf16,f16,f32

run_round \
	3 \
	asc \
	f16,bf16,f32

run_round \
	4 \
	desc \
	f32,bf16,f16

run_round \
	5 \
	asc \
	bf16,f32,f16

run_round \
	6 \
	desc \
	f16,f32,bf16


echo
echo "============================================================"
echo "All Linear M-sweep rounds completed"
echo "============================================================"

echo
echo "Logs:"
echo "${LOG_DIR}"

#!/usr/bin/env bash

set -u
set -o pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/linear_fused_ab"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"


run_impl() {
	local round="$1"
	local impl="$2"
	local order="$3"
	local dtype_order="$4"

	local log="${LOG_DIR}/r${round}_${impl}.log"

	echo
	echo "============================================================"
	echo "Linear portable-vs-fused"
	echo "round=${round}"
	echo "impl=${impl}"
	echo "M_order=${order}"
	echo "dtype_order=${dtype_order}"
	echo "============================================================"

	if ! python \
		/data/llaisys-26s/tmp/profile_linear_fused_ab.py \
		--device metax \
		--impl "${impl}" \
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

	echo "PASS: round=${round} impl=${impl}"

	grep -E \
		"PROFILE LinearFusedAB|LLAISYS metax:" \
		"${log}" \
		|| true
}


# ============================================================
# Six balanced A/B rounds.
# ============================================================

run_impl 1 portable asc  f32,f16,bf16
run_impl 1 fused    asc  f32,f16,bf16

run_impl 2 fused    desc bf16,f16,f32
run_impl 2 portable desc bf16,f16,f32

run_impl 3 portable asc  f16,bf16,f32
run_impl 3 fused    asc  f16,bf16,f32

run_impl 4 fused    desc f32,bf16,f16
run_impl 4 portable desc f32,bf16,f16

run_impl 5 fused    asc  bf16,f32,f16
run_impl 5 portable asc  bf16,f32,f16

run_impl 6 portable desc f16,f32,bf16
run_impl 6 fused    desc f16,f32,bf16


echo
echo "============================================================"
echo "Linear portable-vs-fused A/B COMPLETE"
echo "============================================================"

echo
echo "Expected profile records:"
echo "10 M x 3 dtype x 2 impl x 6 rounds = 360"

echo
echo "Logs:"
echo "${LOG_DIR}"
#!/usr/bin/env bash

set -euo pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/rmsnorm_block_ab"

mkdir -p "${LOG_DIR}"

run_profile() {
	local round="$1"
	local block="$2"

	echo "Round ${round}: block=${block}"

	LLAISYS_METAX_RMS_NORM_BLOCK_SIZE="${block}" \
	python test/ops/rms_norm.py \
		--device metax \
		--profile \
		> "${LOG_DIR}/r${round}_b${block}.log" 2>&1
}

for r in 1 2 3 4 5 6
do
	echo
	echo "===== Round ${r} ====="

	case "${r}" in
		1)
			order=(64 128 256)
			;;
		2)
			order=(64 256 128)
			;;
		3)
			order=(128 64 256)
			;;
		4)
			order=(128 256 64)
			;;
		5)
			order=(256 64 128)
			;;
		6)
			order=(256 128 64)
			;;
	esac

	echo "Order: ${order[*]}"

	for block in "${order[@]}"
	do
		run_profile \
			"${r}" \
			"${block}"
	done
done

echo
echo "===== All rounds finished ====="
echo "Logs: ${LOG_DIR}"

echo
echo "===== Median results ====="

grep -H \
	"median=" \
	"${LOG_DIR}"/*.log \
	|| true

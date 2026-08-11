#!/usr/bin/env bash

set -euo pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/rmsnorm_cub_packed_ab"

mkdir -p "${LOG_DIR}"

run_packed() {
	local round="$1"

	echo "Round ${round}: CUB + PACKED128"

	LLAISYS_METAX_RMS_NORM_BLOCK_SIZE=256 \
	LLAISYS_METAX_RMS_NORM_REDUCTION=cub \
	python test/ops/rms_norm.py \
		--device metax \
		--profile \
		> "${LOG_DIR}/r${round}_cub_packed.log" 2>&1
}

run_scalar() {
	local round="$1"

	echo "Round ${round}: CUB + SCALAR"

	LLAISYS_METAX_RMS_NORM_FORCE_SCALAR=1 \
	LLAISYS_METAX_RMS_NORM_BLOCK_SIZE=256 \
	LLAISYS_METAX_RMS_NORM_REDUCTION=cub \
	python test/ops/rms_norm.py \
		--device metax \
		--profile \
		> "${LOG_DIR}/r${round}_cub_scalar.log" 2>&1
}

for r in 1 2 3 4 5 6
do
	echo
	echo "===== Round ${r} ====="

	if (( r % 2 == 1 )); then
		echo "Order: PACKED -> SCALAR"

		run_packed "${r}"
		run_scalar "${r}"
	else
		echo "Order: SCALAR -> PACKED"

		run_scalar "${r}"
		run_packed "${r}"
	fi
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

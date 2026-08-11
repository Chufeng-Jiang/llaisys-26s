#!/usr/bin/env bash

set -euo pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/rmsnorm_tree_cub_ab"

mkdir -p "${LOG_DIR}"

run_tree() {
	local round="$1"

	echo "Round ${round}: TREE"

	LLAISYS_METAX_RMS_NORM_FORCE_SCALAR=1 \
	LLAISYS_METAX_RMS_NORM_BLOCK_SIZE=256 \
	LLAISYS_METAX_RMS_NORM_REDUCTION=tree \
	python test/ops/rms_norm.py \
		--device metax \
		--profile \
		> "${LOG_DIR}/r${round}_tree.log" 2>&1
}

run_cub() {
	local round="$1"

	echo "Round ${round}: CUB"

	LLAISYS_METAX_RMS_NORM_FORCE_SCALAR=1 \
	LLAISYS_METAX_RMS_NORM_BLOCK_SIZE=256 \
	LLAISYS_METAX_RMS_NORM_REDUCTION=cub \
	python test/ops/rms_norm.py \
		--device metax \
		--profile \
		> "${LOG_DIR}/r${round}_cub.log" 2>&1
}

for r in 1 2 3 4 5 6
do
	echo
	echo "===== Round ${r} ====="

	if (( r % 2 == 1 )); then
		echo "Order: TREE -> CUB"

		run_tree "${r}"
		run_cub "${r}"
	else
		echo "Order: CUB -> TREE"

		run_cub "${r}"
		run_tree "${r}"
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

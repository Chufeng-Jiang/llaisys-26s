#!/usr/bin/env bash

set -euo pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/rmsnorm_treepacked_cubpacked_ab"

mkdir -p "${LOG_DIR}"

# ============================================================
# TREE + Packed128
# ============================================================
#
# No FORCE_SCALAR:
#
#     eligible shape
#         -> Packed128
#
#     non-eligible shape
#         -> Scalar
#
# Fixed:
#
#     block = 256
#
# ============================================================

run_tree_packed() {
	local round="$1"

	echo "Round ${round}: TREE + PACKED128"

	LLAISYS_METAX_RMS_NORM_BLOCK_SIZE=256 \
	LLAISYS_METAX_RMS_NORM_REDUCTION=tree \
	python test/ops/rms_norm.py \
		--device metax \
		--profile \
		> "${LOG_DIR}/r${round}_tree_packed.log" 2>&1
}

# ============================================================
# CUB + Packed128
# ============================================================

run_cub_packed() {
	local round="$1"

	echo "Round ${round}: CUB + PACKED128"

	LLAISYS_METAX_RMS_NORM_BLOCK_SIZE=256 \
	LLAISYS_METAX_RMS_NORM_REDUCTION=cub \
	python test/ops/rms_norm.py \
		--device metax \
		--profile \
		> "${LOG_DIR}/r${round}_cub_packed.log" 2>&1
}

# ============================================================
# Six balanced rounds
# ============================================================
#
# Odd rounds:
#
#     TREE -> CUB
#
# Even rounds:
#
#     CUB -> TREE
#
# Each treatment appears:
#
#     first  3 times
#     second 3 times
#
# ============================================================

for r in 1 2 3 4 5 6
do
	echo
	echo "============================================================"
	echo "Round ${r}"
	echo "============================================================"

	if (( r % 2 == 1 )); then
		echo "Order: TREE+PACKED -> CUB+PACKED"

		run_tree_packed "${r}"
		run_cub_packed "${r}"
	else
		echo "Order: CUB+PACKED -> TREE+PACKED"

		run_cub_packed "${r}"
		run_tree_packed "${r}"
	fi
done

# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "All rounds finished"
echo "============================================================"

echo "Logs:"
echo "${LOG_DIR}"

echo
echo "============================================================"
echo "All median results"
echo "============================================================"

grep -H \
	"median=" \
	"${LOG_DIR}"/*.log \
	|| true

echo
echo "============================================================"
echo "(1, 4096) decode-shaped workload"
echo "============================================================"

grep -H \
	'RMSNorm shape=(1, 4096)' \
	"${LOG_DIR}"/*.log \
	|| true

echo
echo "============================================================"
echo "(512, 4095) negative control"
echo "============================================================"

grep -H \
	'RMSNorm shape=(512, 4095)' \
	"${LOG_DIR}"/*.log \
	|| true

echo
echo "============================================================"
echo "(512, 4096) large Packed128 workload"
echo "============================================================"

grep -H \
	'RMSNorm shape=(512, 4096)' \
	"${LOG_DIR}"/*.log \
	|| true

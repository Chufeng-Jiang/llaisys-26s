#!/usr/bin/env bash

set -euo pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/argmax_block_ab"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"

run_block() {
	local round="$1"
	local block="$2"

	echo "Round ${round}: block=${block}"

	env \
		-u LLAISYS_METAX_ARGMAX_DEBUG \
		LLAISYS_METAX_ARGMAX_BLOCK_SIZE="${block}" \
		python test/ops/argmax.py \
			--device metax \
			--profile \
			--profile-only \
			> "${LOG_DIR}/r${round}_b${block}.log" 2>&1
}

run_round() {
	local round="$1"
	shift

	echo
	echo "============================================================"
	echo "Round ${round}: $*"
	echo "============================================================"

	for block in "$@"
	do
		run_block "${round}" "${block}"
	done
}

run_round 1 64 128 256
run_round 2 64 256 128
run_round 3 128 64 256
run_round 4 128 256 64
run_round 5 256 64 128
run_round 6 256 128 64

echo
echo "============================================================"
echo "All Argmax block-size rounds finished"
echo "============================================================"
echo "Logs: ${LOG_DIR}"

echo
echo "===== Profile results ====="

grep -H \
	"Argmax shape=" \
	"${LOG_DIR}"/*.log \
	|| true

#!/usr/bin/env bash

set -u
set -o pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/argmax_cub_shuffle_ab"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"

run_one() {
	local round="$1"
	local reduction="$2"

	local log="${LOG_DIR}/r${round}_${reduction}.log"

	echo
	echo "============================================================"
	echo "Round ${round}: reduction=${reduction}"
	echo "============================================================"

	if ! env \
		-u LLAISYS_METAX_ARGMAX_DEBUG \
		LLAISYS_METAX_ARGMAX_IMPL=multiblock \
		LLAISYS_METAX_ARGMAX_REDUCTION="${reduction}" \
		LLAISYS_METAX_ARGMAX_BLOCK_SIZE=256 \
		LLAISYS_METAX_ARGMAX_MAX_BLOCKS=256 \
		python test/ops/argmax.py \
			--device metax \
			--profile-only \
			--profile-set reduction \
			> "${log}" 2>&1
	then
		echo
		echo "ERROR:"
		echo "round=${round}"
		echo "reduction=${reduction}"
		echo "log=${log}"

		cat "${log}"
		exit 1
	fi

	echo "PASS: round=${round}, reduction=${reduction}"

	grep \
		"Argmax shape=" \
		"${log}" \
		|| true
}

run_one 1 cub
run_one 1 shuffle

run_one 2 shuffle
run_one 2 cub

run_one 3 cub
run_one 3 shuffle

run_one 4 shuffle
run_one 4 cub

run_one 5 cub
run_one 5 shuffle

run_one 6 shuffle
run_one 6 cub

echo
echo "============================================================"
echo "All Argmax CUB vs Shuffle64 rounds completed"
echo "============================================================"

echo
echo "Logs:"
echo "${LOG_DIR}"

echo
echo "============================================================"
echo "All results"
echo "============================================================"

grep -H \
	"Argmax shape=" \
	"${LOG_DIR}"/*.log \
	|| true

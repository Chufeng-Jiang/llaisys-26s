#!/usr/bin/env bash

set -u
set -o pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/argmax_tree_cub_ab"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"


run_reduction() {
	local round="$1"
	local reduction="$2"

	local log="${LOG_DIR}/r${round}_${reduction}.log"

	echo
	echo "------------------------------------------------------------"
	echo "Round ${round}: reduction=${reduction}"
	echo "------------------------------------------------------------"

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
		echo "============================================================"
		echo "ERROR"
		echo "round=${round}"
		echo "reduction=${reduction}"
		echo "log=${log}"
		echo "============================================================"

		cat "${log}"

		exit 1
	fi

	echo "PASS: round=${round}, reduction=${reduction}"

	grep \
		"Argmax shape=" \
		"${log}" \
		|| true
}


run_round() {
	local round="$1"
	local first="$2"
	local second="$3"

	echo
	echo "============================================================"
	echo "ROUND ${round}: ${first} -> ${second}"
	echo "============================================================"

	run_reduction \
		"${round}" \
		"${first}"

	run_reduction \
		"${round}" \
		"${second}"
}


# ============================================================
# Six balanced/interleaved rounds
# ============================================================

run_round 1 tree cub
run_round 2 cub tree
run_round 3 tree cub
run_round 4 cub tree
run_round 5 tree cub
run_round 6 cub tree


echo
echo "============================================================"
echo "All Argmax Tree/CUB rounds completed"
echo "============================================================"

echo
echo "Logs:"
echo "${LOG_DIR}"

echo
echo "============================================================"
echo "All profile results"
echo "============================================================"

grep -H \
	"Argmax shape=" \
	"${LOG_DIR}"/*.log \
	|| true

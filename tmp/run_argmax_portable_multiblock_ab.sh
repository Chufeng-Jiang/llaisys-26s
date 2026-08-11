#!/usr/bin/env bash

set -u
set -o pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/argmax_portable_multiblock_ab"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"


run_impl() {
	local round="$1"
	local impl="$2"

	local log="${LOG_DIR}/r${round}_${impl}.log"

	echo
	echo "Round ${round}: ${impl}"

	if ! env \
		-u LLAISYS_METAX_ARGMAX_DEBUG \
		LLAISYS_METAX_ARGMAX_IMPL="${impl}" \
		LLAISYS_METAX_ARGMAX_BLOCK_SIZE=256 \
		LLAISYS_METAX_ARGMAX_MAX_BLOCKS=256 \
		python test/ops/argmax.py \
			--device metax \
			--profile-only \
			> "${log}" 2>&1
	then
		echo
		echo "============================================================"
		echo "ERROR: round=${round}, impl=${impl}"
		echo "============================================================"

		cat "${log}"

		exit 1
	fi

	echo "PASS: round=${round}, impl=${impl}"

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
	echo "Round ${round}: ${first} -> ${second}"
	echo "============================================================"

	run_impl "${round}" "${first}"
	run_impl "${round}" "${second}"
}


# Balanced execution order.

run_round 1 portable multiblock
run_round 2 multiblock portable
run_round 3 portable multiblock
run_round 4 multiblock portable
run_round 5 portable multiblock
run_round 6 multiblock portable


echo
echo "============================================================"
echo "All Argmax portable/multiblock rounds finished"
echo "============================================================"

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

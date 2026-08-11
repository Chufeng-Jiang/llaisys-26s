#!/usr/bin/env bash

set -u
set -o pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/argmax_final_ab"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"

run_one() {
	local round="$1"
	local name="$2"
	local reduction="$3"
	local merge="$4"

	local log="${LOG_DIR}/r${round}_${name}.log"

	echo
	echo "============================================================"
	echo "round=${round}"
	echo "path=${name}"
	echo "reduction=${reduction}"
	echo "merge=${merge}"
	echo "============================================================"

	if ! env \
		-u LLAISYS_METAX_ARGMAX_DEBUG \
		LLAISYS_METAX_ARGMAX_IMPL=multiblock \
		LLAISYS_METAX_ARGMAX_REDUCTION="${reduction}" \
		LLAISYS_METAX_ARGMAX_MERGE="${merge}" \
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
		echo "path=${name}"
		echo "log=${log}"

		cat "${log}"
		exit 1
	fi

	echo "PASS: round=${round}, path=${name}"

	grep \
		"Argmax shape=" \
		"${log}" \
		|| true
}


# ============================================================
# Balanced six-round order
# ============================================================

# Round 1
run_one 1 cub_stage2      cub     stage2
run_one 1 shuffle_stage2  shuffle stage2
run_one 1 shuffle_atomic  shuffle atomic

# Round 2
run_one 2 shuffle_atomic  shuffle atomic
run_one 2 shuffle_stage2  shuffle stage2
run_one 2 cub_stage2      cub     stage2

# Round 3
run_one 3 shuffle_stage2  shuffle stage2
run_one 3 cub_stage2      cub     stage2
run_one 3 shuffle_atomic  shuffle atomic

# Round 4
run_one 4 shuffle_atomic  shuffle atomic
run_one 4 cub_stage2      cub     stage2
run_one 4 shuffle_stage2  shuffle stage2

# Round 5
run_one 5 cub_stage2      cub     stage2
run_one 5 shuffle_atomic  shuffle atomic
run_one 5 shuffle_stage2  shuffle stage2

# Round 6
run_one 6 shuffle_stage2  shuffle stage2
run_one 6 shuffle_atomic  shuffle atomic
run_one 6 cub_stage2      cub     stage2


echo
echo "============================================================"
echo "All final Argmax A/B/C rounds completed"
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

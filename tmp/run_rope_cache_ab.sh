#!/usr/bin/env bash

set -euo pipefail

cd /data/llaisys-26s

export PYTHONPATH=/data/llaisys-26s/python:$PYTHONPATH

OUT_DIR="/data/llaisys-26s/tmp/rope_cache_ab"

mkdir -p "${OUT_DIR}"

for i in 1 2 3 4 5
do
	echo "========================================"
	echo "Round ${i} — DIRECT"
	echo "========================================"

	LLAISYS_METAX_ROPE_CACHE=direct \
	python test/ops/rope.py \
		--device metax \
		--profile \
		2>&1 \
		| tee "${OUT_DIR}/round_${i}_direct.log"

	echo

	echo "========================================"
	echo "Round ${i} — CACHED"
	echo "========================================"

	LLAISYS_METAX_ROPE_CACHE=cached \
	python test/ops/rope.py \
		--device metax \
		--profile \
		2>&1 \
		| tee "${OUT_DIR}/round_${i}_cached.log"

	echo
done

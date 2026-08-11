#!/usr/bin/env bash

set -euo pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/rmsnorm_packed_ab"

mkdir -p "${LOG_DIR}"

for r in 1 2 3 4 5 6
do
	echo "===== Round ${r} ====="

	if (( r % 2 == 1 )); then
		echo "PACKED -> SCALAR"

		python test/ops/rms_norm.py \
			--device metax \
			--profile \
			> "${LOG_DIR}/r${r}_packed.log" 2>&1

		LLAISYS_METAX_RMS_NORM_FORCE_SCALAR=1 \
		python test/ops/rms_norm.py \
			--device metax \
			--profile \
			> "${LOG_DIR}/r${r}_scalar.log" 2>&1
	else
		echo "SCALAR -> PACKED"

		LLAISYS_METAX_RMS_NORM_FORCE_SCALAR=1 \
		python test/ops/rms_norm.py \
			--device metax \
			--profile \
			> "${LOG_DIR}/r${r}_scalar.log" 2>&1

		python test/ops/rms_norm.py \
			--device metax \
			--profile \
			> "${LOG_DIR}/r${r}_packed.log" 2>&1
	fi
done

echo
echo "===== All rounds finished ====="
echo "Logs: ${LOG_DIR}"
echo
echo "===== Median results ====="

grep -H "median=" "${LOG_DIR}"/*.log || true

#!/usr/bin/env bash

set -u
set -o pipefail

cd /data/llaisys-26s

LOG_DIR="/data/llaisys-26s/tmp/linear_qkv_fused_ab"

rm -rf "${LOG_DIR}"
mkdir -p "${LOG_DIR}"


run_impl() {
	local round="$1"
	local impl="$2"
	local m_order="$3"
	local shape_order="$4"
	local dtype_order="$5"

	local log="${LOG_DIR}/r${round}_${impl}.log"

	echo
	echo "============================================================"
	echo "QKV Linear portable-vs-fused"
	echo "round=${round}"
	echo "impl=${impl}"
	echo "M_order=${m_order}"
	echo "shape_order=${shape_order}"
	echo "dtype_order=${dtype_order}"
	echo "============================================================"

	if ! python \
		/data/llaisys-26s/tmp/profile_linear_qkv_fused_ab.py \
		--device metax \
		--impl "${impl}" \
		--order "${m_order}" \
		--shape-order "${shape_order}" \
		--dtype-order "${dtype_order}" \
		> "${log}" 2>&1
	then
		echo
		echo "FAILED:"
		echo "${log}"
		echo
		cat "${log}"
		exit 1
	fi

	echo "PASS: round=${round} impl=${impl}"
}


# ============================================================
# 6 balanced rounds
#
# 2 shapes x 10 M x 3 dtype x 2 impl x 6 rounds
# = 720 profile records
# ============================================================

run_impl 1 portable asc  q_proj,kv_proj f32,f16,bf16
run_impl 1 fused    asc  q_proj,kv_proj f32,f16,bf16

run_impl 2 fused    desc kv_proj,q_proj bf16,f16,f32
run_impl 2 portable desc kv_proj,q_proj bf16,f16,f32

run_impl 3 portable asc  kv_proj,q_proj f16,bf16,f32
run_impl 3 fused    asc  kv_proj,q_proj f16,bf16,f32

run_impl 4 fused    desc q_proj,kv_proj f32,bf16,f16
run_impl 4 portable desc q_proj,kv_proj f32,bf16,f16

run_impl 5 fused    asc  kv_proj,q_proj bf16,f32,f16
run_impl 5 portable asc  kv_proj,q_proj bf16,f32,f16

run_impl 6 portable desc q_proj,kv_proj f16,f32,bf16
run_impl 6 fused    desc q_proj,kv_proj f16,f32,bf16


echo
echo "============================================================"
echo "QKV Linear portable-vs-fused COMPLETE"
echo "============================================================"

echo
echo "Expected:"
echo "2 shapes x 10 M x 3 dtype x 2 impl x 6 rounds = 720"

echo
echo "Logs:"
echo "${LOG_DIR}"
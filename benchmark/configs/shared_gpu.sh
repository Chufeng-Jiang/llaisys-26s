#!/usr/bin/env bash

# ============================================================
# Shared cross-GPU benchmark configuration
# ============================================================

export LLAISYS_DEBUG=0

# Same total threads per block on every CUDA-compatible backend.
export LLAISYS_BLOCK_SIZE=256

# Only genuinely operator-specific mechanisms remain separate.
export LLAISYS_EMBEDDING_VECTORIZED=1
export LLAISYS_ROPE_IMPL=direct

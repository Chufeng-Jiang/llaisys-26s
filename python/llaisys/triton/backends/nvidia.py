from contextlib import contextmanager

import torch

from .base import TritonBackend
from ...libllaisys import DataType


_NVIDIA_TRITON_DTYPES = {
    DataType.F32: torch.float32,
    DataType.F16: torch.float16,
    DataType.BF16: torch.bfloat16,
    DataType.I64: torch.int64,
}


def to_triton_dtype(dtype: DataType):
    try:
        return _NVIDIA_TRITON_DTYPES[dtype]

    except KeyError as exc:
        raise TypeError(f"Unsupported NVIDIA Triton dtype: {dtype}") from exc


class NvidiaTritonBackend(TritonBackend):
    def __init__(self):
        self._streams = {}

        self._execution_depth = 0
        self._active_stream_ptr = None
        self._active_device_id = None

    # ============================================================
    # Add
    # ============================================================

    def add_config(self, numel: int) -> dict:
        return {"BLOCK_SIZE": 256, "num_warps": 4}

    # ============================================================
    # SwiGLU
    # ============================================================

    def swiglu_config(self, numel: int) -> dict:
        return {"BLOCK_SIZE": 256, "num_warps": 4}

    # ============================================================
    # RMSNorm
    # ============================================================

    def rms_norm_config(self, ncol: int) -> dict:
        if ncol <= 0:
            raise ValueError("RMSNorm row width must be positive")

        block_size = 1 << (ncol - 1).bit_length()

        if block_size > 65536:
            raise ValueError("RMSNorm Triton baseline currently supports row widths up to 65536 elements")

        if block_size <= 2048:
            num_warps = 4

        else:
            num_warps = 8

        return {"BLOCK_SIZE": block_size, "num_warps": num_warps}

    # ============================================================
    # RoPE
    # ============================================================

    def rope_config(self, head_dim: int) -> dict:
        if head_dim <= 0:
            raise ValueError("RoPE head dimension must be positive")

        if head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")

        return {"BLOCK_SIZE": 128, "num_warps": 4}

    # ============================================================
    # Argmax
    # ============================================================

    def argmax_config(self, numel: int) -> dict:
        if numel <= 0:
            raise ValueError("Argmax input must not be empty")

        return {"STAGE1_BLOCK_SIZE": 1024, "STAGE1_NUM_WARPS": 4, "STAGEN_BLOCK_SIZE": 1024, "STAGEN_NUM_WARPS": 4}

    # ============================================================
    # Embedding
    #
    # Functional baseline only.
    #
    # One Triton program copies:
    #
    #     one row
    #     ×
    #     one 128-element dimension tile
    #
    # Do not import the old RTX4090-tuned 8-warp policy yet.
    # ============================================================

    def embedding_config(self, embedding_dim: int) -> dict:
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        return {"BLOCK_SIZE": 128, "num_warps": 4}

    def linear_config(self, m: int, n: int, k: int) -> dict:
        if m <= 0:
            raise ValueError("Linear row count must be positive")

        if n <= 0:
            raise ValueError("Linear output feature count must be positive")

        if k < 0:
            raise ValueError("Linear input feature count must not be negative")

        # ========================================================
        # Functional portability baseline
        #
        # Do not tune per workload yet.
        #
        # These dimensions are deliberately tensor-core-friendly
        # while remaining small enough for decode-shaped M = 1.
        # ========================================================

        return {
            "BLOCK_M": 16,
            "BLOCK_N": 32,
            "BLOCK_K": 32,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
            "ZERO_K_BLOCK_SIZE": 256,
        }

    def self_attention_config(self, qk_dim: int, value_dim: int, total_len: int) -> dict:
        if qk_dim <= 0:
            raise ValueError("Self-Attention Q/K dimension must be positive")

        if value_dim <= 0:
            raise ValueError("Self-Attention value dimension must be positive")

        if total_len <= 0:
            raise ValueError("Self-Attention KV length must be positive")

        # ========================================================
        # Functional portability baseline
        #
        # BLOCK_M:
        #     query-sequence tile
        #
        # BLOCK_N:
        #     KV-sequence tile
        #
        # BLOCK_D:
        #     Q/K reduction tile. QK_DIM may span multiple tiles.
        #
        # BLOCK_V:
        #     output-value tile. V_DIM may span multiple tiles.
        #
        # This is intentionally fixed-policy code, not autotuning.
        # ========================================================

        return {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_D": 32, "BLOCK_V": 64, "num_warps": 4, "num_stages": 2}

    # ============================================================
    # External CUDA stream
    # ============================================================

    def _get_external_stream(self, stream_ptr: int, device_id: int):
        if stream_ptr == 0:
            raise ValueError("NVIDIA Triton requires a valid LLAISYS CUDA stream")

        key = (device_id, stream_ptr)

        if key not in self._streams:
            self._streams[key] = torch.cuda.ExternalStream(stream_ptr, device=device_id)

        return self._streams[key]

    # ============================================================
    # Execution-context query
    # ============================================================

    def in_execution_context(self, stream_ptr: int, device_id: int) -> bool:
        return (
            self._execution_depth > 0 and self._active_stream_ptr == stream_ptr and self._active_device_id == device_id
        )

    # ============================================================
    # Per-op stream context
    # ============================================================

    @contextmanager
    def stream_context(self, stream_ptr: int, device_id: int):
        external_stream = self._get_external_stream(stream_ptr, device_id)

        with torch.cuda.stream(external_stream):
            yield

    # ============================================================
    # Execution-level context
    # ============================================================

    @contextmanager
    def execution_context(self, stream_ptr: int, device_id: int):
        if self._execution_depth > 0:
            if self._active_stream_ptr != stream_ptr or self._active_device_id != device_id:
                raise RuntimeError("Cannot enter a different NVIDIA Triton stream while an execution context is active")

            self._execution_depth += 1

            try:
                yield

            finally:
                self._execution_depth -= 1

            return

        external_stream = self._get_external_stream(stream_ptr, device_id)

        self._active_stream_ptr = stream_ptr

        self._active_device_id = device_id

        self._execution_depth = 1

        try:
            with torch.cuda.stream(external_stream):
                yield

        finally:
            self._execution_depth = 0
            self._active_stream_ptr = None
            self._active_device_id = None

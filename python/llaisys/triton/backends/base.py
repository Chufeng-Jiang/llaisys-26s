import os

from abc import ABC, abstractmethod
from contextlib import contextmanager


# ============================================================
# Environment helpers
# ============================================================


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")

    return parsed


def _env_power_of_two(name: str, default: int, minimum: int = 1, maximum: int = 65536) -> int:
    value = _env_int(name, default, minimum, maximum)

    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two")

    return value


def _env_num_warps(name: str, default: int) -> int:
    value = _env_int(name, default, 1, 8)

    if value not in (1, 2, 4, 8):
        raise ValueError(f"{name} must be one of: 1, 2, 4, 8")

    return value


# ============================================================
# Triton backend
# ============================================================


class TritonBackend(ABC):
    name = "unknown"
    device_type = None

    def __init__(self):
        self._execution_depth = 0
        self._active_stream_ptr = None
        self._active_device_id = None

    # ============================================================
    # DType bridge
    # ============================================================

    @abstractmethod
    def to_triton_dtype(self, dtype):
        raise NotImplementedError

    # ============================================================
    # Stream bridge
    # ============================================================

    @abstractmethod
    @contextmanager
    def stream_context(self, stream_ptr: int, device_id: int):
        raise NotImplementedError

    # ============================================================
    # Execution context
    # ============================================================

    def in_execution_context(self, stream_ptr: int, device_id: int) -> bool:
        return (
            self._execution_depth > 0 and self._active_stream_ptr == stream_ptr and self._active_device_id == device_id
        )

    @contextmanager
    def execution_context(self, stream_ptr: int, device_id: int):
        if self._execution_depth > 0:
            if self._active_stream_ptr != stream_ptr or self._active_device_id != device_id:
                raise RuntimeError("Cannot enter a different Triton stream while an execution context is active")

            self._execution_depth += 1

            try:
                yield
            finally:
                self._execution_depth -= 1

            return

        self._active_stream_ptr = stream_ptr
        self._active_device_id = device_id
        self._execution_depth = 1

        try:
            with self.stream_context(stream_ptr, device_id):
                yield
        finally:
            self._execution_depth = 0
            self._active_stream_ptr = None
            self._active_device_id = None

    # ============================================================
    # Add
    # ============================================================

    def add_config(self, numel: int) -> dict:
        return {
            "BLOCK_SIZE": _env_power_of_two("LLAISYS_TRITON_BLOCK_SIZE", 256),
            "num_warps": _env_num_warps("LLAISYS_TRITON_NUM_WARPS", 4),
        }

    # ============================================================
    # SwiGLU
    # ============================================================

    def swiglu_config(self, numel: int) -> dict:
        return {
            "BLOCK_SIZE": _env_power_of_two("LLAISYS_TRITON_BLOCK_SIZE", 256),
            "num_warps": _env_num_warps("LLAISYS_TRITON_NUM_WARPS", 4),
        }

    # ============================================================
    # RMSNorm
    # ============================================================

    def rms_norm_config(self, ncol: int) -> dict:
        if ncol <= 0:
            raise ValueError("RMSNorm row width must be positive")

        default_block_size = 1 << (ncol - 1).bit_length()

        if default_block_size > 65536:
            raise ValueError("RMSNorm Triton baseline currently supports row widths up to 65536 elements")

        default_num_warps = 4 if default_block_size <= 2048 else 8

        block_size = _env_power_of_two("LLAISYS_TRITON_BLOCK_SIZE", default_block_size)

        if block_size < ncol:
            raise ValueError("LLAISYS_TRITON_BLOCK_SIZE must be greater than or equal to ncol for RMSNorm")

        return {"BLOCK_SIZE": block_size, "num_warps": _env_num_warps("LLAISYS_TRITON_NUM_WARPS", default_num_warps)}

    # ============================================================
    # RoPE
    # ============================================================

    def rope_config(self, head_dim: int) -> dict:
        if head_dim <= 0:
            raise ValueError("RoPE head dimension must be positive")

        if head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")

        return {
            "BLOCK_SIZE": _env_power_of_two("LLAISYS_TRITON_BLOCK_SIZE", 128),
            "num_warps": _env_num_warps("LLAISYS_TRITON_NUM_WARPS", 4),
        }

    # ============================================================
    # Argmax
    # ============================================================

    def argmax_config(self, numel: int) -> dict:
        if numel <= 0:
            raise ValueError("Argmax input must not be empty")

        default_block_size = _env_power_of_two("LLAISYS_TRITON_BLOCK_SIZE", 1024)

        default_num_warps = _env_num_warps("LLAISYS_TRITON_NUM_WARPS", 4)

        return {
            "STAGE1_BLOCK_SIZE": _env_power_of_two("LLAISYS_TRITON_STAGE1_BLOCK_SIZE", default_block_size),
            "STAGE1_NUM_WARPS": _env_num_warps("LLAISYS_TRITON_STAGE1_NUM_WARPS", default_num_warps),
            "STAGEN_BLOCK_SIZE": _env_power_of_two("LLAISYS_TRITON_STAGEN_BLOCK_SIZE", default_block_size),
            "STAGEN_NUM_WARPS": _env_num_warps("LLAISYS_TRITON_STAGEN_NUM_WARPS", default_num_warps),
        }

    # ============================================================
    # Embedding
    # ============================================================

    def embedding_config(self, embedding_dim: int) -> dict:
        if embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        return {
            "BLOCK_SIZE": _env_power_of_two("LLAISYS_TRITON_BLOCK_SIZE", 128),
            "num_warps": _env_num_warps("LLAISYS_TRITON_NUM_WARPS", 4),
        }

    # ============================================================
    # Linear
    # ============================================================

    def linear_config(self, m: int, n: int, k: int) -> dict:
        if m <= 0:
            raise ValueError("Linear row count must be positive")

        if n <= 0:
            raise ValueError("Linear output feature count must be positive")

        if k < 0:
            raise ValueError("Linear input feature count must not be negative")

        return {
            "BLOCK_M": _env_power_of_two("LLAISYS_TRITON_BLOCK_M", 16),
            "BLOCK_N": _env_power_of_two("LLAISYS_TRITON_BLOCK_N", 32),
            "BLOCK_K": _env_power_of_two("LLAISYS_TRITON_BLOCK_K", 32),
            "GROUP_M": _env_int("LLAISYS_TRITON_GROUP_M", 8, 1, 64),
            "num_warps": _env_num_warps("LLAISYS_TRITON_NUM_WARPS", 4),
            "num_stages": _env_int("LLAISYS_TRITON_NUM_STAGES", 3, 1, 8),
            "ZERO_K_BLOCK_SIZE": _env_power_of_two("LLAISYS_TRITON_ZERO_K_BLOCK_SIZE", 256),
        }

    # ============================================================
    # Self-Attention
    # ============================================================

    def self_attention_config(self, qk_dim: int, value_dim: int, total_len: int) -> dict:
        if qk_dim <= 0:
            raise ValueError("Self-Attention Q/K dimension must be positive")

        if value_dim <= 0:
            raise ValueError("Self-Attention value dimension must be positive")

        if total_len <= 0:
            raise ValueError("Self-Attention KV length must be positive")

        return {
            "BLOCK_M": _env_power_of_two("LLAISYS_TRITON_BLOCK_M", 16),
            "BLOCK_N": _env_power_of_two("LLAISYS_TRITON_BLOCK_N", 32),
            "BLOCK_D": _env_power_of_two("LLAISYS_TRITON_BLOCK_D", 32),
            "BLOCK_V": _env_power_of_two("LLAISYS_TRITON_BLOCK_V", 64),
            "num_warps": _env_num_warps("LLAISYS_TRITON_NUM_WARPS", 4),
            "num_stages": _env_int("LLAISYS_TRITON_NUM_STAGES", 2, 1, 8),
        }

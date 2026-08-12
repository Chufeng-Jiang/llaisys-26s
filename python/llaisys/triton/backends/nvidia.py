from contextlib import contextmanager

import torch

from ...libllaisys import DataType
from .base import TritonBackend

_NVIDIA_TRITON_DTYPES = {DataType.F32: torch.float32, DataType.F16: torch.float16, DataType.BF16: torch.bfloat16}


def to_triton_dtype(dtype: DataType):
    try:
        return _NVIDIA_TRITON_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported NVIDIA Triton dtype: {dtype}") from exc


class NvidiaTritonBackend(TritonBackend):
    def __init__(self):
        self._streams = {}

        # Active execution context state.
        self._execution_depth = 0
        self._active_stream_ptr = None
        self._active_device_id = None

    def add_config(self, numel: int) -> dict:
        return {"BLOCK_SIZE": 256, "num_warps": 4}

    def _get_external_stream(self, stream_ptr: int, device_id: int):
        if stream_ptr == 0:
            raise ValueError("NVIDIA Triton requires a valid LLAISYS CUDA stream")

        key = (device_id, stream_ptr)

        if key not in self._streams:
            self._streams[key] = torch.cuda.ExternalStream(stream_ptr, device=device_id)

        return self._streams[key]

    def in_execution_context(self, stream_ptr: int, device_id: int) -> bool:
        return (
            self._execution_depth > 0 and self._active_stream_ptr == stream_ptr and self._active_device_id == device_id
        )

    @contextmanager
    def stream_context(self, stream_ptr: int, device_id: int):
        external_stream = self._get_external_stream(stream_ptr, device_id)

        with torch.cuda.stream(external_stream):
            yield

    @contextmanager
    def execution_context(self, stream_ptr: int, device_id: int):
        # Nested use on the same stream is allowed.
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

from contextlib import contextmanager

import torch

from .base import TritonBackend
from ...libllaisys import DataType


_NVIDIA_TRITON_DTYPES = {
    DataType.F32: torch.float32,
    DataType.F16: torch.float16,
    DataType.BF16: torch.bfloat16,
}


def to_triton_dtype(dtype: DataType):
    try:
        return _NVIDIA_TRITON_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(
            f"Unsupported NVIDIA Triton dtype: {dtype}"
        ) from exc


class NvidiaTritonBackend(TritonBackend):

    def __init__(self):
        # Cache PyTorch wrappers for LLAISYS-owned CUDA streams.
        #
        # key:
        #     (device_id, raw cudaStream_t)
        #
        # value:
        #     torch.cuda.ExternalStream
        #
        # LLAISYS remains the owner of the real CUDA stream.
        self._streams = {}

    def add_config(
        self,
        numel: int,
    ) -> dict:
        return {
            "BLOCK_SIZE": 256,
            "num_warps": 4,
        }

    def _get_external_stream(
        self,
        stream_ptr: int,
        device_id: int,
    ):
        if stream_ptr == 0:
            raise ValueError(
                "NVIDIA Triton requires a valid "
                "LLAISYS CUDA stream"
            )

        key = (
            device_id,
            stream_ptr,
        )

        if key not in self._streams:
            self._streams[key] = torch.cuda.ExternalStream(
                stream_ptr,
                device=device_id,
            )

        return self._streams[key]

    @contextmanager
    def stream_context(
        self,
        stream_ptr: int,
        device_id: int,
    ):
        external_stream = self._get_external_stream(
            stream_ptr,
            device_id,
        )

        with torch.cuda.stream(
            external_stream
        ):
            yield
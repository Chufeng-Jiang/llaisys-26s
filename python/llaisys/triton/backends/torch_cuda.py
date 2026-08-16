from contextlib import contextmanager

import torch

from .base import TritonBackend
from ...libllaisys import DataType


_TORCH_TRITON_DTYPES = {
    DataType.F32: torch.float32,
    DataType.F16: torch.float16,
    DataType.BF16: torch.bfloat16,
    DataType.I64: torch.int64,
}


class TorchCudaLikeTritonBackend(TritonBackend):
    def __init__(self):
        super().__init__()
        self._streams = {}

    def to_triton_dtype(self, dtype: DataType):
        try:
            return _TORCH_TRITON_DTYPES[dtype]
        except KeyError as exc:
            raise TypeError(f"Unsupported Triton dtype: {dtype}") from exc

    def _get_external_stream(self, stream_ptr: int, device_id: int):
        if stream_ptr == 0:
            raise ValueError(f"{self.name} Triton requires a valid external stream")

        key = (device_id, stream_ptr)

        if key not in self._streams:
            self._streams[key] = torch.cuda.ExternalStream(stream_ptr, device=device_id)

        return self._streams[key]

    @contextmanager
    def stream_context(self, stream_ptr: int, device_id: int):
        external_stream = self._get_external_stream(stream_ptr, device_id)

        with torch.cuda.stream(external_stream):
            yield

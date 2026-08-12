from contextlib import contextmanager

from ..libllaisys import DeviceType
from ..runtime import RuntimeAPI
from .ops import _nvidia_backend


@contextmanager
def execution_context(device_type: DeviceType, device_id: int = 0):
    if device_type != DeviceType.NVIDIA:
        raise ValueError(f"Unsupported Triton execution device: {device_type}")

    runtime = RuntimeAPI(device_type)

    stream_ptr = runtime.get_context_stream(device_id)

    with _nvidia_backend.execution_context(stream_ptr, device_id):
        yield

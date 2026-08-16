from contextlib import contextmanager

from ..libllaisys import DeviceType
from ..runtime import RuntimeAPI
from .backends.registry import get_triton_backend


@contextmanager
def execution_context(device_type: DeviceType, device_id: int = 0):
    backend = get_triton_backend(device_type)
    runtime = RuntimeAPI(device_type)

    stream_ptr = runtime.get_context_stream(device_id)

    with backend.execution_context(stream_ptr, device_id):
        yield

from ...libllaisys import DeviceType

from .metax import MetaXTritonBackend
from .nvidia import NvidiaTritonBackend


_BACKENDS = {DeviceType.NVIDIA: NvidiaTritonBackend(), DeviceType.METAX: MetaXTritonBackend()}


def is_triton_device_supported(device_type: DeviceType) -> bool:
    return device_type in _BACKENDS


def get_triton_backend(device_type: DeviceType):
    try:
        return _BACKENDS[device_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported Triton device type: {device_type}") from exc

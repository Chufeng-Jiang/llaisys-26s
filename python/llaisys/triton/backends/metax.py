from .torch_cuda import TorchCudaLikeTritonBackend
from ...libllaisys import DeviceType


class MetaXTritonBackend(TorchCudaLikeTritonBackend):
    name = "metax"
    device_type = DeviceType.METAX

from .torch_cuda import TorchCudaLikeTritonBackend
from ...libllaisys import DeviceType


class NvidiaTritonBackend(TorchCudaLikeTritonBackend):
    name = "nvidia"
    device_type = DeviceType.NVIDIA

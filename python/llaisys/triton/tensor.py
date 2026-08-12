from .backends.nvidia import to_triton_dtype


class TritonTensor:
    def __init__(self, tensor):
        self._tensor = tensor
        self.dtype = to_triton_dtype(tensor.dtype())

    def data_ptr(self) -> int:
        return self._tensor.data_ptr()

    @property
    def tensor(self):
        return self._tensor


def as_nvidia_triton_tensor(tensor) -> TritonTensor:
    return TritonTensor(tensor)

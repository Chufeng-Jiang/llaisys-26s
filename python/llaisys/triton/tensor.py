from .backends.registry import get_triton_backend


class TritonTensor:
    def __init__(self, tensor, backend=None):
        self._tensor = tensor

        if backend is None:
            backend = get_triton_backend(tensor.device_type())

        self.dtype = backend.to_triton_dtype(tensor.dtype())

    def data_ptr(self) -> int:
        return self._tensor.data_ptr()

    @property
    def tensor(self):
        return self._tensor


def as_triton_tensor(tensor, backend=None) -> TritonTensor:
    return TritonTensor(tensor, backend=backend)

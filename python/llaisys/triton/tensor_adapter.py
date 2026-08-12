class TritonTensorAdapter:
    def __init__(self, tensor, ptr: int, dtype):
        self.tensor = tensor
        self._ptr = ptr
        self.dtype = dtype

    def data_ptr(self) -> int:
        return self._ptr

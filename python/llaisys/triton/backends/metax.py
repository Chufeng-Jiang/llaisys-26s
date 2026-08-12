from .base import TritonBackend


class MetaXTritonBackend(TritonBackend):
    def add_config(self, numel: int) -> dict:
        raise NotImplementedError("MetaX Triton backend is not implemented yet.")

from abc import ABC, abstractmethod


class TritonBackend(ABC):

    @abstractmethod
    def add_config(self, numel: int) -> dict:
        pass

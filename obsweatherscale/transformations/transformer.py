import abc
import torch


class Transformer():
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def description(self) -> str:
        pass

    @abc.abstractmethod
    def transform(self, y: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def noise_transform(self, data: torch.Tensor) -> torch.Tensor:
        pass

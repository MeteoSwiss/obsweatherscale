import abc
import torch


class Transformer:

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Return a short description of the transformation."""

    @abc.abstractmethod
    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Apply transformation to the input data."""

    @abc.abstractmethod
    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        """Apply inverse transformation to input data."""

    @abc.abstractmethod
    def noise_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply noise transformation to input data."""

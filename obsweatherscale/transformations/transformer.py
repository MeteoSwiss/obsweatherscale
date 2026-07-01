"""Data transformation classes for obsweatherscale.

Provides a base :class:`Transformer`.

Classes
-------
Transformer
    Abstract base class for all data transformations.
"""


import abc
import torch


class Transformer:
    """Abstract base class for all data transformations.
 
    All transformers must implement:
      - transform: forward transformation y → z
      - inverse_transform: backward transformation z → y
      - noise_transform: how noise scales under the transformation
      - description: human-readable summary
    """

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
    def noise_transform(self, z: torch.Tensor) -> torch.Tensor:
        """Return the Jacobian factor for noise propagation at z.

        Given targets in the transformed space z = f(y), returns
        f'(f⁻¹(z)), the local derivative of the forward transform
        evaluated at the corresponding original-space value. This factor
        is used to scale noise variance under the transformation:

            σ²_z = σ²_y · [f'(f⁻¹(z))]²

        Parameters
        ----------
        z : torch.Tensor
            Targets in the transformed space.

        Returns
        -------
        torch.Tensor
            Pointwise Jacobian factor f'(f⁻¹(z)), same shape as z.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.description})"

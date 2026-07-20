"""Data transformation classes for obsweatherscale.

Provides a base :class:`Transformer` and two child abstract
implementations: one fitted transformations that learn parameters from
data and one for parametric transformations with fixed parameters.

Classes
-------
Transformer
    Abstract base class for all data transformations.
ParametricTransformer
    Base class for transformers with fixed parameters (no fitting
    needed).
FittedTransformer
    Base class for transformers that learn parameters from data.
"""


import abc
import warnings

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


class ParametricTransformer(Transformer, abc.ABC):
    """Base class for transformers with fixed parameters (no fitting needed).
 
    Parameters are set at construction time. No data is required before
    calling transform().
    """


class FittedTransformer(Transformer, abc.ABC):
    """Base class for transformers that learn parameters from data.
 
    Subclasses must call fit() before transform(), or provide sensible
    defaults that make the unfitted state explicit.
    """

    _fitted: bool = False

    @abc.abstractmethod
    def fit(self, data: torch.Tensor) -> None:
        """Fit transformation parameters to input data."""

    @property
    def is_fitted(self) -> bool:
        """Getter of fitted state."""
        return self._fitted

    def _check_fitted(self) -> None:
        """Warn if the transformer has not been fitted yet."""
        if not self._fitted:
            warnings.warn(
                f"{self.__class__.__name__} has not been fitted. "
                "Calling transform() will use default parameters. "
                "Call fit() first to learn parameters from data.",
                UserWarning,
                stacklevel=3,
            )

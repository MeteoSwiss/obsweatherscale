"""Standardization transformation class.

Classes
-------
Standardizer
    Standardization transformation (zero mean, unit variance).
"""

import torch

from .transformer import FittedTransformer


class Standardizer(FittedTransformer):
    """Standardization transformation (zero mean, unit variance).

    Can be initialized in two ways:
      1. Pass ``data`` to fit immediately at construction time.
      2. Pass explicit ``mean`` / ``std``, or leave as defaults and call
         ``fit()`` later.

    Parameters
    ----------
    data : torch.Tensor, optional
        If provided, ``fit()`` is called immediately on this data.
    mean : float or torch.Tensor
        Initial mean. Overwritten by ``fit()``. Default is 0.
    std : float or torch.Tensor
        Initial standard deviation. Overwritten by ``fit()``. Default is 1.
    dims : tuple[int, ...] or int, optional
        Dimensions over which to compute mean and std.
        If None, all dimensions are used.
    """

    def __init__(
        self,
        data: torch.Tensor | None = None,
        mean: float | torch.Tensor = 0.,
        std: float | torch.Tensor = 1.,
        dims: tuple[int, ...] | int | None = None,
    ) -> None:
        self.dims = dims
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

        if data is not None:
            self.fit(data)

    @property
    def description(self) -> str:
        """Return a short description of the sandard normalization."""
        return "Standard normalization: f(y) = (y - mean(y)) / std(y)"

    def fit(self, data: torch.Tensor) -> None:
        """Fit mean and standard deviation from input data.

        Parameters
        ----------
        data : torch.Tensor
            The data to be standardized.
        """
        self.mean = data.mean(dim=self.dims).squeeze()
        self.std = data.std(dim=self.dims).squeeze()
        self._fitted = True

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Apply standardization: z = (y - mean) / std."""
        self._check_fitted()
        return (y - self.mean) / self.std

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        """Invert standardization: y = z * std + mean."""
        self._check_fitted()
        return z * self.std + self.mean

    def transform_derivative(self, y: torch.Tensor) -> torch.Tensor:
        self._check_fitted()
        return torch.ones_like(y) / self.std

    def noise_transform(self, z: torch.Tensor) -> torch.Tensor:
        """Return the Jacobian factor for noise propagation at z.

        For standardization f(y) = (y - μ) / σ, the derivative
        f'(y) = 1/σ is constant, so the Jacobian factor is independent
        of z.

        Parameters
        ----------
        z : torch.Tensor
            Targets in the transformed space. Used only to determine the
            output shape; the values do not affect the result.

        Returns
        -------
        torch.Tensor
            Tensor of shape matching z, filled with 1/σ.
        """
        self._check_fitted()
        return torch.ones_like(z) / self.std

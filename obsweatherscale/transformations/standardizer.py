"""Standardization transformation class.

Classes
-------
Standardizer
    Standardization transformation (zero mean, unit variance).
"""

import torch


class Standardizer:
    """Standardization transformation (zero mean, unit variance).

    Parameters
    ----------
    data : torch.Tensor
        The data to be standardized.
    variables : tuple[int, ...] or int, optional
        The dimensions to be used for standardization.
        If None, all dimensions will be used.
    """

    def __init__(
        self,
        data: torch.Tensor,
        variables: tuple[int, ...] | int | None = None,
    ) -> None:
        self.fit(data, variables)

    @property
    def description(self) -> str:
        """Return a short description of the sandard normalization."""
        return "Standard normalization: f(y) = (y - mean(y)) / std(y)"

    def fit(
        self,
        data: torch.Tensor,
        variables: tuple[int, ...] | int | None = None
    ) -> None:
        """Fit mean and standard deviation from input data.

        Parameters
        ----------
        data : torch.Tensor
            The data to be standardized.
        variables : tuple[int, ...] or int, optional
            The dimensions to be used for standardization.
            If None, all dimensions will be used.
        """
        self.mean = data.mean(dim=variables).squeeze()
        self.std = data.std(dim=variables).squeeze()

    def transform(self, y: torch.Tensor, copy: bool = False) -> torch.Tensor:
        """Apply standardization: z = (y - mean) / std."""
        if copy:
            y = y.detach().clone()
        return (y - self.mean) / self.std

    def inverse_transform(
        self,
        z: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        """Invert standardization: y = z * std + mean."""
        if copy:
            z = z.detach().clone()
        return z * self.std + self.mean

"""QuantileFittedTransformer class.

Classes
-------
QuantileFittedTransformer
    Continuous approximation of a quantile transform.
"""

import torch

from .transformer import ParametricTransformer


class QuantileFittedTransformer(ParametricTransformer):
    """Continuous approximation of a quantile transform.

    Approximates the quantile transform using:
        f(y)    = -log(a / y - c) / b
        f⁻¹(z) =  a / (c + exp(-b * z))

    Parameters
    ----------
    a, b, c : float
        Shape parameters of the transformation. Defaults are fitted to
        a reference quantile distribution.
    """

    def __init__(
        self, a: float = 4.66628594,
        b: float = 0.73680252,
        c: float = 0.07385268,
    ) -> None:
        self.a = a
        self.b = b
        self.c = c

    @property
    def description(self) -> str:
        """Return a short description of the quantile fitted transformation."""
        return (
            "Continuous function approximating quantile transform: "
            "f(y) = log(a / y - c) / b"
        )

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Apply quantile fitted transformation to input data."""
        y = torch.clip(y, 1e-3, 70.0)
        return -torch.log(self.a / y - self.c) / self.b

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        """Apply inverse quantile fitted transformation to input data."""
        return self.a / (self.c + torch.exp(-self.b * z))

    def transform_derivative(self, y: torch.Tensor) -> torch.Tensor:
        """Compute df/dy."""
        return self.a / (self.b * y * (self.a - self.c * y))

    def inv_transform_derivative(self, z: torch.Tensor) -> torch.Tensor:
        """Compute df⁻¹/dz."""
        exp_neg_bz = torch.exp(-self.b * z)
        return (self.a * self.b * exp_neg_bz) / ((self.c + exp_neg_bz) ** 2)

    def noise_transform(self, z: torch.Tensor) -> torch.Tensor:
        """Return the Jacobian factor for noise propagation at z.

        Evaluates f'(f⁻¹(z)), the derivative of the forward transform at
        the original-space value corresponding to z.

        Parameters
        ----------
        z : torch.Tensor
            Targets in the transformed space.

        Returns
        -------
        torch.Tensor
            Pointwise Jacobian factor f'(f⁻¹(z)), same shape as z.
        """
        return self.transform_derivative(self.inverse_transform(z))

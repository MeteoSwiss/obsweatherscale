import torch

from .transformer import Transformer

class QuantileFittedTransformer(Transformer):
    """QuantileFittedTransformer class.

    This class implements a quantile fitted transformation for continuous
    functions. It approximates the quantile transform using a continuous
    function defined by the formula: f(y) = log(a / y - c) / b, where
    a, b, and c are parameters of the transformation. The inverse
    transformation is defined by the formula: f^-1(z) = a / (c + exp(-b * z)).

    """
    def __init__(
        self,
        a: float = 4.66628594,
        b: float = 0.73680252,
        c: float = 0.07385268
    ) -> None:
        """Initialize the QuantileFittedTransformer.

        Parameters
        ----------
        a : float, default=4.66628594
            The parameter a for the transformation.
        b : float, default=0.73680252
            The parameter b for the transformation.
        c : float, default=0.07385268
            The parameter c for the transformation.
        """
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
    
    def description(self) -> str:
        return (
            f"Continuous function approximating quantile transform: "
            f"f(y) = log(a / y - c) / b"
        )

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the quantile fitted transformation to the input data."""
        y = torch.clip(y, 1e-3, 70.0)
        return -torch.log(self.a / y - self.c) / self.b

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the inverse quantile fitted transformation to the input data."""
        return self.a / (self.c + torch.exp(-self.b * z))

    def transform_derivative(self, y: torch.Tensor ) -> torch.Tensor:
        """Compute the derivative of the quantile fitted transformation."""
        return self.a / (self.b * y * (self.a - self.c * y))

    def inv_transform_derivative(self, z: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of the inverse quantile fitted transformation."""
        exp_neg_bz = torch.exp(-self.b * z)
        return (self.a * self.b * exp_neg_bz) / ((self.c + exp_neg_bz) ** 2)

    def noise_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the noise transformation to the input data."""
        return self.transform_derivative(self.inverse_transform(data))

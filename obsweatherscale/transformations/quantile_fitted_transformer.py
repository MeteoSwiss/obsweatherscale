import torch

from .transformer import Transformer
from ..constants import A, B, C


class QuantileFittedTransformer(Transformer):
    def __init__(
        self,
        a: float = A,
        b: float = B,
        c: float = C
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        y = torch.clip(y, 1e-3, 70.0)
        return -torch.log(self.a / y - self.c) / self.b

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        return self.a / (self.c + torch.exp(-self.b * z))

    def transform_derivative(self, y: torch.Tensor ) -> torch.Tensor:
        # y += 1e-3
        return self.a / (self.b * y * (self.a - self.c * y))

    def inv_transform_derivative(self, z: torch.Tensor) -> torch.Tensor:
        exp_neg_bz = torch.exp(-self.b * z)
        return (self.a * self.b * exp_neg_bz) / ((self.c + exp_neg_bz) ** 2)

    def noise_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform_derivative(data)

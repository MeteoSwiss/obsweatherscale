import torch

from .transformer import Transformer
from ..constants import A, B, C

class QuantileFittedTransformer(Transformer):
    def __init__(
        self,
        a: float = A,
        b: float = B,
        c: float = C
    ):
        super(QuantileFittedTransformer, self).__init__()
        self.a = a
        self.b = b
        self.c = c
    
    def transform(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        data = torch.clip(data, 1e-3, 70.0)
        return -torch.log(self.a / data - self.c) / self.b
    
    def inverse_transform(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        return self.a / (self.c + torch.exp(-self.b * data))
    
    def transform_derivative(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        data += 1e-3
        return self.a / (self.b * data * (self.a - self.c * data))
    
    def inv_transform_derivative(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        return (
            (self.a * self.b * torch.exp(self.b * data)) /
            ((self.c * torch.exp(self.b * data) + 1) ** 2)
        )
    
    def noise_transform(
        self,
        data: torch.Tensor
    ) -> torch.Tensor:
        return self.transform_derivative(
            self.inverse_transform(data)
        )

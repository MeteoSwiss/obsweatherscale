from typing import Optional
import torch

class Standardizer():
    def __init__(
        self,
        data: torch.Tensor,
        variables: Optional[tuple[int, ...]] = None
    ):
        self.fit(data, variables)
    
    def description(self):
        return (
            f"Standard normalization: "
            f"f(y) = (y - mean(y) / stddev(y))"
        )

    def fit(
        self,
        data: torch.Tensor,
        variables: Optional[tuple[int, ...]] = None
    ):
        self.mean = data.mean(dim=variables).squeeze()
        self.std = data.std(dim=variables).squeeze()

    def transform(
        self,
        y: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        if copy:
            y = y.detach().clone()
        return (y - self.mean) / self.std

    def inverse_transform(
        self,
        z: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        if copy:
            z = z.detach().clone()
        return z*self.std + self.mean

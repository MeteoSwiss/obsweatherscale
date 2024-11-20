from typing import Optional

import torch


class Standardizer():
    def __init__(
        self,
        data: torch.Tensor,
        variables: Optional[tuple[int]] = None
    ):
        self.fit(data, variables)

    def fit(
        self,
        data: torch.Tensor,
        variables: Optional[list[str]] = None
    ):
        self.mean = data.mean(axis=variables).squeeze()
        self.std = data.std(axis=variables).squeeze()

    def transform(
        self,
        y: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        if copy:
            y = y.copy()
        return (y - self.mean) / self.std

    def inverse_transform(
        self,
        z: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        if copy:
            z = z.copy()
        return z*self.std + self.mean

import torch

class Standardizer():
    def __init__(
        self,
        data: torch.Tensor,
        variables: tuple[int] = None
    ):
        self.fit(data, variables)

    def fit(
        self,
        data: torch.Tensor,
        variables: list[str] = None
    ):
        self.mean = data.mean(axis=variables).squeeze()
        self.std = data.std(axis=variables).squeeze()

    def transform(
        self,
        data: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        if copy:
            data = data.copy()
        return (data - self.mean) / self.std

    def inverse_transform(
        self,
        data: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        if copy:
            data = data.copy()
        return data*self.std + self.mean
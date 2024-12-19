from typing import Callable, Optional

import torch
from torch import nn

from ..utils import set_active_dims


class MLP(nn.Module):
    def __init__(
        self,
        dimensions: list[int],
        output_activation_fct: Optional[Callable] = None,
        active_dims: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(len(dimensions)-2):
            layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))

        self.mlp = nn.Sequential(*layers)
        self.output_activation_fct = output_activation_fct
        self.active_dims = set_active_dims(active_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x[..., self.active_dims])
        if self.output_activation_fct is not None:
            return self.output_activation_fct(x)
        return x

from typing import Optional

import torch
import torch.nn as nn
from gpytorch.means import Mean

from ..utils.utils import set_active_dims

class NeuralMean(Mean):
    def __init__(
        self,
        net: nn.Module,
        active_dims: Optional[list[int]] = None
    ):
        super().__init__()
        self.net = net
        self.active_dims = set_active_dims(active_dims)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., self.active_dims]
        output = self.net(x)
        return output.squeeze(-1)
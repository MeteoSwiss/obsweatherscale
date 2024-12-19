from typing import Any, Union

import torch
from gpytorch.kernels import Kernel
from linear_operator.operators import LinearOperator


class NeuralKernel(Kernel):
    def __init__(self, net: torch.nn.Module, kernel: Kernel) -> None:
        super().__init__()
        self.net = net
        self.kernel = kernel

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *params: Any,
        **kwargs: Any
    ) -> Union[torch.Tensor, LinearOperator]:
        x1 = self.net(x1)
        x2 = self.net(x2)

        return self.kernel(x1, x2, *params, **kwargs)

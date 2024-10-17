import torch
from gpytorch.kernels import Kernel

class NeuralKernel(Kernel):
    def __init__(
        self,
        active_dims: list[int] | None,
        net: torch.nn.Module,
        kernel: Kernel
    ):
        super().__init__(active_dims=active_dims)
        self.net = net
        self.kernel = kernel

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *params,
        **kwargs
    ) -> torch.Tensor:
        if x1.shape[-1] > len(self.active_dims):
            x1 = x1[..., self.active_dims]
            x2 = x2[..., self.active_dims]
        x1 = self.net(x1)
        x2 = self.net(x2)

        return self.kernel(x1, x2, *params, **kwargs)

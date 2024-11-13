import torch
from gpytorch.means import Mean


class NeuralMean(Mean):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return output.squeeze(-1)

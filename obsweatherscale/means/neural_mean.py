import torch
from gpytorch.means import Mean


class NeuralMean(Mean):
    """Neural Mean class.

    Uses a neural network to compute the mean of the Gaussian process
    prior, starting from the input data.
    """
    def __init__(self, net: torch.nn.Module) -> None:
        """Initializes the NeuralMean.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network to compute the mean.
        """
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return output.squeeze(-1)

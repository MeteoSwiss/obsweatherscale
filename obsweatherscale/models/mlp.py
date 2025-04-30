from typing import Callable

import torch
from torch import nn

from ..utils import set_active_dims


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) class.

    This class implements a simple multi-layer perceptron with ReLU
    activation functions and an optional output activation function.
    """

    def __init__(
        self,
        dimensions: list[int],
        output_activation_fct: Callable | None = None,
        active_dims: list[int] | None = None,
    ) -> None:
        """Initialize the MLP model.

        Parameters
        ----------
        dimensions : list of int
            List of integers representing the dimensions of each layer.
            The first element is the input dimension, the last element
            is the output dimension, and the intermediate elements are
            hidden layer dimensions.
        output_activation_fct : callable, optional
            Activation function to apply to the output.
            If None, no activation function is applied.
        active_dims : list of int, optional
            List of indices specifying which dimensions of the input to
            use. If None, all dimensions are used.

        Examples
        --------
        # 10D input, 1D output with sigmoid
        >>> mlp = MLP([10, 64, 32, 1], torch.sigmoid)
        """
        super().__init__()

        layers = []
        for i in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
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

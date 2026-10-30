"""NeuralKernel class.

This module implements a composable neural network kernel for use with
GPyTorch.

Classes
-------
NeuralKernel
    A GPyTorch kernel that applies a neural feature map before a base
    kernel.
"""

from typing import Any

import torch
from gpytorch.kernels import Kernel
from linear_operator.operators import LinearOperator


class NeuralKernel(Kernel):
    """A kernel that applies a learned neural feature map prior to a
    base kernel.

    ``NeuralKernel`` wraps an arbitrary :class:`torch.nn.Module` and a
    GPyTorch :class:`~gpytorch.kernels.Kernel` into a single composable
    kernel. On each forward pass, the input data is transformed by the
    network into a learned feature space; the base kernel is then
    applied on the resulting representations.

    Parameters
    ----------
    net : torch.nn.Module
        Neural network used to project inputs into the feature space.
        Must accept tensors of shape ``(*, D_in)`` and return tensors of
        shape ``(*, D_out)``, where ``D_out`` is compatible with the
        expected input dimensionality of *kernel*.
    kernel : Kernel
        Base GPyTorch kernel evaluated on the projected representations.
        Any :class:`~gpytorch.kernels.Kernel` subclass is supported
        (e.g. :class:`~gpytorch.kernels.RBFKernel`,
        :class:`~gpytorch.kernels.MaternKernel`).

    Attributes
    ----------
    net : torch.nn.Module
        The neural feature extractor.
    kernel : Kernel
        The base kernel applied after the feature transformation.

    Notes
    -----
    ``NeuralKernel`` subclasses :class:`~gpytorch.kernels.Kernel`. Its
    parameters (both ``net`` weights and ``kernel`` hyperparameters) are
    registered as part of the GP model's parameter tree and are updated
    jointly during optimisation.

    The output dimensionality of *net* must match the input
    dimensionality expected by *kernel*. No shape validation is
    performed at construction time; a mismatch will surface as a runtime
    error during the first forward pass.

    .. todo::
        Add literature reference for the neural kernel construction.
    """

    def __init__(self, net: torch.nn.Module, kernel: Kernel) -> None:
        super().__init__()
        self.net = net
        self.kernel = kernel

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *params: Any,
        **kwargs: Any,
    ) -> torch.Tensor | LinearOperator:
        x1 = self.net(x1)
        x2 = self.net(x2)

        return self.kernel(x1, x2, *params, **kwargs)

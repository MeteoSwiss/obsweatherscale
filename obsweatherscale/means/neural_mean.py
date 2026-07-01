"""NeuralMean class.

This module provides :class:`NeuralMean`, a GPyTorch-compatible mean
function that parameterises the Gaussian Process prior mean using an
arbitrary :class:`torch.nn.Module`. This allows the GP to learn a
flexible, data-driven prior mean end-to-end alongside the kernel
hyperparameters, rather than relying on a fixed constant or linear mean.

Classes
-------
NeuralMean
    A GP mean function backed by a learnable neural network.
"""
import torch
from gpytorch.means import Mean


class NeuralMean(Mean):
    """A GP mean function that uses a neural network to compute the prior mean.

    Wraps an arbitrary :class:`torch.nn.Module` as a GPyTorch
    :class:`~gpytorch.means.Mean`, enabling a fully learnable prior mean
    function :math:`m(\\mathbf{x})`. 

    Uses a neural network to compute the mean of the Gaussian process
    prior, starting from the input data.

    .. math::

        m(\\mathbf{x}) = f_{\\theta}(\\mathbf{x})

    where :math:`f_{\\theta}` is a neural network with parameters
    :math:`\\theta` trained jointly with the GP kernel hyperparameters.

    Parameters
    ----------
    net : torch.nn.Module
        Neural network used to compute the prior mean. Must accept input
        tensors of shape ``(*, N, D_in)`` and return tensors of shape
        ``(*, N, 1)`` or ``(*, N)``; a trailing size-1 dimension is
        squeezed automatically.

    Attributes
    ----------
    net : torch.nn.Module
        The neural network computing the prior mean.

    Notes
    -----
    Because ``NeuralMean`` subclasses :class:`~gpytorch.means.Mean`,
    the parameters of *net* are registered in the GP model's parameter
    tree and receive gradient updates during optimization.

    The ``forward`` method squeezes the last dimension of the network
    output, so *net* returns either shape ``(*, N, 1)`` or ``(*, N)``.
    Both are handled correctly.
    """

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the neural prior mean at input locations *x*.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(*, N, D_in)``.

        Returns
        -------
        torch.Tensor
            Prior mean vector of shape ``(*, N)``.
        """
        output = self.net(x)
        return output.squeeze(-1)

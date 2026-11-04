"""GP mean functions for obsweatherscale.

Provides custom mean functions that subclass
:class:`gpytorch.means.Mean` and are fully compatible with GPyTorch's
GP model interface.

Classes
-------
NeuralMean
    A GP mean function backed by an arbitrary :class:`torch.nn.Module`.
"""

from .neural_mean import NeuralMean

__all__ = ['NeuralMean']

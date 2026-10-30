"""Custom GP kernels for obsweatherscale.

All kernels subclass :class:`gpytorch.kernels.Kernel` and are compatible
with GPyTorch's composable kernel algebra (addition, multiplication,
scaling).

Classes
-------
NeuralKernel
    A GPyTorch kernel that applies a neural feature map before a base
    kernel.
ScaledRBFKernel
    A scaled RBF kernel with optional ARD, priors, constraints, and
    parameter freezing.
"""

from .neural_kernel import NeuralKernel
from .scaled_rbf_kernel import ScaledRBFKernel

__all__ = ['NeuralKernel', 'ScaledRBFKernel']

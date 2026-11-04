"""GP model definitions for obsweatherscale.

Provides the core model components used to construct Gaussian Process
models throughout the package. All models are compatible with GPyTorch's
training and inference interfaces.

Classes
-------
GPModel
    An exact Gaussian Process model with flexible mean and covariance
    modules.
MLP
    A multi-layer perceptron used as a feature extractor within kernel
    or mean modules.
"""

from .gp_model import GPModel
from .mlp import MLP

__all__ = ["GPModel", "MLP"]

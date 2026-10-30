"""obsweatherscale - Gaussian Process models for downscaling of weather
fields, with conditioning on in-situ observations.

Provides GP model components, kernels, likelihoods, training utilities,
and data interfaces tailored for weather-scale observational datasets.

Subpackages
-----------
data
    Dataset abstractions and loaders (``GPDataset`` and subclasses).
kernels
    Custom GP kernels (``ScaledRBFKernel`, ``NeuralKernel``).
likelihoods
    Custom likelihood functions.
logger
    Terminal and file logging interfaces.
means
    Custom mean functions (``NeuralMean``).
models
    GP model definitions built on GPyTorch's ``ExactGP``.
training
    Training utilities (``Trainer``).
transformations
    Input and output data transformation abstractions and concrete
    implementations (``Standardizer``, ``QuantileFittedTransformer``).
sampling
    Distribution sampling utilities.
"""

from .data import GPDataset
from .kernels import NeuralKernel, ScaledRBFKernel
from .likelihoods import (
    TransformedGaussianLikelihood,
    ExactMarginalLogLikelihoodFill,
    TransformedHomoskedasticNoise,
    TransformedHeteroskedasticNoise,
    TransformedFixedGaussianNoise,
)
from .logger import Logger, TerminalLogger, CSVLogger, MLflowLogger
from .means import NeuralMean
from .models import GPModel, MLP
from .sampling import sample
from .training import (
    Trainer, crps_normal, make_crps_loss, make_mll_loss, make_loss,
)
from .transformations import (
    Transformer, FittedTransformer, ParametricTransformer,
    QuantileFittedTransformer, Standardizer,
)

__all__ = [
    "GPDataset",
    "NeuralKernel", "ScaledRBFKernel",
    "TransformedGaussianLikelihood", "ExactMarginalLogLikelihoodFill",
    "TransformedHomoskedasticNoise",
    "TransformedHeteroskedasticNoise",
    "TransformedFixedGaussianNoise",
    "Logger", "TerminalLogger", "CSVLogger", "MLflowLogger",
    "NeuralMean",
    "GPModel", "MLP",
    "sample",
    "Trainer", "crps_normal", "make_crps_loss", "make_mll_loss", "make_loss",
    "Transformer", "FittedTransformer", "ParametricTransformer",
    "QuantileFittedTransformer", "Standardizer",
]

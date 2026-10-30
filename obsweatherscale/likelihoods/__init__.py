"""Custom likelihoods and noise models for obsweatherscale.

Provides a likelihood class that supports transformed target data by
correctly propagating the noise in transformed space, as well as several
noise models supporting the data transformation. Also provides a mll
implementation that fills NaN values instead of masking them.

Classes
-------
ExactMarginalLogLikelihoodFill
    Extension of :class:`~gpytorch.ExactMarginalLogLikelihood` with
    support for handling NaN values by filling.
TransformedGaussianLikelihood
    A Gaussian likelihood with a transformed noise model.
TransformedNoise
    Base class for noise models that incorporate a transformation of the
    target data.
TransformedFixedGaussianNoise
    Fixed (non-trainable) Gaussian noise across all inputs.
TransformedHeteroskedasticNoise
    A different, trainable noise variance for each data point
    (heteroskedastic).
TransformedHomoskedasticNoise
    Constant, trainable noise across all inputs (homoskedastic).
"""

from .transformed_likelihood import (
    TransformedGaussianLikelihood,
    ExactMarginalLogLikelihoodFill,
)
from .noise_models import (
    TransformedHomoskedasticNoise,
    TransformedHeteroskedasticNoise,
    TransformedFixedGaussianNoise,
)

__all__ = [
    "ExactMarginalLogLikelihoodFill",
    "TransformedGaussianLikelihood",
    "TransformedFixedGaussianNoise",
    "TransformedHeteroskedasticNoise",
    "TransformedHomoskedasticNoise",
]

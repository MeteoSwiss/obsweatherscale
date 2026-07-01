"""Transformations for input and output Gaussian Processes data.

This module provides a set of transformations that can be applied to
the input and output data of Gaussian Process models. These
transformations include fitted transformations that learn parameters
from data, and parametric transformations with fixed parameters.

Classes
-------
Transformer
    Abstract base class for all data transformations.
QuantileFittedTransformer
    Continuous approximation of a quantile transform.
Standardizer
    Standardization transformation (zero mean, unit variance).
"""

from .quantile_fitted_transformer import QuantileFittedTransformer
from .standardizer import Standardizer
from .transformer import Transformer

__all__ = ["QuantileFittedTransformer", "Standardizer", "Transformer"]

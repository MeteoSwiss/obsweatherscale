"""Trainer utilities for Gaussian Process models.

This module provides utilities for training and validating GPyTorch-
based Gaussian Process models, including a random state context manager
for reproducible evaluation, a ``Trainer`` class that encapsulates
the training loop logic, and several loss functions.

Classes
-------
Trainer
    Orchestrates training and validation of an ``ExactGP`` model.

Functions
---------
crps_normal
    Computes the Continuous Ranked Probability Score (CRPS) for a
    univariate normal distribution.
make_crps_loss
    Creates a CRPS loss function for normal distributions that handles
    missing values and optionally transforms the distribution.
make_mll_loss
    Creates a negative log-likelihood loss function for a multivariate
    normal distribution, optionally transformed by a likelihood
    function.
make_loss
    Factory function to create a loss function based on the specified
    loss type.
"""

from .losses import crps_normal, make_crps_loss, make_mll_loss, make_loss
from .trainer import Trainer

__all__ = [
    "Trainer", "crps_normal", "make_crps_loss", "make_mll_loss", "make_loss",
]

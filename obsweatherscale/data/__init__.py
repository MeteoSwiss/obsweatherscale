"""Datasets for Gaussian Process models.

This module provides an abstract base class for datasets used in
Gaussian Process models. It extends the PyTorch Dataset class with
additional functionality specific to Gaussian Processes.

Classes
-------
GPDataset
    Abstract base class for Gaussian Process datasets.
"""

from .dataset import GPDataset

__all__ = ["GPDataset"]

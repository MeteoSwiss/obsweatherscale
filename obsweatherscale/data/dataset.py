"""Datasets for Gaussian Process models.

Classes
-------
GPDataset
    Abstract base class for Gaussian Process datasets.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class GPDataset(Dataset, ABC):
    """Abstract base class for Gaussian Process datasets.

    Extends :class:`torch.utils.data.Dataset` with a standardized
    interface for datasets consumed by Gaussian Process models. Beyond
    the standard PyTorch ``__len__`` / ``__getitem__`` protocol,
    subclasses are expected to expose the full dataset (input tensor
    ``x`` and target tensor ``y``), and implement device transfer so
    that dataset tensors can be moved alongside a model.

    Notes
    -----
    Subclasses must implement all methods decorated with
    ``@abstractmethod``. Attempting to instantiate ``GPDataset``
    directly will raise :class:`TypeError`.

    Subclasses should call ``super().__init__()`` in their own
    ``__init__`` if they rely on any initialisation logic defined by
    :class:`torch.utils.data.Dataset`.

    See Also
    --------
    torch.utils.data.Dataset : PyTorch's base dataset class
    """

    @abstractmethod
    def __getitem__(self, index: int | list[int] | slice) -> Any:
        """Get a sample from the dataset at the specified index.

        Parameters
        ----------
        index : int | list[int] | slice
            Index of the sample to retrieve.

        Returns
        -------
        Any
            The sample at the specified index.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """

    @abstractmethod
    def get_dataset(self) -> tuple[torch.Tensor, ...]:
        """Get the entire dataset as tensors.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple of tensors representing the entire dataset.
            Typically contains input features and target values.
        """

    @abstractmethod
    def to(self, device: torch.device) -> None:
        """Move the dataset to the specified device.

        Parameters
        ----------
        device : torch.device
            The device to move the dataset to (e.g., CPU or CUDA).

        Returns
        -------
        self
            The dataset instance moved to the specified device.
        """

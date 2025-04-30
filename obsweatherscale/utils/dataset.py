from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class GPDataset(Dataset, ABC):
    """Abstract base class for Gaussian Process datasets.

    This class extends PyTorch's Dataset class to provide a standardized
    interface for datasets used in Gaussian Process models. It defines
    required methods for data access, transformation, and device
    management.

    All subclasses must implement the abstract methods defined here.

    See Also
    --------
    torch.utils.data.Dataset : PyTorch's base dataset class
    """

    @abstractmethod
    def __getitem__(self, index) -> Any:
        """Get a sample from the dataset at the specified index.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        Any
            The sample at the specified index.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        pass

    @abstractmethod
    def to(self, device: torch.device):
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
        pass

    @abstractmethod
    def get_dataset(self) -> tuple[torch.Tensor, ...]:
        """Get the entire dataset as tensors.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple of tensors representing the entire dataset.
            Typically contains input features and target values.
        """
        pass

    @property
    @abstractmethod
    def standardizer(self) -> Any:
        """Get the standardizer used for feature normalization.

        Returns
        -------
        Any
            The standardizer object used to normalize input features.
        """
        pass

    @property
    @abstractmethod
    def y_transformer(self) -> Any:
        """Get the transformer used for target variable transformation.

        Returns
        -------
        Any
            The transformer object used to transform target variables.
        """
        pass

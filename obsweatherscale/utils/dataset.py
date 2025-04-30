from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class GPDataset(Dataset, ABC):

    @abstractmethod
    def __getitem__(self, index) -> Any:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def to(self, device: torch.device):
        pass

    @abstractmethod
    def get_dataset(self) -> tuple[torch.Tensor, ...]:
        pass

    @property
    @abstractmethod
    def standardizer(self) -> Any:
        pass

    @property
    @abstractmethod
    def y_transformer(self) -> Any:
        pass

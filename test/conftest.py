import math

import pytest
import torch

from obsweatherscale.transformations import (
    Standardizer, QuantileFittedTransformer
)
from obsweatherscale.utils import GPDataset


# Create dataset
class MyDataset(GPDataset):
    def __init__(self, ds_x: torch.Tensor, ds_y: torch.Tensor) -> None:
        self.x = ds_x
        self.y = ds_y
        self.n_samples = ds_x.shape[0]

    def to(self, device: torch.device) -> None:
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self,
        idx: int | list[int] | slice
    ) -> tuple[torch.Tensor, ...]:
        return self.x[idx, ...], self.y[idx, ...]

    def get_dataset(self) -> tuple[torch.Tensor, ...]:
        return self.x, self.y


def create_test_data() -> dict[str, MyDataset]:
    # 1. Input data
    nx, ny, nt = 30, 20, 20  # e.g.: 30 x pts, 20 y pts, 10 timesteps
    x_coords = torch.linspace(0, 1, nx)
    y_coords = torch.linspace(0, 1, ny)
    timesteps = torch.linspace(0, 1, nt)

    time_grid, x_grid, y_grid = torch.meshgrid(
        timesteps, x_coords, y_coords, indexing='ij'
    )

    # train_x contains all predictors, here: x, y, and t grids (3 preds)
    # shape: (nx, ny, nt, npreds)
    ds_x = torch.stack([time_grid, x_grid, y_grid])

    # True function: a simplified weather surface
    # (sin variation in space/time)
    # shape: (nx, ny, nt)
    ds_y = (
        torch.sin(x_grid * (2 * math.pi)) *
        torch.cos(y_grid * (2 * math.pi)) +
        torch.sin(time_grid * (2 * math.pi)) +
        torch.randn(time_grid.size()) * math.sqrt(0.04)
    )

    # Reshape x and y to shape: (nt, ns, ...) where ns is the number
    # of spatial points
    ns = nx * ny
    ds_x = ds_x.reshape(nt, ns, 3)  # shape: (n_times, ns, npred)
    ds_y = ds_y.reshape(nt, ns)     # shape: (nt, ns)

    # Split into train and validation
    points_idx = torch.randperm(ns)
    train_frac_times = 0.7
    train_frac_points = 0.8
    train_points = points_idx[:int(train_frac_points * ns)]
    val_points = points_idx[int(train_frac_points * ns):]

    train_x = ds_x[:int(train_frac_times * nt), train_points]
    train_y = ds_y[:int(train_frac_times * nt), train_points]
    val_x = ds_x[int(train_frac_times * nt):]
    val_y = ds_y[int(train_frac_times * nt):]

    # Normalize data
    standardizer = Standardizer(train_x)
    y_transformer = QuantileFittedTransformer()

    train_x = standardizer.transform(train_x)
    train_y = y_transformer.transform(train_y)
    val_x = standardizer.transform(val_x)
    val_y = y_transformer.transform(val_y)

    ds_train = MyDataset(train_x, train_y)
    ds_val_c = MyDataset(val_x[:, train_points], val_y[:, train_points])
    ds_val_t = MyDataset(val_x[:, val_points], val_y[:, val_points])

    res = {}
    res["train"] = ds_train
    res["val_context"] = ds_val_c
    res["val_target"] = ds_val_t
    return res

test_data = create_test_data()


@pytest.fixture
def dataset_train() -> MyDataset:
    return test_data["train"]


@pytest.fixture
def dataset_val_c() -> MyDataset:
    return test_data["val_context"]


@pytest.fixture
def dataset_val_t() -> MyDataset:
    return test_data["val_target"]

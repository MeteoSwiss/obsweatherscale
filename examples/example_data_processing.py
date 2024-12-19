from pathlib import Path
from typing import Optional, Union

import torch
import xarray as xr

from obsweatherscale.data_io import open_zarr_file
from obsweatherscale.transformations import (
    QuantileFittedTransformer, Standardizer, Transformer
)
from obsweatherscale.utils import coords_to_station
from obsweatherscale.utils.dataset import GPDataset


MF_INPUTS = [
    "weather:wind_speed_of_gust",
    "weather:wind_speed_of_gust_p1h",
    "weather:wind_speed_of_gust_m1h",
    "weather:sx_500m",
    "static:model_elevation_difference",
    "static:tpi_500m",
    "static:tpi_2000m",
    "static:elevation",
    "static:we_derivative_2000m",
    "static:sn_derivative_2000m",
    "time:sin_hourofday",
    "time:cos_hourofday"
]

SPATIAL_INPUTS = [
    "static:elevation",
    "static:easting",
    "static:northing",
]

K_INPUTS = [
    "weather:eastward_wind",
    "weather:northward_wind",
    "static:easting",
    "static:northing",
    "static:tpi_500m",
    "static:tpi_2000m",
    "static:elevation",
    "static:we_derivative_2000m",
    "static:sn_derivative_2000m",
    "time:sin_hourofday",
    "time:cos_hourofday",
]

INPUTS = sorted(
    list(set(MF_INPUTS) | set(K_INPUTS) | set(SPATIAL_INPUTS))
)


class WindDataset(GPDataset):
    """ Generic dataset derived for wind specific needs"""
    def __init__(
        self,
        ds_x: xr.Dataset,
        ds_y: xr.Dataset,
        standardizer: Optional[Standardizer] = None,
        y_transformer: Optional[Transformer] = None,
    ):
        # Transform dataset into dataarray
        var_name = "variable"  # new dimension name
        x = self.reshape_dataset(ds_x, var_name)
        y = self.reshape_dataset(ds_y, var_name)

        # Save dimensions and coords before transforming data to tensor
        self.n_samples = x.shape[0]
        self.dims = x.dims
        self.coords = {
            key: val for key, val in x.coords.items() if key != var_name
        }
        self.dim_to_idx = {dim: idx for idx, dim in enumerate(x.dims)}
        self.var_to_idx = {
            dim: idx for idx, dim in enumerate(x.coords[var_name].values)
        }

        # Normalize inputs
        x = torch.tensor(x.compute().values, dtype=torch.float32)
        if standardizer is None:
            standardizer = self.create_standardizer(x)
        self._standardizer = standardizer
        self.x = self._standardizer.transform(x).contiguous()

        # Normalize outputs
        y = torch.tensor(y.values)
        if y_transformer is None:
            y_transformer = QuantileFittedTransformer()
        self._y_transformer = y_transformer
        self.y = self._y_transformer.transform(y).squeeze(-1).contiguous()

    @property
    def standardizer(self) -> Standardizer:
        return self._standardizer

    @standardizer.setter
    def standardizer(self, value: Standardizer):
        self._standardizer = value

    @property
    def y_transformer(self) -> Transformer:
        return self._y_transformer

    @y_transformer.setter
    def y_transformer(self, value: Transformer):
        self._y_transformer = value

    def to(self, device: torch.device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return self.n_samples

    def __getitem__(
        self,
        idx: Union[int, list[int], slice]
    ) -> tuple[torch.Tensor, ...]:
        return self.x[idx, ...], self.y[idx, ...]

    def get_dataset(self) -> tuple[torch.Tensor, ...]:
        return self.x, self.y

    def reshape_dataset(self, ds, var_name):
        return ds.to_array(var_name).transpose("time", "station", var_name)

    def create_standardizer(
        self,
        x: torch.Tensor,
        mean_vars: Optional[list[str]] = None,
    ) -> Standardizer:
        if mean_vars is None:
            mean_vars = ["time", "station"]

        var_idxs = tuple(self.dim_to_idx[var] for var in mean_vars)
        easting_idx = self.var_to_idx['static:easting']
        northing_idx = self.var_to_idx['static:northing']

        standardizer = Standardizer(x, variables=var_idxs)
        standardizer.std[easting_idx] = standardizer.std[northing_idx]

        return standardizer


class WindDatasetTarget(GPDataset):
    """ Target dataset derived for wind specific needs"""
    def __init__(
        self,
        ds_x: xr.Dataset,
        standardizer: Standardizer,
    ):
        # Transform dataset into dataarray
        var_name = "variable"  # new dimension name
        ds_x = coords_to_station(ds_x)
        x = self.reshape_dataset(ds_x, var_name)

        # Save dimensions and coords before transforming data to tensor
        self.n_samples = x.shape[0]
        self.dims = x.dims
        self.coords = {
            key: val for key, val in x.coords.items() if key != var_name
        }

        # Normalize inputs
        x = torch.tensor(x.compute().values, dtype=torch.float32)
        self._standardizer = standardizer
        self.x = self._standardizer.transform(x).contiguous()

    @property
    def standardizer(self) -> Standardizer:
        return self._standardizer

    @standardizer.setter
    def standardizer(self, value: Standardizer):
        self._standardizer = value

    @property
    def y_transformer(self) -> None:
        return None

    def to(self, device: torch.device):
        self.x = self.x.to(device)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: Union[int, list[int], slice]) -> torch.Tensor:
        return self.x[idx, ...]

    def get_dataset(self) -> torch.Tensor:
        return self.x

    def reshape_dataset(self, ds, var_name):
        return ds.to_array(var_name).transpose("time", "station", var_name)


def get_dataset(
    x_filename: Path,
    y_filename: Path,
    inputs: Optional[list[str]],
    targets: Optional[list[str]],
    standardizer: Optional[Standardizer] = None,
    y_transformer: Optional[Transformer] = None,
    time_slice: slice = slice(None)
) -> WindDataset:
    x = open_zarr_file(x_filename, data_key=inputs, time_slice=time_slice)
    y = open_zarr_file(y_filename, data_key=targets, time_slice=time_slice)

    return WindDataset(
        x, y, standardizer=standardizer, y_transformer=y_transformer
    )


def get_training_data(
    data_dir: Path,
    x_train_filename: str,
    y_train_filename: str,
    x_val_c_filename: str,
    y_val_c_filename: str,
    x_val_t_filename: str,
    y_val_t_filename: str,
    inputs: Optional[list[str]],
    targets: Optional[list[str]],
) -> tuple[WindDataset, ...]:

    dataset_train = get_dataset(
        x_filename=data_dir / x_train_filename,
        y_filename=data_dir / y_train_filename,
        inputs=inputs,
        targets=targets,
    )
    dataset_val_c = get_dataset(
        x_filename=data_dir / x_val_c_filename,
        y_filename=data_dir / y_val_c_filename,
        inputs=inputs,
        targets=targets,
        standardizer=dataset_train.standardizer,
        y_transformer=dataset_train.y_transformer
    )
    dataset_val_t = get_dataset(
        x_filename=data_dir / x_val_t_filename,
        y_filename=data_dir / y_val_t_filename,
        inputs=inputs,
        targets=targets,
        standardizer=dataset_train.standardizer,
        y_transformer=dataset_train.y_transformer
    )

    return dataset_train, dataset_val_c, dataset_val_t

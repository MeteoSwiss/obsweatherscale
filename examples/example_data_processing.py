from pathlib import Path

import torch
import xarray as xr

from obsweatherscale.data_io.zarr_utils import open_zarr_file
from obsweatherscale.transformations import (QuantileFittedTransformer,
                                             Standardizer,
                                             Transformer)

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


class WindDataset(torch.utils.data.Dataset):
    """ Generic dataset derived for wind specific needs"""
    def __init__(
        self,
        ds_x: xr.Dataset,
        ds_y: xr.Dataset,
        standardizer: Standardizer = None,
        y_transformer: Transformer = None,
    ):
        # Dimension names
        self.realization_name = "realization"
        self.var_name = "variable"

        # Transform dataset into dataarray using "variable" as 3rd dimension
        x = ds_x.to_array(self.var_name).transpose(
            "time", "station", self.var_name
        )
        y = ds_y.to_array(self.var_name).transpose(
            "time", "station", self.var_name
        )

        # Save dimensions and coords before transforming data to tensor
        self.n_samples = x.shape[0]
        self.n_stations = x.shape[1]
        self.dims = x.dims
        self.coords = {
            key: val for key, val in x.coords.items()
            if key != self.var_name
        }
        self.dim_to_idx = {dim: idx for idx, dim in enumerate(x.dims)}
        self.var_to_idx = {
            dim: idx for idx, dim
            in enumerate(x.coords[self.var_name].values)
        }

        # Normalize inputs
        x = torch.tensor(x.compute().values, dtype=torch.float32)
        if standardizer is None:
            standardizer = self.create_standardizer(x)
        self.standardizer = standardizer
        self.x = self.standardizer.transform(x).contiguous()

        # Normalize outputs
        y = torch.tensor(y.values)
        if y_transformer is None:
            y_transformer = QuantileFittedTransformer()
        self.y_transformer = y_transformer
        self.y = self.y_transformer.transform(
            y
        ).squeeze(-1).contiguous()
    
    def create_standardizer(
        self,
        x: torch.Tensor,
        mean_vars: list[str] = ["time", "station"]
    ) -> Standardizer:
        var_idxs = tuple(self.dim_to_idx[var] for var in mean_vars)
        easting_idx = self.var_to_idx['static:easting']
        northing_idx = self.var_to_idx['static:northing']

        standardizer = Standardizer(x, variables=var_idxs)
        standardizer.std[easting_idx] = standardizer.std[northing_idx]

        return standardizer

    def to(self, device: torch.device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return self.n_samples

    def __getitem__(
        self,
        idx: list[int] | int
    ) -> tuple[torch.Tensor]:
        return self.x[idx, ...], self.y[idx, ...]

    def get_dataset(self) -> tuple[torch.Tensor]:
        return self.x, self.y

    def wrap_pred(self, pred: torch.Tensor, name="") -> xr.DataArray:
        # Create dimensions
        dims = self.dims
        if len(pred.shape) > 3:
            dims = (self.realization_name,) + dims
        
        # Transform back to DataArray
        pred = xr.DataArray(pred.detach(), self.coords, dims, name=name)

        return pred


def get_training_data(
    data_dir: Path,
    x_train_filename: str,
    y_train_filename: str,
    x_val_c_filename: str,
    y_val_c_filename: str,
    x_val_t_filename: str,
    y_val_t_filename: str,
    inputs: list[str] | None,
    targets: list[str] | None,
) -> tuple[torch.utils.data.Dataset]:

    x = open_zarr_file(data_dir / x_train_filename, data_key=inputs)
    y = open_zarr_file(data_dir / y_train_filename, targets)
    x_val_c = open_zarr_file(data_dir / x_val_c_filename, data_key=inputs)
    y_val_c = open_zarr_file(data_dir / y_val_c_filename, data_key=targets)
    x_val_t = open_zarr_file(data_dir / x_val_t_filename, data_key=inputs)
    y_val_t = open_zarr_file(data_dir / y_val_t_filename, data_key=targets)

    dataset_train = WindDataset(x, y)
    dataset_val_c = WindDataset(
        x_val_c,
        y_val_c,
        standardizer=dataset_train.standardizer,
        y_transformer=dataset_train.y_transformer
    )
    dataset_val_t = WindDataset(
        x_val_t,
        y_val_t,
        standardizer=dataset_train.standardizer,
        y_transformer=dataset_train.y_transformer
    )

    return dataset_train, dataset_val_c, dataset_val_t


def get_eval_data(
    data_dir: Path,
    x_context_filename: str,
    y_context_filename: str,
    x_target_filename: str,
    y_target_filename: str,
    inputs: list[str] | None,
    targets: list[str] | None,
    standardizer: Standardizer, 
    y_transformer: QuantileFittedTransformer, 
    time_slice: slice = slice(None)
) -> tuple[torch.utils.data.Dataset]:
    x_target = open_zarr_file(
        data_dir / x_target_filename,
        data_key=inputs, time_slice=time_slice
    )
    x_context = open_zarr_file(
        data_dir / x_context_filename,
        data_key=inputs, time_slice=time_slice
    )
    y_target = open_zarr_file(
        data_dir / y_target_filename,
        data_key=targets, time_slice=time_slice
    )
    y_context = open_zarr_file(
        data_dir / y_context_filename,
        data_key=targets, time_slice=time_slice
    )

    dataset_context = WindDataset(
        x_context,
        y_context,
        standardizer=standardizer,
        y_transformer=y_transformer
    )
    dataset_target = WindDataset(
        x_target,
        y_target,
        standardizer=standardizer,
        y_transformer=y_transformer
    )

    return dataset_context, dataset_target


def wrap_and_denormalize(
        *data: tuple[torch.Tensor],
        wrap_fun,
        denormalize_fun,
        name: str
    ) -> xr.DataArray:
    result = []
    for tensor in data:
        tensor = denormalize_fun(tensor)
        tensor = wrap_fun(tensor, name=name)
        tensor = tensor.to_dataset(name=name)
        result.append(tensor)
    return tuple(result)

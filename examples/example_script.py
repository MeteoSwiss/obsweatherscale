# %%
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from gpytorch.mlls import ExactMarginalLogLikelihood

from obsweatherscale.data_io.zarr_utils import open_zarr_file
from obsweatherscale.kernels import NeuralKernel, ScaledRBFKernel
from obsweatherscale.likelihoods import TransformedGaussianLikelihood
from obsweatherscale.likelihoods.noise_models import TransformedFixedGaussianNoise
from obsweatherscale.means import NeuralMean
from obsweatherscale.models import GPModel, MLP
from obsweatherscale.training.loss_functions import crps_normal_loss_fct, mll_loss_fct
from obsweatherscale.training.training import train_model
from obsweatherscale.transformations import QuantileFittedTransformer, Standardizer, Transformer


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

INPUTS = sorted(list(set(MF_INPUTS) | set(K_INPUTS) | set(SPATIAL_INPUTS)))

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

        # Transform dataset into dataarray using "variable" as a 3rd dimension
        x = ds_x.to_array(self.var_name).transpose("time", "station", self.var_name)
        y = ds_y.to_array(self.var_name).transpose("time", "station", self.var_name)

        # Save dimensions and coords before transforming data to tensor
        self.n_samples = x.shape[0]
        self.n_stations = x.shape[1]
        self.dims = x.dims
        self.coords = {key: val for key, val in x.coords.items()
                       if key != self.var_name}
        self.dim_to_idx = {dim: idx for idx, dim in enumerate(x.dims)}
        self.var_to_idx = {dim: idx for idx, dim
                           in enumerate(x.coords[self.var_name].values)}

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
        self.y = self.y_transformer.transform(y).squeeze(-1).contiguous()
    
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

    def __getitem__(self, idx: list[int] | int) -> tuple[torch.Tensor]:
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
    dataset_val_c = WindDataset(x_val_c, y_val_c,
                                standardizer=dataset_train.standardizer,
                                y_transformer=dataset_train.y_transformer)
    dataset_val_t = WindDataset(x_val_t, y_val_t,
                                standardizer=dataset_train.standardizer,
                                y_transformer=dataset_train.y_transformer)

    return dataset_train, dataset_val_c, dataset_val_t
    

def main(config):
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_filename = "model"

    start = time.time()
    torch.manual_seed(config.seed)

    # Get data
    dataset_train, dataset_val_c, dataset_val_t = get_training_data(config.data_dir,
                                                                    config.x_train_filename,
                                                                    config.y_train_filename,
                                                                    config.x_val_c_filename,
                                                                    config.y_val_c_filename,
                                                                    config.x_val_t_filename,
                                                                    config.y_val_t_filename,
                                                                    config.inputs,
                                                                    config.targets)
    train_x, train_y = dataset_train.get_dataset()

    # Initialize device
    device = torch.device("cuda:0") if torch.cuda.is_available() and config.use_gpu else torch.device("cpu")

    # Initialize likelihood
    transformer = QuantileFittedTransformer()
    noise_model = TransformedFixedGaussianNoise(transformer, obs_noise_var=torch.tensor(1.0))
    likelihood = TransformedGaussianLikelihood(noise_model)

    # Initialize model
    # optim lengthscale: [0.25992706, 0.1681214, 0.08974447]
    spatial_kernel = ScaledRBFKernel(lengthscale=torch.tensor([0.25992706, 0.1681214, 0.08974447]),
                                     active_dims=[INPUTS.index(v) for v in SPATIAL_INPUTS],
                                     train_lengthscale=False)
    neural_kernel = NeuralKernel(net=MLP(len(K_INPUTS), [32, 32], 4),
                                    kernel=ScaledRBFKernel(),
                                    active_dims=[INPUTS.index(v) for v in K_INPUTS])
    kernel = spatial_kernel * neural_kernel
    mean_function = NeuralMean(net=MLP(len(MF_INPUTS), [32, 32], 1),
                               active_dims=[INPUTS.index(v) for v in MF_INPUTS])

    model = GPModel(mean_function, kernel, train_x, train_y, likelihood)

    # Initialize optimizer and loss
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
        ],
        lr=config.learning_rate)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    train_loss_fct = mll_loss_fct(mll)
    val_loss_fct = crps_normal_loss_fct()

    # Train
    model, train_progress = train_model(dataset_train, dataset_val_c, dataset_val_t,
                                        model, likelihood,
                                        train_loss_fct, val_loss_fct,
                                        device, optimizer,
                                        config.batch_size,
                                        output_dir, model_filename, config.n_iter,
                                        random_masking=True, seed=config.seed,
                                        prec_size=config.prec_size)
    
    # Identify and label the best performing model
    best_model_idx = np.argmin(train_progress["val loss"])
    best_model_path = config.output_dir / "{}_iter_{}".format(model_filename, best_model_idx)
    model.load_state_dict(torch.load(best_model_path))

    # Save
    torch.save(model.state_dict(), config.output_dir / "best_model")
    with open(output_dir / "train_progress.pkl", 'wb') as train_progress_path:
        pickle.dump(train_progress, train_progress_path)
    with open(output_dir / config.standardizer_filename, 'wb') as standardizer_path:
        pickle.dump(dataset_train.standardizer, standardizer_path, pickle.HIGHEST_PROTOCOL)
    with open(output_dir / config.y_transformer_filename, 'wb') as y_transformer_path:
        pickle.dump(dataset_train.y_transformer, y_transformer_path, pickle.HIGHEST_PROTOCOL)

    # Free GPU
    torch.cuda.empty_cache()
    stop = time.time()
    print(f"Total time: {stop - start:.3f} s / {(stop - start)/60:.3f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=Path('/', 'scratch', 'mch', 'illornsj',
                                     'data', 'cosmo-1e'))
    parser.add_argument('--output_dir', type=str,
                        default=Path('/', 'scratch', 'mch', 'illornsj',
                                     'data', 'experiments',
                                     'spatial_deep_kernel',
                                     'artifacts'))
    parser.add_argument('--x_train_filename', type=str, default="x_train_replicate.zarr")
    parser.add_argument('--y_train_filename', type=str, default="y_train_replicate.zarr")
    parser.add_argument('--x_val_c_filename', type=str, default="x_val_context_replicate.zarr")
    parser.add_argument('--y_val_c_filename', type=str, default="y_val_context_replicate.zarr")
    parser.add_argument('--x_val_t_filename', type=str, default="x_val_target_replicate.zarr")
    parser.add_argument('--y_val_t_filename', type=str, default="y_val_target_replicate.zarr")
    parser.add_argument('--standardizer_filename', type=str, default="standardizer.pkl")
    parser.add_argument('--y_transformer_filename', type=str, default="y_transformer.pkl")
    parser.add_argument('--inputs', type=list, default=INPUTS)
    parser.add_argument('--targets', type=list, default=["weather:wind_speed_of_gust"])
    parser.add_argument('--random_masking', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_iter', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--prec_size', type=int, default=100)

    args, _ = parser.parse_known_args()

    main(args)

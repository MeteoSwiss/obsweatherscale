# %%
import math
import time
from typing import Union

import torch
from torch.optim.adam import Adam

from obsweatherscale.kernels import NeuralKernel, ScaledRBFKernel
from obsweatherscale.likelihoods import (
    TransformedGaussianLikelihood, ExactMarginalLogLikelihoodFill
)
from obsweatherscale.likelihoods.noise_models import TransformedFixedGaussianNoise
from obsweatherscale.means import NeuralMean
from obsweatherscale.models import GPModel, MLP
from obsweatherscale.training import (
    crps_normal_loss_fct, mll_loss_fct, Trainer
)
from obsweatherscale.transformations import (
    QuantileFittedTransformer, Standardizer
)
from obsweatherscale.utils import init_device, GPDataset


def main():
    seed = 123
    start = time.time()

    # 1. Input data
    nx, ny, nt = 30, 20, 10  # e.g.: 30 x points, 20 y points, 10 timesteps
    x_coords = torch.linspace(0, 1, nx)
    y_coords = torch.linspace(0, 1, ny)
    timesteps = torch.linspace(0, 1, nt)

    time_grid, x_grid, y_grid = torch.meshgrid(
        timesteps, x_coords, y_coords, indexing='ij'
    )

    # train_x contains all predictors, here: x, y, and t grids (3 predictors)
    # shape: (nx, ny, nt, npred)
    ds_x = torch.stack([time_grid, x_grid, y_grid])
    npred = ds_x.shape[-1] # number of predictors

    # True function: a simplified weather surface (sin variation in space/time)
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

    train_x = ds_x[:int(train_frac_times * nt), train_points]  # shape: (nt_train, ns_train, npred)
    train_y = ds_y[:int(train_frac_times * nt), train_points]     # shape: (nt_train, ns_train)
    val_x = ds_x[int(train_frac_times * nt):]    # shape: (nt_val, ns, npred)
    val_y = ds_y[int(train_frac_times * nt):]       # shape: (nt_val, ns)

    # Normalize data
    standardizer = Standardizer(train_x)
    y_transformer = QuantileFittedTransformer()

    train_x = standardizer.transform(train_x)
    train_y = y_transformer.transform(train_y)
    val_x = standardizer.transform(val_x)
    val_y = y_transformer.transform(val_y)

    # Create dataset
    class MyDataset(GPDataset):
        def __init__(self, ds_x: torch.Tensor, ds_y: torch.Tensor):
            self.x = ds_x
            self.y = ds_y
            self.n_samples = ds_x.shape[0]

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

    dataset_train = MyDataset(train_x, train_y)
    dataset_val_c = MyDataset(val_x[:, train_points], val_y[:, train_points])
    dataset_val_t = MyDataset(val_x[:, val_points], val_y[:, val_points])

    # Initialize device
    device = init_device(gpu=None, use_gpu=True)

    # Initialize likelihood
    # Noise model chosen to have a non-trainable constant variance of 1
    # across all data points
    noise_model = TransformedFixedGaussianNoise(y_transformer, obs_noise_var=1)
    likelihood = TransformedGaussianLikelihood(noise_model)

    # Initialize model
    torch.manual_seed(seed)
    mean_function = NeuralMean(net=MLP(dimensions=[3, 32, 32, 1]))

    torch.manual_seed(seed)
    kernel = NeuralKernel(
        net=MLP(dimensions=[3, 32, 32, 4]),
        kernel=ScaledRBFKernel()
    )

    model = GPModel(train_x, train_y, likelihood, mean_function, kernel)

    # Initialize optimizer and loss functions
    optimizer = Adam(
        [{'params': model.parameters()}, {'params': likelihood.parameters()}],
        lr=0.005
    )
    mll = ExactMarginalLogLikelihoodFill(likelihood, model)
    train_loss_fct = mll_loss_fct(mll)
    val_loss_fct = crps_normal_loss_fct(likelihood)

    # Train
    trainer = Trainer(
        model, likelihood, train_loss_fct, val_loss_fct, device, optimizer
    )
    model, train_progress = trainer.fit(
        dataset_train,
        dataset_val_c,
        dataset_val_t,
        batch_size=2,
        n_iter=100,
        random_masking=True,
        seed=seed,
        verbose=True
    )

    # Free GPU
    torch.cuda.empty_cache()

    stop = time.time()
    print(
        f"Total time: {stop - start:.3f} s / {(stop - start)/60:.3f} min",
        flush=True
    )


if __name__ == "__main__":
    main()

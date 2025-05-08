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
from obsweatherscale.likelihoods.noise_models import (
    TransformedFixedGaussianNoise
)
from obsweatherscale.means import NeuralMean
from obsweatherscale.models import GPModel, MLP
from obsweatherscale.training import (
    crps_normal_loss_fct, mll_loss_fct, Trainer
)
from obsweatherscale.transformations import (
    QuantileFittedTransformer, Standardizer
)
from obsweatherscale.utils import GPDataset


def main():
    seed = 123
    start = time.time()

    ##### Data ####
    # e.g. 100 stations, 10 timesteps, on a [0, 1] x [0, 1] x-y grid

    def true_signal(x, y, t): # simulated weather signal
        return (
            torch.sin(math.pi * x) *
            torch.cos(math.pi * y) *
            torch.sin(2 * math.pi * t / t.shape[0])
        )

    n_points, n_times, noise_var = 100, 10, 0.1
    t = torch.linspace(0, n_times - 1, n_times)
    x_coords, y_coords = torch.rand(n_points), torch.rand(n_points)

    x_coords = x_coords.unsqueeze(0).expand(n_times, -1) # [n_times, n_points]
    y_coords = y_coords.unsqueeze(0).expand(n_times, -1) # [n_times, n_points]
    t = t.unsqueeze(-1).expand(-1, n_points)             # [n_times, n_points]

    ds_x = torch.stack([x_coords, y_coords, t], dim=-1)
    ds_y = true_signal \
        + math.sqrt(noise_var) * torch.randn_like(x_coords)

    # Split into train and validation
    frac_t_train = 0.7
    frac_p_train = 0.8
    nt_train, np_train = int(frac_t_train*n_times), int(frac_p_train*n_points)

    points_idx = torch.randperm(n_points)
    train_points = points_idx[:np_train]
    val_points = points_idx[np_train:]

    train_x = ds_x[:nt_train, train_points] # [nt_train, np_train, n_preds]
    train_y = ds_y[:nt_train, train_points] # [nt_train, np_train]
    val_x = ds_x[nt_train:]     # [nt_val, n_points, n_preds]
    val_y = ds_y[nt_train:]     # [nt_val, n_points]

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

    #### Initialize model ####

    # Likelihood and noise model
    # Non-trainable constant variance across all data points
    noise_model = TransformedFixedGaussianNoise(y_transformer, noise_var)
    likelihood = TransformedGaussianLikelihood(noise_model)

    # GP model
    torch.manual_seed(seed)
    mean_function = NeuralMean(net=MLP(dimensions=[3, 32, 32, 1]))

    torch.manual_seed(seed)
    kernel = NeuralKernel(
        net=MLP(dimensions=[3, 32, 32, 4]),
        kernel=ScaledRBFKernel()
    )

    model = GPModel(train_x, train_y, likelihood, mean_function, kernel)

    #### Loss functions ####
    mll = ExactMarginalLogLikelihoodFill(likelihood, model)
    train_loss_fct = mll_loss_fct(mll)
    val_loss_fct = crps_normal_loss_fct(likelihood)

    #### Train ####
    optimizer = Adam(
        [{'params': model.parameters()}, {'params': likelihood.parameters()}],
        lr=0.005
    )

    device = torch.device("cuda")

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

    #### Free GPU ####
    torch.cuda.empty_cache()

    #### Log ####
    stop = time.time()
    print(
        f"Total time: {stop - start:.3f} s / {(stop - start)/60:.3f} min",
        flush=True
    )


if __name__ == "__main__":
    main()

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


# Custom dataset inheriting from GPDataset
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


def generate_toy_data(n_stations, n_times, noise_var):
    # Simulated true weather signal
    def true_signal(x, y, t):
        return (
            torch.sin(math.pi * x) *
            torch.cos(math.pi * y) *
            torch.sin(2 * math.pi * t / t.shape[0])
        )

    t = torch.linspace(0, n_times - 1, n_times)
    x_coords, y_coords = torch.rand(n_stations), torch.rand(n_stations)

    x_coords = x_coords.unsqueeze(0).expand(n_times, -1) # [n_t, n_stations]
    y_coords = y_coords.unsqueeze(0).expand(n_times, -1) # [n_t, n_stations]
    t = t.unsqueeze(-1).expand(-1, n_stations)           # [n_t, n_stations]

    ds_x = torch.stack([x_coords, y_coords, t], dim=-1)
    ds_y = true_signal(x_coords, y_coords, t) \
        + math.sqrt(noise_var) * torch.randn_like(x_coords)

    return ds_x, ds_y


def split_data(
    ds_x: torch.Tensor,
    ds_y: torch.Tensor,
    frac_t_train: float = 0.7,
    frac_s_train: float = 0.8
) -> dict[str, dict[str, torch.Tensor]]:
    n_times, n_stations = ds_x.shape

    nt_train = int(frac_t_train * n_times)
    ns_train = int(frac_s_train * n_stations)

    stations_idx = torch.randperm(n_stations)
    train_stations = stations_idx[:ns_train]
    val_stations = stations_idx[ns_train:]

    data: dict[str, dict] = {'train': {}, 'val_c': {}, 'val_t': {}}

    data['train']['x'] = ds_x[:nt_train, train_stations, ...]
    data['train']['y'] = ds_y[:nt_train, train_stations]
    data['val_c']['x'] = ds_x[nt_train:, train_stations, ...]
    data['val_c']['y'] = ds_y[nt_train:, train_stations]
    data['val_t']['x'] = ds_x[nt_train:, val_stations, ...]
    data['val_t']['y'] = ds_y[nt_train:, val_stations]

    return data


def main():
    seed = 123
    start = time.time()

    ##### Data ####
    # e.g. 100 stations, 10 timesteps, on a [0, 1] x [0, 1] x-y grid
    n_stations, n_times, noise_var = 100, 10, 0.1

    ds_x, ds_y = generate_toy_data(n_stations, n_times, noise_var)

    # Split into train and validation (context and target)
    data = split_data(ds_x, ds_y)

    # Normalize
    standardizer = Standardizer(data['train']['x'])
    y_transformer = QuantileFittedTransformer()

    data['train']['x'] = standardizer.transform(data['train']['x'])
    data['train']['y'] = y_transformer.transform(data['train']['y'])
    data['val_c']['x'] = standardizer.transform(data['val_c']['x'])
    data['val_c']['y'] = y_transformer.transform(data['val_c']['y'])
    data['val_t']['x'] = standardizer.transform(data['val_t']['x'])
    data['val_t']['y'] = y_transformer.transform(data['val_t']['y'])

    # Initialize datasets and further split val into context and target
    dataset_train = MyDataset(data['train']['x'], data['train']['y'])
    dataset_val_c = MyDataset(data['val_c']['x'], data['val_c']['y'])
    dataset_val_t = MyDataset(data['val_t']['x'], data['val_t']['y'])


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

    model = GPModel(
        data['train']['x'], data['train']['y'],
        likelihood, mean_function, kernel
    )

    #### Loss functions ####
    mll = ExactMarginalLogLikelihoodFill(likelihood, model)
    train_loss_fct = mll_loss_fct(mll)
    val_loss_fct = crps_normal_loss_fct(likelihood)

    #### Train ####
    optimizer = Adam(
        [{'params': model.parameters()}, {'params': likelihood.parameters()}],
        lr=0.005
    )

    if torch.cuda.is_available():
        torch.cuda.init()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

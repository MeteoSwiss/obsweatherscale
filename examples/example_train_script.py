# %%
import math
from typing import TypeAlias

import torch
from torch.optim.adam import Adam

import obsweatherscale as ows


# Custom dataset inheriting from GPDataset
DataDict: TypeAlias = dict[str, dict[str, torch.Tensor]]


class MyDataset(ows.GPDataset):
    def __init__(self, ds_x: torch.Tensor, ds_y: torch.Tensor) -> None:
        self.x = ds_x
        self.y = ds_y
        self.n_samples = ds_x.shape[0]

    def __getitem__(
        self,
        idx: int | list[int] | slice,
    ) -> tuple[torch.Tensor, ...]:
        return self.x[idx, ...], self.y[idx, ...]

    def __len__(self) -> int:
        return self.n_samples

    def get_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y

    def to(self, device: torch.device) -> None:
        self.x = self.x.to(device)
        self.y = self.y.to(device)


def generate_toy_data(
    n_stations: int,
    n_times: int,
    noise_var: float
) -> tuple[torch.Tensor, torch.Tensor]:
    def true_signal(
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Simulated true weather signal."""
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
) -> DataDict:
    n_times, n_stations, _ = ds_x.shape

    nt_train = int(frac_t_train * n_times)
    ns_train = int(frac_s_train * n_stations)

    stations_idx = torch.randperm(n_stations)
    train_stations = stations_idx[:ns_train]
    val_stations = stations_idx[ns_train:]

    splits: dict[str, dict[str, slice | torch.Tensor]] = {
        'train': {'time': slice(0, nt_train), 'station': train_stations},
        'val_c': {'time': slice(nt_train, None), 'station': train_stations},
        'val_t': {'time': slice(nt_train, None), 'station': val_stations},
    }

    data: DataDict = {}
    for split, idx in splits.items():
        data[split]['x'] = ds_x[idx['time'], idx['station'], ...]
        data[split]['y'] = ds_y[idx['time'], idx['station']]

    return data


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.init()
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    seed = 123

    ##### Data ####
    # Generate toy data, e.g. 100 stations, 10 timesteps on a [0,1]x[0,1] grid
    n_stations, n_times, noise_var = 100, 10, 0.1
    ds_x, ds_y = generate_toy_data(n_stations, n_times, noise_var)

    # Split into train and validation (context and target)
    raw_data = split_data(ds_x, ds_y)

    # Normalize
    standardizer = ows.Standardizer(raw_data['train']['x'])
    y_transformer = ows.QuantileFittedTransformer()

    data: DataDict = {}
    for split in ['train', 'val_c', 'val_t']:
        data[split]['x'] = standardizer.transform(raw_data[split]['x'])
        data[split]['y'] = y_transformer.transform(raw_data[split]['y'])

    # Initialize datasets
    dataset_train = MyDataset(data['train']['x'], data['train']['y'])
    dataset_val_c = MyDataset(data['val_c']['x'], data['val_c']['y'])
    dataset_val_t = MyDataset(data['val_t']['x'], data['val_t']['y'])

    #### Initialize model ####
    # Likelihood and noise model (non-trainable constant variance
    # across all data points)
    noise_model = ows.TransformedFixedGaussianNoise(y_transformer, noise_var)
    likelihood = ows.TransformedGaussianLikelihood(noise_covar=noise_model)

    # GP model
    torch.manual_seed(seed)
    mean_function = ows.NeuralMean(net=ows.MLP(dimensions=[3, 32, 32, 1]))

    torch.manual_seed(seed)
    kernel = ows.NeuralKernel(
        net=ows.MLP(dimensions=[3, 32, 32, 4]),
        kernel=ows.ScaledRBFKernel()
    )

    model = ows.GPModel(
        mean_function,
        kernel,
        likelihood,
        *dataset_train.get_dataset(), # train_x, train_y
    )

    #### Loss functions ####
    mll = ows.ExactMarginalLogLikelihoodFill(likelihood, model)
    train_loss_fct = ows.make_mll_loss(mll)
    val_loss_fct = ows.make_crps_loss(likelihood)

    #### Train ####
    device = get_device()

    optimizer = Adam(
        [{'params': model.parameters()}, {'params': likelihood.parameters()}],
        lr=0.005,
    )

    trainer = ows.Trainer(
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
        verbose=True,
    )

    # Get the iteration of best model
    min_val_loss = min(train_progress["val loss"])
    best_val_loss_idx = train_progress["val loss"].index(min_val_loss)

    print(
        f"Best model found at "
        f"iteration {train_progress["iter"][best_val_loss_idx]} with "
        f"validation loss: {min_val_loss:.4f}"
    )

    #### Free GPU ####
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

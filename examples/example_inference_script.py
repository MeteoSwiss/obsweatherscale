# %%
import math
import torch
from gpytorch import settings

from obsweatherscale.inference import predict_posterior, predict_prior, sample
from obsweatherscale.kernels import NeuralKernel, ScaledRBFKernel
from obsweatherscale.likelihoods import TransformedGaussianLikelihood
from obsweatherscale.likelihoods.noise_models import (
    TransformedFixedGaussianNoise
)
from obsweatherscale.means import NeuralMean
from obsweatherscale.models import GPModel, MLP
from obsweatherscale.transformations import QuantileFittedTransformer
from obsweatherscale.transformations.standardizer import Standardizer


def main():
    #### Data ####
    def true_signal(x, y, t):
        return (
            torch.sin(math.pi * x) *
            torch.cos(math.pi * y) *
            torch.sin(2 * math.pi * t / t.shape[0])
        )

    n_stations, n_x, n_y, n_times, noise_var = 100, 30, 20, 10, 0.1
    n_points = n_x * n_y
    t = torch.linspace(0, n_times - 1, n_times)

    # Target data: the desired grid
    t_grid, x_grid, y_grid = torch.meshgrid(
        t, torch.linspace(0, 1, n_x), torch.linspace(0, 1, n_y), indexing='ij'
    )

    target_x = torch.stack([t_grid, x_grid, y_grid])
    target_y = true_signal(x_grid, y_grid, t_grid) + \
        math.sqrt(noise_var) * torch.randn_like(x_grid)

    target_x = target_x.reshape(n_times, n_points, 3) # [n_times, n_points, 3]
    target_y = target_y.reshape(n_times, n_points)    # [n_times, n_points]

    # Context data: the stations (used for training)
    x_coords, y_coords = torch.rand(1, n_stations), torch.rand(1, n_stations)
    x_coords = x_coords.expand(n_times, -1)    # [n_times, n_stations]
    y_coords = y_coords.expand(n_times, -1)    # [n_times, n_stations]
    t = t.unsqueeze(-1).expand(-1, n_stations) # [n_times, n_stations]

    context_x = torch.stack([x_coords, y_coords, t], dim=-1)
    context_y = true_signal(x_coords, y_coords, t) + \
        math.sqrt(noise_var) * torch.randn_like(x_coords)

    # Load artifacts and normalize
    # Note: here we instantiate them untrained, but they should be loaded
    standardizer = Standardizer(context_x)
    y_transformer = QuantileFittedTransformer()

    context_x = standardizer.transform(context_x)
    context_y = y_transformer.transform(context_y)
    target_x = standardizer.transform(target_x)
    target_y = y_transformer.transform(target_y)

    # Initialize device
    device = torch.device("cuda")

    # Initialize likelihood
    noise_model = TransformedFixedGaussianNoise(y_transformer, obs_noise_var=1)
    likelihood = TransformedGaussianLikelihood(noise_model)

    # Initialize model
    mean_function = NeuralMean(net=MLP(dimensions=[3, 32, 32, 1]))
    kernel = NeuralKernel(
        net=MLP(dimensions=[3, 32, 32, 4]),
        kernel=ScaledRBFKernel()
    )
    # Load the trained model (here we instantiate it but it should be loaded)
    model = GPModel(context_x, context_y, likelihood, mean_function, kernel)

    ## Evaluate
    model.to(device)
    likelihood.to(device)

    # 1. Predict distributions
    with (
        torch.no_grad(),
        settings.memory_efficient(True),
        settings.observation_nan_policy("fill")
    ):
        posterior = predict_posterior(
            model, likelihood, context_x, context_y, target_x
        )
        prior = predict_prior(model, likelihood, target_x, target_y)

    # 2. Sample from distributions
    # shape: (n_times, n_points, n_variables, n_samples)
    seed = 123
    n_samples = 101
    torch.manual_seed(seed)
    samples_posterior = sample(posterior, n_samples=n_samples)
    samples_prior = sample(prior, n_samples=n_samples)

    # 3. Reshape samples
    # Only one variable was predicted (true_signal), so we can squeeze it
    samples_posterior = samples_posterior.squeeze()
    samples_prior = samples_prior.squeeze()

    samples_posterior = samples_posterior.reshape(n_times, n_x, n_y, n_samples)
    samples_prior = samples_prior.reshape(n_times, n_x, n_y, n_samples)


if __name__ == "__main__":
    main()

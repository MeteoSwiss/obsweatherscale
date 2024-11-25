# %%
import argparse
import pickle
from pathlib import Path

import torch
import zarr
from gpytorch import settings

from examples.example_data_processing import (
    INPUTS, K_INPUTS, MF_INPUTS, SPATIAL_INPUTS, get_dataset,
    wrap_and_denormalize
)
from obsweatherscale.data_io import to_zarr_with_compressor
from obsweatherscale.inference import (
    marginal, predict_posterior, predict_prior, sample
)
from obsweatherscale.kernels import NeuralKernel, ScaledRBFKernel
from obsweatherscale.likelihoods import TransformedGaussianLikelihood
from obsweatherscale.likelihoods.noise_models import TransformedFixedGaussianNoise
from obsweatherscale.means import NeuralMean
from obsweatherscale.models import GPModel, MLP
from obsweatherscale.transformations import QuantileFittedTransformer
from obsweatherscale.utils import init_device


def main(config):
    artifacts_dir = config.artifacts_dir
    output_dir = config.output_dir
    model_path = artifacts_dir / config.model_filename

    # Load normalizers with training params
    with open(
        artifacts_dir / config.standardizer_filename, 'rb'
    ) as standardizer_path:
        standardizer = pickle.load(standardizer_path)
    with open(
        artifacts_dir / config.y_transformer_filename, 'rb'
    ) as y_transformer_path:
        y_transformer = pickle.load(y_transformer_path)

    # Load and normalize test data
    dataset_context = get_dataset(
        x_filename=config.data_dir / config.x_context_filename,
        y_filename=config.data_dir / config.y_context_filename,
        inputs=config.inputs,
        targets=config.targets,
        standardizer=standardizer,
        y_transformer=y_transformer
    )
    dataset_target = get_dataset(
        x_filename=config.data_dir / config.x_target_filename,
        y_filename=config.data_dir / config.y_target_filename,
        inputs=config.inputs,
        targets=config.targets,
        standardizer=standardizer,
        y_transformer=y_transformer
    )

    context_x, context_y = dataset_context.get_dataset()
    target_x, target_y = dataset_target.get_dataset()
    context_y, target_y = context_y.squeeze(-1), target_y.squeeze(-1)

    # Initialize device
    device = init_device(config.gpu, config.use_gpu)

    # Initialize likelihood
    transformer = QuantileFittedTransformer()
    noise_model = TransformedFixedGaussianNoise(
        transformer, obs_noise_var=torch.tensor(1.0)
    )
    likelihood = TransformedGaussianLikelihood(noise_model)

    # Initialize model
    mean_function = NeuralMean(
        net=MLP(
            dimensions=[len(MF_INPUTS), 32, 32, 1],
            active_dims=[INPUTS.index(v) for v in MF_INPUTS]
        )
    )
    spatial_kernel = ScaledRBFKernel(
        lengthscale=torch.tensor([0.25992706, 0.1681214, 0.08974447]),
        active_dims=[INPUTS.index(v) for v in SPATIAL_INPUTS],
        train_lengthscale=False
    )
    neural_kernel = NeuralKernel(
        net=MLP(
            dimensions=[len(K_INPUTS), 32, 32, 4],
            active_dims=[INPUTS.index(v) for v in K_INPUTS]
        ),
        kernel=ScaledRBFKernel()
    )
    kernel = spatial_kernel * neural_kernel
    model = GPModel(
        context_x, context_y, likelihood,
        modules={'mean_module': mean_function, 'covar_module': kernel}
    )
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device('cpu'),
            weights_only=True
        )
    )

    ## Evaluate
    model.to(device)
    likelihood.to(device)
    dataset_context.to(device)
    dataset_target.to(device)

    # 1. Predict distributions
    with settings.observation_nan_policy(config.nan_policy):
         # Evaluate posterior on target stations
         # using context stations ("context")
        context_distr = marginal(
            predict_posterior(
                model, likelihood, context_x, context_y, target_x
            )
        )

        # Evaluate posterior on target stations
        # using target stations ("hyperlocal")
        hyperloc_distr = marginal(
            predict_posterior(
                model, likelihood, target_x, target_y, target_x
            )
        )

        # Evaluate prior on target stations ("prior")
        prior_distr = marginal(
            predict_prior(model, likelihood, target_x, target_y)
        )

    # 2. Sample from distributions
    torch.manual_seed(config.seed)
    samples_context = sample(context_distr, config.n_samples)
    torch.manual_seed(config.seed)
    samples_hyperloc = sample(hyperloc_distr, config.n_samples)
    torch.manual_seed(config.seed)
    samples_prior = sample(prior_distr, config.n_samples)

    # 3. Wrap into xr.Dataset
    inv_transform = y_transformer.inverse_transform
    wrap_fun = dataset_target.wrap_pred
    name = config.targets[0]
    samples_context, samples_hyperloc, samples_prior = wrap_and_denormalize(
        samples_context,
        samples_hyperloc,
        samples_prior,
        wrap_fun=wrap_fun,
        denormalize_fun=inv_transform,
        name=name
    )

    # 4. Save
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE)
    for samples, version in zip(
        (samples_context, samples_hyperloc, samples_prior),
        ("context", "hyperlocal", "prior")
    ):
        directory = output_dir / version

        to_zarr_with_compressor(
            samples,
            filename=directory / "samples.zarr",
            compressor=compressor
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=Path('/', 'scratch', 'mch', 'illornsj', 'data', 'cosmo-1e')
    )
    parser.add_argument(
        '--artifacts_dir',
        type=str,
        default=Path(
            '/', 'scratch', 'mch', 'illornsj', 'data', 'experiments',
            'spatial_deep_kernel', 'artifacts'
        )
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=Path(
            '/', 'scratch', 'mch', 'illornsj', 'data', 'experiments',
            'spatial_deep_kernel', 'results'
        )
    )
    parser.add_argument(
        '--x_target_filename', type=str, default="x_test_target.zarr"
    )
    parser.add_argument(
        '--x_context_filename', type=str, default= "x_test_context.zarr"
    )
    parser.add_argument(
        '--y_target_filename', type=str, default="y_test_target.zarr"
    )
    parser.add_argument(
        '--y_context_filename', type=str, default="y_test_context.zarr"
    )
    parser.add_argument('--model_filename', type=str, default="best_model")

    parser.add_argument(
        '--standardizer_filename', type=str, default="standardizer.pkl")
    parser.add_argument(
        '--y_transformer_filename', type=str, default="y_transformer.pkl"
    )
    parser.add_argument(
        '--targets', type=list, default=["weather:wind_speed_of_gust"]
    )
    parser.add_argument('--inputs', type=list, default=INPUTS)
    parser.add_argument('--n_samples', type=int, default=101)
    parser.add_argument('--gpu', type=list, default=None)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--nan_policy', type=str, default='fill')

    args, _ = parser.parse_known_args()

    main(args)

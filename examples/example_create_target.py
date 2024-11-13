# %%
import argparse
import os
from pathlib import Path
import pickle
import shutil
import time

import gc
import torch
import xarray as xr
from gpytorch import settings

from examples.example_data_processing import (INPUTS, K_INPUTS, MF_INPUTS,
                                              SPATIAL_INPUTS, WindDataset,
                                              WindDatasetTarget,
                                              wrap_and_denormalize)
from obsweatherscale.inference import predict_posterior
from obsweatherscale.kernels import NeuralKernel, ScaledRBFKernel
from obsweatherscale.likelihoods.noise_models import TransformedFixedGaussianNoise
from obsweatherscale.likelihoods.transformed_likelihood import TransformedGaussianLikelihood
from obsweatherscale.means.neural_mean import NeuralMean
from obsweatherscale.models import GPModel, MLP
from obsweatherscale.transformations import QuantileFittedTransformer
from obsweatherscale.utils import init_device


def main(config):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    start = time.time()

    artifacts_dir = config.artifacts_dir
    data_dir = config.data_dir

    # 1. Load normalizers with training params
    with open(
        artifacts_dir / config.standardizer_filename, 'rb'
    ) as standardizer_path:
        standardizer = pickle.load(standardizer_path)
    with open(
        artifacts_dir / config.y_transformer_filename, 'rb'
    ) as y_transformer_path:
        y_transformer = pickle.load(y_transformer_path)

    # 2. Get context and target data
    x_target = xr.open_zarr(data_dir / config.x_t_filename)
    x_context = xr.open_zarr(data_dir / config.x_c_filename)
    y_context = xr.open_zarr(data_dir / config.y_c_filename)

    x_target = x_target[config.inputs].fillna(0.0)

    dataset_context = WindDataset(
        x_context[config.inputs].fillna(0.0),
        y_context[config.targets],
        standardizer=standardizer,
        y_transformer=y_transformer
    )
    timesteps = x_target.sizes['time']
    context_x, context_y = dataset_context.get_dataset()

    # 3. Load model and put in evaluation mode
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
        lengthscale=torch.tensor([0.2, 0.1, 0.1]),
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
            artifacts_dir / config.model_filename,
            map_location=torch.device('cpu'),
            weights_only=True
        )
    )
    # Put model on device
    model.to(device)
    likelihood.to(device)

    # 4. Initialize output Zarr arrays
    output_dir = config.output_dir / f"grid_{config.resolution}m"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path_mean = output_dir / "posterior_mean.zarr"
    output_path_std = output_dir / "posterior_std.zarr"

    # Start predicting timestep by timestep
    stop = time.time()
    print(f"Before loop - time: {stop - start:.3f}")

    model.eval()
    likelihood.eval()
    if output_path_mean.exists() and config.delete_previous:
        shutil.rmtree(output_path_mean)
    if output_path_std.exists() and config.delete_previous:
        shutil.rmtree(output_path_std)

    for i in range(0, timesteps, config.batch_size):
        start = time.time()

        # 1. Create dataset
        dataset_target = WindDatasetTarget(
            x_target.isel(time=slice(i, i + config.batch_size)),
            standardizer=standardizer,
        )
        wrap_fun = dataset_target.wrap_pred
        print(f"Dataset creation - time: {time.time() - start:.3f}")

        # 2. Get batch data
        x_c, y_c = dataset_context[i:i + config.batch_size]
        x_c, y_c = x_c.to(device), y_c.to(device).squeeze(-1)
        x_t = dataset_target[:].to(device)

        # 3. Predict
        with (
            torch.no_grad(),
            settings.memory_efficient(True),
            settings.observation_nan_policy(config.nan_policy)
        ):
            posterior = predict_posterior(model, likelihood, x_c, y_c, x_t)
            pred_mean, pred_std = posterior.mean, posterior.stddev

        pred_mean_da, pred_std_da = wrap_and_denormalize(
            pred_mean.cpu().unsqueeze(-1),
            pred_std.cpu().unsqueeze(-1),
            wrap_fun=wrap_fun,
            denormalize_fun=y_transformer.inverse_transform,
            name=config.targets[0]
        )

        # 4. Save
        if i == 0:
            pred_mean_da.to_zarr(output_path_mean)
            pred_std_da.to_zarr(output_path_std)
        else:
            pred_mean_da.to_zarr(output_path_mean, append_dim="time")
            pred_std_da.to_zarr(output_path_std, append_dim="time")

        # 5. Clear cache and collect garbage
        del (
            dataset_target, posterior, pred_mean, pred_std,
            pred_mean_da, pred_std_da, x_c, y_c, x_t
        )
        torch.cuda.empty_cache()
        gc.collect()

        # 6. Log
        stop = time.time()
        print(f"Iter {i}/{timesteps} - total time: {stop - start:.3f}")

    print("Prediction completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=Path(
            '/', 'scratch', 'mch', 'illornsj', 'data', 'cosmo-1e',
            'all_switzerland_250'
        )
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
    parser.add_argument('--x_t_filename', type=str, default="x_grid.zarr")
    parser.add_argument('--x_c_filename', type=str, default= "x.zarr")
    parser.add_argument('--y_c_filename', type=str, default="y.zarr")
    parser.add_argument('--model_filename', type=str, default="best_model")

    parser.add_argument(
        '--standardizer_filename', type=str, default="standardizer.pkl"
    )
    parser.add_argument(
        '--y_transformer_filename', type=str, default="y_transformer.pkl"
    )
    parser.add_argument(
        '--targets', type=list, default=["weather:wind_speed_of_gust"]
    )
    parser.add_argument('--inputs', type=list, default=INPUTS)

    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--gpu', type=list, default=None)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--prec_size', type=int, default=1000)
    parser.add_argument('--nan_policy', type=str, default='fill')

    parser.add_argument('--nwp_interp_method', type=str, default='linear')
    parser.add_argument('--resolution', type=int, default=250)

    args, _ = parser.parse_known_args()

    main(args)

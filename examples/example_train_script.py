# %%
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.adam import Adam

from examples.example_data_processing import (
    INPUTS, K_INPUTS, MF_INPUTS, SPATIAL_INPUTS, get_training_data
)
from obsweatherscale.kernels import NeuralKernel, ScaledRBFKernel
from obsweatherscale.likelihoods import (
    TransformedGaussianLikelihood, ExactMarginalLogLikelihoodFill
)
from obsweatherscale.likelihoods.noise_models import TransformedFixedGaussianNoise
from obsweatherscale.means import NeuralMean
from obsweatherscale.models import GPModel, MLP
from obsweatherscale.training import (
    crps_normal_loss_fct, mll_loss_fct, train_model
)
from obsweatherscale.transformations import QuantileFittedTransformer
from obsweatherscale.utils import init_device


def main(config):
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_filename = "model"

    start = time.time()
    torch.manual_seed(config.seed)

    # Get data
    dataset_train, dataset_val_c, dataset_val_t = get_training_data(
        config.data_dir,
        config.x_train_filename,
        config.y_train_filename,
        config.x_val_c_filename,
        config.y_val_c_filename,
        config.x_val_t_filename,
        config.y_val_t_filename,
        config.inputs,
        config.targets
    )
    train_x, train_y = dataset_train.get_dataset()

    # Initialize device
    device = init_device(config.gpu, config.use_gpu)

    # Initialize likelihood
    transformer = QuantileFittedTransformer()
    noise_model = TransformedFixedGaussianNoise(
        transformer, obs_noise_var=torch.tensor(1.0)
    )
    likelihood = TransformedGaussianLikelihood(noise_model)

    # Initialize model
    torch.manual_seed(config.seed)
    mean_function = NeuralMean(
        net=MLP(
            dimensions=[len(MF_INPUTS), 32, 32, 1],
            active_dims=[INPUTS.index(v) for v in MF_INPUTS]
        )
    )
    # optim lengthscale: [0.25992706, 0.1681214, 0.08974447]
    spatial_kernel = ScaledRBFKernel(
        lengthscale=torch.tensor([0.25992706, 0.1681214, 0.08974447]),
        active_dims=tuple(INPUTS.index(v) for v in SPATIAL_INPUTS),
        train_lengthscale=False
    )
    torch.manual_seed(config.seed)
    neural_kernel = NeuralKernel(
        net=MLP(
            dimensions=[len(K_INPUTS), 32, 32, 4],
            active_dims=[INPUTS.index(v) for v in K_INPUTS]
        ),
        kernel=ScaledRBFKernel()
    )
    kernel = spatial_kernel * neural_kernel
    model = GPModel(
        train_x, train_y, likelihood,
        modules={'mean_module': mean_function, 'covar_module': kernel}
    )

    # Initialize optimizer and loss
    optimizer = Adam(
        [{'params': model.parameters()},
         {'params': likelihood.parameters()}],
        lr=config.learning_rate
    )
    mll = ExactMarginalLogLikelihoodFill(likelihood, model)
    train_loss_fct = mll_loss_fct(mll)
    val_loss_fct = crps_normal_loss_fct(likelihood)

    # Train
    model, train_progress = train_model(
        dataset_train,
        dataset_val_c,
        dataset_val_t,
        model,
        likelihood,
        train_loss_fct=train_loss_fct,
        val_loss_fct=val_loss_fct,
        device=device,
        optimizer=optimizer,
        batch_size=config.batch_size,
        output_dir=output_dir,
        model_filename=model_filename,
        n_iter=config.n_iter,
        random_masking=config.random_masking,
        seed=config.seed,
        nan_policy=config.nan_policy,
        prec_size=config.prec_size
    )

    # Identify and label the best performing model
    best_model_idx = np.argmin(train_progress["val loss"])
    best_val_loss = np.min(train_progress["val loss"])
    print(
        f"Best model idx: {best_model_idx}, best val loss: {best_val_loss}",
        flush=True
    )
    best_model_filename = f"{model_filename}_iter_{best_model_idx}"
    model.load_state_dict(
        torch.load(config.output_dir / best_model_filename, weights_only=True)
    )

    # Save
    torch.save(model.state_dict(), output_dir / "best_model")
    with open(
        output_dir / "train_progress.pkl", 'wb'
    ) as train_progress_path:
        pickle.dump(train_progress, train_progress_path)
    with open(
        output_dir / config.standardizer_filename, 'wb'
    ) as standardizer_path:
        pickle.dump(
            dataset_train.standardizer,
            standardizer_path,
            pickle.HIGHEST_PROTOCOL
        )
    with open(
        output_dir / config.y_transformer_filename, 'wb'
    ) as y_transformer_path:
        pickle.dump(
            dataset_train.y_transformer,
            y_transformer_path,
            pickle.HIGHEST_PROTOCOL
        )

    # Free GPU
    torch.cuda.empty_cache()

    stop = time.time()
    print(
        f"Total time: {stop - start:.3f} s / {(stop - start)/60:.3f} min",
        flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=Path('/', 'scratch', 'mch', 'illornsj', 'data', 'cosmo-1e')
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=Path(
            '/', 'scratch', 'mch', 'illornsj', 'data', 'experiments',
            'spatial_deep_kernel', 'artifacts'
        )
    )
    parser.add_argument('--x_train_filename', type=str, default="x_train.zarr")
    parser.add_argument('--y_train_filename', type=str, default="y_train.zarr")
    parser.add_argument(
        '--x_val_c_filename', type=str, default="x_val_context.zarr"
    )
    parser.add_argument(
        '--y_val_c_filename', type=str, default="y_val_context.zarr"
    )
    parser.add_argument(
        '--x_val_t_filename', type=str, default="x_val_target.zarr"
    )
    parser.add_argument(
        '--y_val_t_filename', type=str, default="y_val_target.zarr"
    )
    parser.add_argument(
        '--standardizer_filename', type=str, default="standardizer.pkl"
    )
    parser.add_argument(
        '--y_transformer_filename', type=str, default="y_transformer.pkl"
    )
    parser.add_argument('--inputs', type=list, default=INPUTS)
    parser.add_argument(
        '--targets', type=list, default=["weather:wind_speed_of_gust"]
    )
    parser.add_argument('--random_masking', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_iter', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.025)
    parser.add_argument('--gpu', type=list, default=None)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--prec_size', type=int, default=1000)
    parser.add_argument('--nan_policy', type=str, default='fill')

    args, _ = parser.parse_known_args()

    main(args)

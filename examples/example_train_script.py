# %%
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood


from examples.example_data_processing import (INPUTS,
                                              K_INPUTS,
                                              MF_INPUTS,
                                              SPATIAL_INPUTS,
                                              get_training_data)
from obsweatherscale.kernels import (NeuralKernel,
                                     ScaledRBFKernel)
from obsweatherscale.likelihoods import TransformedGaussianLikelihood
from obsweatherscale.likelihoods.noise_models import TransformedFixedGaussianNoise
from obsweatherscale.means import NeuralMean
from obsweatherscale.models import GPModel, MLP
from obsweatherscale.training.loss_functions import (crps_normal_loss_fct,
                                                     mll_loss_fct)
from obsweatherscale.training.training import train_model
from obsweatherscale.transformations import QuantileFittedTransformer
    

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
    device = torch.device("cuda:0") \
        if torch.cuda.is_available() and config.use_gpu else torch.device("cpu")

    # Initialize likelihood
    transformer = QuantileFittedTransformer()
    noise_model = TransformedFixedGaussianNoise(
        transformer, obs_noise_var=torch.tensor(1.0)
    )
    likelihood = TransformedGaussianLikelihood(noise_model)

    # Initialize model
    # optim lengthscale: [0.25992706, 0.1681214, 0.08974447]
    spatial_kernel = ScaledRBFKernel(
        lengthscale=torch.tensor([0.25992706, 0.1681214, 0.08974447]),
        active_dims=[INPUTS.index(v) for v in SPATIAL_INPUTS],
        train_lengthscale=False
    )
    neural_kernel = NeuralKernel(
        net=MLP(len(K_INPUTS), [32, 32], 4),
        kernel=ScaledRBFKernel(),
        active_dims=[INPUTS.index(v) for v in K_INPUTS]
    )
    kernel = spatial_kernel * neural_kernel
    mean_function = NeuralMean(
        net=MLP(len(MF_INPUTS), [32, 32], 1),
        active_dims=[INPUTS.index(v) for v in MF_INPUTS]
    )

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
        prec_size=config.prec_size
    )
    
    # Identify and label the best performing model
    best_model_idx = np.argmin(train_progress["val loss"])
    best_model_path = config.output_dir / "{}_iter_{}".format(
        model_filename, best_model_idx
    )
    model.load_state_dict(torch.load(best_model_path))

    # Save
    torch.save(
        model.state_dict(), config.output_dir / "best_model"
    )
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
    print(f"Total time: {stop - start:.3f} s "
          f"/ {(stop - start)/60:.3f} min"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default=Path('/', 'scratch', 'mch', 'illornsj', 'data', 'cosmo-1e'))
    parser.add_argument(
        '--output_dir',
        type=str,
        default=Path(
            '/', 'scratch', 'mch', 'illornsj',
            'data', 'experiments', 'spatial_deep_kernel', 'artifacts'
            )
        )
    parser.add_argument(
        '--x_train_filename',
        type=str,
        default="x_train_replicate.zarr"
    )
    parser.add_argument(
        '--y_train_filename',
        type=str,
        default="y_train_replicate.zarr"
    )
    parser.add_argument(
        '--x_val_c_filename',
        type=str,
        default="x_val_context_replicate.zarr"
    )
    parser.add_argument(
        '--y_val_c_filename',
        type=str,
        default="y_val_context_replicate.zarr"
    )
    parser.add_argument(
        '--x_val_t_filename',
        type=str,
        default="x_val_target_replicate.zarr"
    )
    parser.add_argument(
        '--y_val_t_filename',
        type=str,
        default="y_val_target_replicate.zarr"
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
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--prec_size', type=int, default=100)

    args, _ = parser.parse_known_args()

    main(args)

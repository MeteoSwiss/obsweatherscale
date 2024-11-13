import time
from pathlib import Path
from typing import Optional, Union

import torch
from gpytorch import settings
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ExactGP
from torch.utils.data import Dataset

from ..utils import apply_random_masking, sample_batch_idx


def train_step(
    model: ExactGP,
    likelihood: torch.Tensor,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    loss_fct: callable
) -> float:
    model.train()
    likelihood.train()

    model.set_train_data(inputs=batch_x, targets=batch_y, strict=False)
    distribution = model(batch_x)
    loss = loss_fct(distribution, batch_y)

    loss.backward()

    return loss.item()


def val_step(
    model: ExactGP,
    likelihood: _GaussianLikelihoodBase,
    batch_x_c: torch.Tensor,
    batch_y_c: torch.Tensor,
    batch_x_t: torch.Tensor,
    batch_y_t: torch.Tensor,
    loss_fct: callable
) -> float:
    model.eval()
    likelihood.eval()

    model.set_train_data(batch_x_c, batch_y_c, strict=False)
    distribution_val = model(batch_x_t)
    loss = loss_fct(distribution_val, batch_y_t)

    return loss.item()


def train_model(
    dataset_train: Dataset,
    dataset_val_c: Dataset,
    dataset_val_t: Dataset,
    model: ExactGP,
    likelihood: _GaussianLikelihoodBase,
    train_loss_fct: callable,
    val_loss_fct: callable,
    device: torch.device,
    optimizer: torch.optim,
    batch_size: int,
    output_dir: Path,
    model_filename: str,
    n_iter: int,
    random_masking: bool = True,
    seed: Optional[int] = None,
    nan_policy: str = 'fill',
    prec_size: int = 100,
) -> Union[torch.nn.Module, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_length = len(dataset_train)
    dataset_val_length = len(dataset_val_c)
    train_progression = {"iter": [], "train loss": [], "val loss": []}

    torch.manual_seed(seed)

    # Transfer everything to device at the beginning
    model.to(device)
    likelihood.to(device)
    dataset_train.to(device)
    dataset_val_c.to(device)
    dataset_val_t.to(device)

    for i in range(n_iter):
        start = time.time()

        with settings.max_preconditioner_size(prec_size):
            optimizer.zero_grad()

            ### Training ###
            # Get iter data
            batch_idx = sample_batch_idx(dataset_length, batch_size)
            batch_x, batch_y = dataset_train[batch_idx]

            if random_masking:
                batch_y = apply_random_masking(batch_y)

            with settings.observation_nan_policy(nan_policy):
                train_loss = train_step(
                    model, likelihood, batch_x, batch_y, train_loss_fct
                )

            optimizer.step()
            stop_train = time.time()

            ### Validation ###
            # Get iter data
            batch_idx = sample_batch_idx(dataset_val_length, batch_size)
            batch_x_c, batch_y_c = dataset_val_c[batch_idx]
            batch_x_t, batch_y_t = dataset_val_t[batch_idx]

            with torch.no_grad(), settings.observation_nan_policy(nan_policy):
                val_loss = val_step(
                    model, likelihood,
                    batch_x_c, batch_y_c, batch_x_t, batch_y_t,
                    val_loss_fct
                )

        ### Logging ###
        # Save model at each iteration
        torch.save(
            model.state_dict(), output_dir / f"{model_filename}_iter_{i}"
        )

        # Save training log
        train_progression["iter"].append(i+1)
        train_progression["train loss"].append(train_loss)
        train_progression["val loss"].append(val_loss)

        stop = time.time()

        # Print training log
        print(
            f"Iter {i + 1}/{n_iter} - "
            f"Loss: {train_loss:.3f}   "
            f"Val loss: {val_loss:.3f}   "
            f"train time: {stop_train - start:.3f}   "
            f"time: {stop - start:.3f}"
        )

    return model, train_progression

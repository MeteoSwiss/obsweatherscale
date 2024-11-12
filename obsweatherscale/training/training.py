import time
from pathlib import Path

import gpytorch.settings as settings
import torch
from torch.utils.data import Dataset

from ..utils import RandomStateContext, sample_batch_idx


def train_model(
    dataset_train: Dataset,
    dataset_val_c: Dataset,
    dataset_val_t: Dataset,
    model: torch.nn.Module,
    likelihood,
    train_loss_fct,
    val_loss_fct,
    device: torch.device,
    optimizer: torch.optim,
    batch_size: int,
    output_dir: Path,
    model_filename: str,
    n_iter: int,
    random_masking: bool = True,
    seed: int = None, 
    nan_policy: str = 'fill',
    prec_size: int = 100,
) -> tuple[torch.nn.Module, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_length = len(dataset_train)
    dataset_val_length = len(dataset_val_c)
    train_progression = {"iter": [],
                         "train loss": [],
                         "val loss": []}
    mask_shape = (1, *dataset_train.y.shape[1:])

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
            model.train()
            likelihood.train()

            # Get iter data
            batch_idx = sample_batch_idx(
                dataset_length,
                batch_size
            )
            batch_x, batch_y = dataset_train[batch_idx]

            # Random masking
            if random_masking:
                with RandomStateContext():
                    random_mask = torch.bernoulli(
                        torch.ones(mask_shape)*0.5
                    ).bool().expand_as(batch_y)
                    batch_y[random_mask] = torch.nan

            with settings.observation_nan_policy(nan_policy):
                model.set_train_data(
                    inputs=batch_x,
                    targets=batch_y,
                    strict=False
                )
                distribution = model(batch_x)
                loss = train_loss_fct(distribution, batch_y)
                train_loss = loss.item()
                
                loss.backward()
                optimizer.step()
            
            stop_train = time.time()

            ### Validation ###
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                batch_idx_val = sample_batch_idx(
                    dataset_val_length,
                    batch_size
                )
                batch_x_c, batch_y_c = dataset_val_c[batch_idx_val]
                batch_x_t, batch_y_t = dataset_val_t[batch_idx_val]

                with settings.observation_nan_policy(nan_policy):
                    model.set_train_data(
                        batch_x_c,
                        batch_y_c,
                        strict=False
                    )
                    distribution_val = model(batch_x_t)
                    val_loss = val_loss_fct(
                        distribution_val,
                        batch_y_t
                    ).item()
            
        ### Logging ###
        # Save model at each iteration
        torch.save(
            model.state_dict(),
            output_dir / f"{model_filename}_iter_{i}"
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

import random
import time
from pathlib import Path
from typing import Callable

import torch
from gpytorch import settings
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ExactGP
from torch.optim.optimizer import Optimizer

from obsweatherscale.data import GPDataset

__all__ = ["Trainer"]

from obsweatherscale.logger import TerminalLogger, Logger


class RandomStateContext:
    """Context manager for preserving and restoring PyTorch's RNG state.

    This context manager saves the current random number generator (RNG)
    state upon entering and restores it upon exiting. This is useful to
    ensure deterministic behavior when randomness is used within a
    controlled block of code, without affecting the global RNG state
    outside the block.

    Notes
    -----
    - The RNG state is retrieved and stored using
    `torch.random.get_rng_state()` and `torch.random.set_rng_state()`.
    - A new seed is set upon entering the context using
    `torch.manual_seed(torch.seed())`.
    """

    def __init__(self) -> None:
        self.current_state: torch.Tensor

    def __enter__(self) -> 'RandomStateContext':
        """Enter the context, saving the current RNG state and reseeding.

        Returns
        -------
        RandomStateContext
            The context manager instance.
        """
        self.current_state = torch.random.get_rng_state()
        torch.manual_seed(torch.seed())
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Exit the context and restore the original RNG state.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            The exception type if an exception occurred, else None.
        exc_value : BaseException | None
            The exception instance if an exception occurred, else None.
        traceback : object | None
            Traceback object if an exception occurred, else None.
        """
        torch.random.set_rng_state(self.current_state)


class Trainer:
    """Trainer class for Gaussian Process models."""

    def __init__(
        self,
        model: ExactGP,
        likelihood: _GaussianLikelihoodBase,
        train_loss_fct: Callable,
        val_loss_fct: Callable,
        device: torch.device,
        optimizer: Optimizer,
    ) -> None:
        """Initialize the Trainer class.

        Parameters
        ----------
        model : ExactGP
            The Gaussian Process prior model.
        likelihood : _GaussianLikelihoodBase
            The likelihood function for the model.
        train_loss_fct : Callable
            The loss function to use for training.
        val_loss_fct : Callable
            The loss function to use for validation.
        device : torch.device
            The device to use for training (CPU or GPU).
        optimizer : Optimizer
            The optimizer to use for training the model.
        """
        self.model = model
        self.best_model = model
        self.likelihood = likelihood
        self.train_loss_fct = train_loss_fct
        self.val_loss_fct = val_loss_fct
        self.device = device
        self.optimizer = optimizer

    def fit(
        self,
        train: GPDataset,
        val_context: GPDataset,
        val_target: GPDataset,
        batch_size: int,
        n_iter: int,
        random_masking: bool = True,
        seed: int | None = None,
        nan_policy: str = "fill",
        prec_size: int = 100,
        output_dir: Path | None = None,
        verbose: bool = True,
        loggers: list[Logger] | None = None,
    ) -> tuple[ExactGP, dict[str, list]]:
        """Train the Gaussian Process model.

        Parameters
        ----------
        train : GPDataset
            The training dataset.
        val_context : GPDataset
            The validation dataset (context).
        val_target : GPDataset
            The validation dataset (target).
        batch_size : int
            The size of the batches for training.
        n_iter : int
            The number of iterations for training.
        random_masking : bool, default=True
            Whether to apply random masking to the training data.
        seed : int, optional, default=None
            The random seed for reproducibility.
        nan_policy : str, default='fill'
            The policy for handling NaN values in the data. Options are
                - 'mask': removes all data points where the y is nan
                - 'fill': replaces nan values and keeps data points
        prec_size : int, default=100
            The size of the preconditioner for the optimizer.
        output_dir : Path, optional, default=None
            The directory to save the model checkpoints. If None, the
            model does not get saved during training.
        verbose : bool, default=True
            If True, prints training status (loss function values, iter,
            time)
        loggers : Sequence of TrainingLogger, optional, default=None
            Sequence of :class:`TrainingLogger` instances (e.g.
            :class:`TerminalLogger`, :class:`CSVLogger`,
            :class:`MLflowLogger`).  Each logger receives
            hyperparameters once before training and per-iteration
            metrics.  When *None*, no additional logging is performed.
        Returns
        -------
        model : ExactGP
            The trained Gaussian Process model.
        train_progression : dict[str, list]
            Dictionary with training progression metrics with keys:
            - 'iter': List of iteration numbers
            - 'train loss': List of training loss values
            - 'val loss': List of validation loss values
            - 'train time': List of training step durations (seconds)
            - 'iter time': List of total iteration durations (seconds)
        """

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        length = len(train)
        val_length = len(val_context)
        train_progression : dict[str, list] = {
            "iter": [], "train loss": [], "val loss": [], "iter time": [], "train time": []
        }

        torch.manual_seed(seed)

        # Transfer everything to device at the beginning
        self.model.to(self.device)
        self.likelihood.to(self.device)
        train.to(self.device)
        val_context.to(self.device)
        val_target.to(self.device)

        best_val_loss = torch.inf

        loggers_list: list[Logger] = list(loggers) if loggers else []
        if verbose:
            loggers_list.append(TerminalLogger())

        log_params: dict = {
            "batch_size": batch_size,
            "n_iter": n_iter,
            "seed": seed,
            "random_masking": random_masking,
            "nan_policy": nan_policy,
            "prec_size": prec_size,
            "device": str(self.device),
            "model": type(self.model).__name__,
            "optimizer": type(self.optimizer).__name__,
        }
        for group in self.optimizer.param_groups:
            log_params["learning_rate"] = group["lr"]
            break
        for logger in loggers_list:
            logger.log_params(log_params)

        for i in range(n_iter):
            start = time.time()

            with settings.max_preconditioner_size(prec_size):
                self.optimizer.zero_grad()

                # Training
                # Get iter data
                batch_idx = self._sample_batch_idx(length, batch_size)
                batch_x, batch_y = train[batch_idx]

                if random_masking:
                    batch_y = self._apply_random_masking(batch_y)

                with settings.observation_nan_policy(nan_policy):
                    train_loss = self._train_step(batch_x, batch_y)

                self.optimizer.step()
                stop_targetrain = time.time()

                # Validation
                # Get iter data
                batch_idx = self._sample_batch_idx(val_length, batch_size)
                batch_x_context, batch_y_context = val_context[batch_idx]
                batch_x_target, batch_y_target = val_target[batch_idx]

                with (
                    torch.no_grad(),
                    settings.observation_nan_policy(nan_policy)
                ):
                    val_loss = self._val_step(
                        batch_x_context,
                        batch_y_context,
                        batch_x_target,
                        batch_y_target
                    )

            # Logging
            # Save best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = self.model

                if output_dir is not None:
                    torch.save(self.model.state_dict(), output_dir / "model")

            stop = time.time()

            iter_metrics = {
                "iter": i + 1,
                "train loss": train_loss,
                "val loss": val_loss,
                "train time": stop_targetrain - start,
                "iter time": stop - start
            }
            for logger in loggers_list:
                logger.log_metrics(iter_metrics, step=i + 1)

            for k, v in iter_metrics.items():
                train_progression[k].append(v)

        for logger in loggers_list:
            logger.close()

        return self.best_model, train_progression

    def _train_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> float:
        """Perform a training step on the model.

        Parameters
        ----------
        batch_x : torch.Tensor
            The input data for the training step.
        batch_y : torch.Tensor
            The target data for the training step.

        Returns
        -------
        float
            The value of the loss function for this training step.
        """
        self.model.train()
        self.likelihood.train()

        self.model.set_train_data(
            inputs=batch_x, targets=batch_y, strict=False
        )
        distribution = self.model(batch_x)
        loss = self.train_loss_fct(distribution, batch_y)

        loss.backward()

        return loss.item()

    def _val_step(
        self,
        batch_x_context: torch.Tensor,
        batch_y_context: torch.Tensor,
        batch_x_target: torch.Tensor,
        batch_y_target: torch.Tensor,
    ) -> float:
        """Perform a validation step on the model.

        The validation loss is computed on target {}_target data
        conditioned on the context {}_context data. It can be used to
        diagnose the model's generalization performance.

        Parameters
        ----------
        batch_x_context : torch.Tensor
            The input data for the validation step (conditional).
        batch_y_context : torch.Tensor
            The target data for the validation step (conditional).
        batch_x_target : torch.Tensor
            The input data for the validation step (target).
        batch_y_target : torch.Tensor
            The target data for the validation step (target).

        Returns
        -------
        float
            The value of the loss function for this validation step.
        """
        self.model.eval()
        self.likelihood.eval()

        self.model.set_train_data(
            batch_x_context, batch_y_context, strict=False
        )
        distribution_val = self.model(batch_x_target)
        loss = self.val_loss_fct(distribution_val, batch_y_target)

        return loss.item()

    def _sample_batch_idx(self, length: int, batch_size: int) -> list[int]:
        """Randomly sample a batch of unique indices.

        Parameters
        ----------
        length : int
            The total number of available items to sample from.
        batch_size : int
            The number of unique indices to sample.

        Returns
        -------
        list of int
            A list of `batch_size` unique indices randomly sampled from
            the range [0, length).
        """
        return random.sample(range(length), batch_size)

    def _apply_random_masking(
        self,
        data: torch.Tensor,
        p: float = 0.5,
    ) -> torch.Tensor:
        """Apply random NaN masking to the input data.

        Parameters
        ----------
        data : torch.Tensor
            The input tensor to mask.
        p : float, default=0.5
            The probability of masking each element.

        Returns
        -------
        torch.Tensor
            The masked tensor with NaN values inserted.
        """
        mask_shape = (1, *data.shape[1:])

        with RandomStateContext():
            random_mask = torch.bernoulli(
                torch.ones(mask_shape) * p
            ).bool().expand_as(data)
            data[random_mask] = torch.nan

        return data

"""Trainer class for Gaussian Process models.

This module provides utilities for training and validating GPyTorch-
based Gaussian Process models, including a random state context manager
for reproducible evaluation and a ``Trainer`` class that encapsulates
the training loop logic.

Classes
-------
RandomStateContext
    Context manager for preserving and restoring PyTorch's RNG state.
Trainer
    Orchestrates training and validation of an ``ExactGP`` model.
"""

import copy
import random
import time
from pathlib import Path
from typing import Callable
import warnings

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
        """Enter the context, save the current RNG state and reseed.

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


class Trainer:  # pylint: disable=too-many-instance-attributes
    """Orchestrates training and validation of a GPyTorch ``ExactGP``
    model.

    ``Trainer`` wraps an ``ExactGP`` model together with its likelihood,
    loss functions, optimiser, and target device into a single object
    that manages the training loop. It tracks the best validation loss
    seen so far and keeps a snapshot of the corresponding model weights.

    Parameters
    ----------
    model : ExactGP
        The Gaussian Process prior model to be trained.
    train_loss_fn : Callable
        Loss function used during training. Expected signature::
            loss = train_loss_fn(output, target)
    val_loss_fn : Callable
        Loss function used during validation. Expected signature::
            loss = val_loss_fn(output, target)
    device : torch.device
        The device (``"cpu"`` or ``"cuda"``) on which tensors and the
        model will reside during training.
    optimizer : Optimizer
        A PyTorch-compatible optimizer responsible for updating the
        model and likelihood parameters.

    Attributes
    ----------
    model : ExactGP
        The GP model being trained.
    best_model : ExactGP
        A snapshot of *model* at the epoch with the lowest validation
        loss. Initialized to *model* at construction time.
    likelihood : _GaussianLikelihoodBase
        The Gaussian likelihood used for training and evaluation.
    train_loss_fn : Callable
        The training loss function.
    val_loss_fn : Callable
        The validation loss function.
    device : torch.device
        The compute device used for training.
    optimizer : Optimizer
        The parameter optimiser.
    best_val_loss : float
        The lowest validation loss recorded across all training
        iterations. Initialized to ``torch.inf``.
    nan_policy : str, (one of {'fill', 'mask'})
        The policy for handling NaN values in the data. Options are
            - 'mask': removes all data points where the y is nan
            - 'fill': replaces nan values and keeps data points
        Is taken as the model's nan_policy attribute, and if not
        available, defaults to 'mask' (default gpytorch value).
    history : list[dict]
        Per-iteration metrics recorded during last call to ``fit()``.

    Examples
    --------
    >>> trainer = Trainer(
    ...     model=gp_model,
    ...     train_loss_fn=mll,
    ...     val_loss_fn=rmse,
    ...     optimizer=torch.optim.Adam(gp_model.parameters(), lr=0.01),
    ...     device=torch.device("cuda"),
    ... )
    """

    def __init__(
        self,
        model: ExactGP,
        train_loss_fn: Callable,
        val_loss_fn: Callable,
        device: torch.device,
        optimizer: Optimizer,
    ) -> None:
        self.model = model
        self.likelihood: _GaussianLikelihoodBase = model.likelihood # type: ignore
        self.train_loss_fn = train_loss_fn
        self.val_loss_fn = val_loss_fn
        self.device = device
        self.optimizer = optimizer
        self.nan_policy = getattr(self.model, "nan_policy", "mask")

        self.history: list[dict] = []
        self.best_val_loss = torch.inf
        self._best_state: dict = copy.deepcopy(self.model.state_dict())

    @property
    def best_model(self) -> ExactGP:
        """The model, holding best-validation-loss weights post-fit."""
        best_model = copy.deepcopy(self.model)
        best_model.load_state_dict(self._best_state)
        return best_model

    def fit(
        self,
        train: GPDataset,
        val_context: GPDataset,
        val_target: GPDataset,
        batch_size: int,
        n_iter: int,
        random_masking: bool = True,
        seed: int = 123,
        prec_size: int = 100,
        output_dir: Path | None = None,
        verbose: bool = True,
        loggers: list[Logger] | None = None,
    ) -> "Trainer":
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
        seed : int, default=123
            The random seed for reproducibility.
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
        Trainer
            ``self``, so calls can be chained (e.g.
            ``trainer.fit(...).best_model``). The best model, its
            validation loss, and per-iteration history are available
            as attributes afterward.
        """

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        length = len(train)
        val_length = len(val_context)

        torch.manual_seed(seed)

        # Transfer everything to device at the beginning
        self.model.to(self.device)
        self.likelihood.to(self.device)
        train.to(self.device)
        val_context.to(self.device)
        val_target.to(self.device)

        loggers_list: list[Logger] = list(loggers) if loggers else []
        if verbose:
            loggers_list.append(TerminalLogger())

        log_params: dict = {
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "batch_size": batch_size,
            "n_iter": n_iter,
            "seed": seed,
            "random_masking": random_masking,
            "nan_policy": self.nan_policy,
            "prec_size": prec_size,
            "device": str(self.device),
            "model": type(self.model).__name__,
            "optimizer": type(self.optimizer).__name__,
        }
        for logger in loggers_list:
            logger.log_params(log_params)

        self.history = []

        for i in range(n_iter):
            start = time.time()

            with settings.max_preconditioner_size(prec_size):
                self.optimizer.zero_grad()

                # Training
                batch_idx = self._sample_batch_idx(length, batch_size)
                batch_x, batch_y = train[batch_idx]

                if random_masking:
                    batch_y = self._apply_random_masking(batch_y)

                train_loss = self._train_step(batch_x, batch_y)

                self.optimizer.step()
                stop_train = time.time()

                # Validation
                batch_idx = self._sample_batch_idx(val_length, batch_size)
                batch_x_context, batch_y_context = val_context[batch_idx]
                batch_x_target, batch_y_target = val_target[batch_idx]

                val_loss = self._val_step(
                    batch_x_context,
                    batch_y_context,
                    batch_x_target,
                    batch_y_target,
                )

            # Logging
            # Save checkpoint if output_dir is provided
            if output_dir is not None:
                torch.save(
                    self.model.state_dict(),
                    output_dir / f"model_{i}.pt",
                )

            # Keep track of best validation loss and best model so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._best_state = copy.deepcopy(self.model.state_dict())

            stop = time.time()

            iter_metrics = {
                "iter": i,
                "train loss": train_loss,
                "val loss": val_loss,
                "train time": stop_train - start,
                "iter time": stop - start,
            }
            for logger in loggers_list:
                logger.log_metrics(iter_metrics, step=i)
            self.history.append(iter_metrics)

        # Restore best-validation-loss weights into self.model
        self.model.load_state_dict(self._best_state)

        for logger in loggers_list:
            logger.log_metrics(
                {"best_val_loss": self.best_val_loss},
                step=None,
            )

        return self

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

        with settings.observation_nan_policy(self.nan_policy):
            self.model.set_train_data(
                inputs=batch_x, targets=batch_y, strict=False,
            )
            distribution = self.model(batch_x)
            loss = self.train_loss_fn(distribution, batch_y)

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

        with (
            torch.no_grad(),
            settings.observation_nan_policy(self.nan_policy),
        ):
            self.model.set_train_data(
                batch_x_context, batch_y_context, strict=False,
            )
            distribution_val = self.model(batch_x_target)
            loss = self.val_loss_fn(distribution_val, batch_y_target)

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
        if batch_size > length:
            warnings.warn(
                f"batch_size {batch_size} exceeds dataset size ({length}). "
                f"Using {length} as batch size.",
                UserWarning,
                stacklevel=3,
            )
            batch_size = length

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
            random_mask = (
                torch.bernoulli(torch.ones(mask_shape) * p)
                .bool()
                .expand_as(data)
            )
            data[random_mask] = torch.nan

        return data

"""GPModel class.

This module provides :class:`GPModel`, a GPyTorch
:class:`~gpytorch.models.ExactGP` subclass that composes an arbitrary
mean function and covariance kernel into a full GP model. It extends the
base class with convenience methods for prior and posterior prediction
that handle train/eval mode switching automatically.

Classes
-------
GPModel
    An exact Gaussian Process model with flexible mean and covariance
    modules.
"""

from contextlib import contextmanager
from typing import Any, cast, Generator

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.means import Mean
from gpytorch.models import ExactGP


class GPModel(ExactGP):
    """An exact Gaussian Process model with flexible mean and covariance
    modules.

    Composes an arbitrary :class:`~gpytorch.means.Mean` and
    :class:`~gpytorch.kernels.Kernel` into a GPyTorch
    :class:`~gpytorch.models.ExactGP`, and extends it with high-level
    :meth:`predict_prior` and :meth:`predict_posterior` methods that
    handle train/eval mode switching transparently.

    The prior distribution over function values at inputs
    :math:`\\mathbf{X}` is:

    .. math::
        f(\\mathbf{X}) \\sim \\mathcal{GP}\\!
            \\left(m(\\mathbf{X}),\\, k(\\mathbf{X}, \\mathbf{X})\\right)

    and the posterior is conditioned on context observations
    :math:`(\\mathbf{X}_c, \\mathbf{y}_c)` via exact GP inference.

    Parameters
    ----------
    mean_module : Mean
        The GP prior mean function :math:`m(\\mathbf{x})`.
    covar_module : Kernel
        The GP prior covariance kernel
        :math:`k(\\mathbf{x}, \\mathbf{x}')`.
    likelihood : _GaussianLikelihoodBase
        Gaussian likelihood used both for training and for adding
        observation noise to predictive distributions.
    train_x : torch.Tensor
        Initial training inputs of shape ``(N, D)``. Can be updated at
        prediction time via :meth:`predict`.
    train_y : torch.Tensor
        Initial training targets of shape ``(N,)``. Can be updated at
        prediction time via :meth:`predict`.

    Attributes
    ----------
    mean_module : Mean
        The prior mean function.
    covar_module : Kernel
        The prior covariance kernel.

    Notes
    -----
    Calls :class:`~gpytorch.models.ExactGP`'s ``__init__`` with the
    training data and likelihood, then stores the mean and covariance
    modules as sub-modules so their parameters are included in the
    model's parameter tree.

    :meth:`predict_prior` and :meth:`predict_posterior` both call
    :meth:`predict` internally but differ in the PyTorch training mode
    used: ``train=True`` for the prior (no conditioning on context) and
    ``train=False`` for the posterior (exact GP conditioning). Mode
    switching is handled by :meth:`_set_mode`, which restores the
    previous mode even if an exception is raised.
    """

    def __init__(
        self,
        mean_module: Mean,
        covar_module: Kernel,
        likelihood: _GaussianLikelihoodBase,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> None: # pylint: disable=arguments-differ
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward( # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> MultivariateNormal:
        """Computes the GP prior distribution at inputs ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(*, N, D)``.
        **kwargs : Any
            Additional keyword arguments forwarded to the mean and
            covariance modules.

        Returns
        -------
        MultivariateNormal
            The GP prior distribution over function values at *x*, with
            mean of shape ``(*, N)`` and covariance of shape
            ``(*, N, N)``.
        """
        mean_x = cast(torch.Tensor, self.mean_module(x))
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def predict(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor | None = None,
    ) -> MultivariateNormal:
        """Update training data and return the noisy predictive
        distribution.

        Sets the model's training data to *(x_context, y_context)*,
        evaluates the GP at *x_target*, and passes the result through
        the likelihood to obtain the noise-inclusive predictive
        distribution. The training mode (prior vs. posterior) is
        determined by the caller. Use :meth:`predict_prior` or
        :meth:`predict_posterior` for the common cases.

        Parameters
        ----------
        x_context : torch.Tensor
            Context (conditioning) inputs of shape ``(N_c, D)``.
        y_context : torch.Tensor
            Context (conditioning) targets of shape ``(N_c,)``.
        x_target : torch.Tensor, optional
            Target inputs at which to evaluate the predictive
            distribution of shape ``(N_t, D)``.
            Defaults to *x_context* when ``None``.

        Returns
        -------
        MultivariateNormal
            Noisy predictive distribution (likelihood-convolved) at
            *x_target*, with mean of shape ``(N_t,)`` and covariance of
            shape ``(N_t, N_t)``.

        Raises
        ------
        AssertionError
            If ``self.likelihood`` is ``None``.

        Notes
        -----
        Training data is updated in-place via
        :meth:`~gpytorch.models.ExactGP.set_train_data` with
        ``strict=False``, which allows the context set to differ in size
        from the original *train_x* supplied at construction.
        """
        assert self.likelihood is not None, "Likelihood is not set"

        if x_target is None:
            x_target = x_context

        self.set_train_data(inputs=x_context, targets=y_context, strict=False)

        distribution = self(x_target)
        distribution_with_noise = self.likelihood(distribution)

        return cast(MultivariateNormal, distribution_with_noise)

    def predict_prior(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor
    ) -> MultivariateNormal:
        """Return the noisy GP prior distribution at the context inputs.

        Evaluates the prior (i.e. without conditioning on context
        observations) by temporarily switching both the model and
        likelihood to training mode. The context data is still used to
        set the training inputs, but no posterior conditioning is
        performed.

        Parameters
        ----------
        x_context : torch.Tensor
            Context inputs of shape ``(N_c, D)``.
        y_context : torch.Tensor
            Context targets of shape ``(N_c,)``.

        Returns
        -------
        MultivariateNormal
            Noisy prior predictive distribution at *x_context*, with
            mean of shape ``(N_c,)`` and covariance of shape
            ``(N_c, N_c)``.
        """
        with self._set_mode(train=True):
            return self.predict(x_context, y_context, x_context)

    def predict_posterior(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
    ) -> MultivariateNormal:
        """Return the noisy GP posterior distribution at target inputs.

        Conditions the GP on *(x_context, y_context)* and evaluates the
        resulting posterior predictive distribution at *x_target* by
        temporarily switching both the model and likelihood to eval
        mode.

        Parameters
        ----------
        x_context : torch.Tensor
            Context (conditioning) inputs of shape ``(N_c, D)``.
        y_context : torch.Tensor
            Context (conditioning) targets of shape ``(N_c,)``.
        x_target : torch.Tensor
            Target inputs of shape ``(N_t, D)``.

        Returns
        -------
        MultivariateNormal
            Noisy posterior predictive distribution at *x_target*, with
            mean of shape ``(N_t,)`` and covariance of shape
            ``(N_t, N_t)``.
        """
        with self._set_mode(train=False):
            return self.predict(x_context, y_context, x_target)

    @contextmanager
    def _set_mode(self, train: bool) -> Generator[None, None, None]:
        """Context manager that temporarily sets the model and
        likelihood mode.

        Saves the current training flags for both ``self`` and
        ``self.likelihood``, switches them to *train*, yields control,
        then unconditionally restores the original flags, even if an
        exception is raised inside the ``with`` block.

        Parameters
        ----------
        train : bool
            If ``True``, switches to training mode (prior).
            If ``False``, switches to eval mode (posterior).

        Yields
        ------
        None
            Control is yielded to the ``with`` block with model and
            likelihood in the requested mode.

        Raises
        ------
        AssertionError
            If ``self.likelihood`` is ``None``.
        """
        assert self.likelihood is not None, "Likelihood is not set."
        prev_model = self.training
        prev_likelihood = self.likelihood.training
        try:
            self.train(train)
            self.likelihood.train(train)
            yield
        finally:
            self.train(prev_model)
            self.likelihood.train(prev_likelihood)

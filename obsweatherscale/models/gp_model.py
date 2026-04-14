from contextlib import contextmanager
from typing import Any, cast, Generator

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.means import Mean
from gpytorch.models import ExactGP


class GPModel(ExactGP):
    """Gaussian Process model class."""

    def __init__(
        self,
        mean_module: Mean,
        covar_module: Kernel,
        likelihood: _GaussianLikelihoodBase,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> None: # pylint: disable=arguments-differ
        """Initialize the GPModel.

        Parameters
        ----------
        train_x : torch.Tensor
            The training input data.
        train_y : torch.Tensor
            The training output data.
        likelihood : _GaussianLikelihoodBase
            The likelihood function for the model.
        mean_module : Mean
            The mean module
        covar_module : Kernel
            The covariance module (kernel)
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward( # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        **kwargs: Any
    ) -> MultivariateNormal:
        mean_x = cast(torch.Tensor, self.mean_module(x))
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def predict(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor | None = None,
    ) -> MultivariateNormal:
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
        with self._set_mode(train=True):
            return self.predict(x_context, y_context, x_context)

    def predict_posterior(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_target: torch.Tensor,
    ) -> MultivariateNormal:
        with self._set_mode(train=False):
            return self.predict(x_context, y_context, x_target)

    @contextmanager
    def _set_mode(self, train: bool) -> Generator[None, None, None]:
        assert self.likelihood is not None, "Likelihood is not set"
        prev_model = self.training
        prev_likelihood = self.likelihood.training
        try:
            self.train(train)
            self.likelihood.train(train)
            yield
        finally:
            self.train(prev_model)
            self.likelihood.train(prev_likelihood)

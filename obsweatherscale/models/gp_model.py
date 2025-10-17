from typing import Any

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
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: _GaussianLikelihoodBase,
        mean_module: Mean,
        covar_module: Kernel,
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

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor, **kwargs: Any) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)  # type: ignore

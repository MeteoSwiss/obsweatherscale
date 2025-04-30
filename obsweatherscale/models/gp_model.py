from typing import Any

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ExactGP


class GPModel(ExactGP):
    """Gaussian Process model class."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: _GaussianLikelihoodBase,
        modules: dict,
    ) -> None:
        """Initialize the GPModel.

        Parameters
        ----------
        train_x : torch.Tensor
            The training input data.
        train_y : torch.Tensor
            The training output data.
        likelihood : _GaussianLikelihoodBase
            The likelihood function for the model.
        modules : dict
            A dictionary containing the mean and covariance modules.
        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = modules["mean_module"]
        self.covar_module = modules["covar_module"]

    def forward(self, x: torch.Tensor, **kwargs: Any) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

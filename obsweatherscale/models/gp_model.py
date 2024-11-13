import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ExactGP


class GPModel(ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: _GaussianLikelihoodBase,
        modules: dict,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = modules['mean_module']
        self.covar_module = modules['covar_module']

    def forward(self, x: torch.Tensor, **kwargs) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

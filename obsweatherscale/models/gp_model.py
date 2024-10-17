import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP

class GPModel(ExactGP):
    def __init__(
        self,
        mean_function,
        kernel,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood
    ):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_function
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

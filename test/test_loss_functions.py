from typing import Any

import gpytorch
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood

import obsweatherscale as ows


def test_crps_normal() -> None:
    # Test the CRPS normal function
    mu = torch.tensor([0.0])
    sigma = torch.tensor([1.0])
    y = torch.tensor([0.5])

    crps_value = ows.crps_normal(y, mu, sigma)

    assert crps_value.shape == (), "CRPS value shape mismatch"
    assert crps_value.item() > 0, "CRPS value should be positive"


def test_crps_loss_fn() -> None:

    likelihood = GaussianLikelihood()
    mu = torch.tensor([0.0])
    sigma = torch.tensor([1.0])
    y = torch.tensor([0.5])
    dist = MultivariateNormal(mu, covariance_matrix=torch.diag_embed(sigma**2))

    loss_fn = ows.make_crps_loss(likelihood)

    loss_value = loss_fn(dist, y)

    assert loss_value.shape == (), "Loss value shape mismatch"
    assert loss_value.item() > 0, "Loss value should be positive"


def test_mll_loss_fn() -> None:
    # Prepare training data
    train_x = torch.randn(10, 3)
    train_y = torch.randn(10, 1)

    # Define a simple Exact GP model
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(
            self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            likelihood: GaussianLikelihood,
        ) -> None:
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        # pylint: disable=arguments-differ
        def forward(
            self,
            x: torch.Tensor,
            **kwargs: Any,
        ) -> gpytorch.distributions.MultivariateNormal:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(
                mean_x, covar_x  # type: ignore
            )

    # Instantiate GP model, likelihood and the loss, then wrap it
    likelihood = GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    loss_fn = ows.make_mll_loss(mll)

    # Generate a distribution to test
    mu = torch.tensor([0.0])
    sigma = torch.tensor([1.0])
    y = torch.tensor([0.5])
    dist = MultivariateNormal(mu, covariance_matrix=torch.diag_embed(sigma**2))

    # Compute the loss and check its properties
    loss_value = loss_fn(dist, y)
    assert loss_value.shape == (), "Loss value shape mismatch"
    assert loss_value.item() > 0, "Loss value should be positive"

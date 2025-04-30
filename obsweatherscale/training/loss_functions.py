from typing import Callable, cast, Optional

import torch
import torch.distributions as dist
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase


def crps_normal(obs: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Computes the Continuous Ranked Probability Score (CRPS)
    for a normal distribution.

    Parameters:
        obs (torch.Tensor): Observed values.
        mu (torch.Tensor): Mean of the normal distribution.
        sigma (torch.Tensor): std of the normal distribution.

    Returns:
        torch.Tensor: The CRPS for the normal distribution.
    """
    # Create normal distribution
    normal_dist = dist.Normal(mu, sigma)

    # Compute cumul distr function and prob density function
    cdf_obs = normal_dist.cdf(obs)
    pdf_obs = normal_dist.log_prob(obs).exp()

    # Compute CRPS
    term1 = (obs - mu) * (2 * cdf_obs - 1)
    term2 = 2 * sigma * pdf_obs - 1 / torch.sqrt(torch.tensor(torch.pi))
    crps = term1 + term2

    return crps.mean()


def crps_normal_loss_fct(likelihood: _GaussianLikelihoodBase | None = None) -> Callable:
    """Create a CRPS loss function for normal distributions that
    handles missing values and optionally transforms the distribution.
    
    """

    def loss_fct(distribution: MultivariateNormal, obs: torch.Tensor) -> torch.Tensor:
        mask = torch.isnan(obs)
        obs = torch.where(mask, 0.0, obs)

        if likelihood is not None:
            distribution = cast(MultivariateNormal, likelihood(distribution))

        mu = torch.where(mask, 0.0, distribution.mean)
        sigma = torch.where(mask, 1 / torch.sqrt(torch.tensor(torch.pi)), distribution.stddev)
        return crps_normal(obs, mu, sigma)

    return loss_fct


def mll_loss_fct(mll: ExactMarginalLogLikelihood):
    """Return a loss function that computes the negative log-likelihood
    of a multivariate normal distribution, optionally transformed by a
    likelihood function.
    
    """

    def loss_fct(distribution: MultivariateNormal, obs: torch.Tensor) -> torch.Tensor:
        log_likelihood = mll(distribution, obs)
        if isinstance(log_likelihood, torch.Tensor):
            return -log_likelihood.mean()

        raise TypeError(f"Expected mll to return a torch.Tensor, "
                        f"got {type(log_likelihood)}")

    return loss_fct

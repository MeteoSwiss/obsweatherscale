from typing import Callable, cast

import torch
import torch.distributions as dist
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase


def crps_normal(
    obs: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor
) -> torch.Tensor:
    """Wrapper to compute the Continuous Ranked Probability Score (CRPS)
    for a Normal distribution.

    Parameters
    ----------
    obs : torch.Tensor 
        Observed values.
    mu : torch.Tensor
        Mean of the Normal distribution.
    sigma : torch.Tensor
        Standard deviation of the Normal distribution.

    Returns
    -------
    torch.Tensor
        The CRPS for the normal distribution, averaged across all
        observations.

    Notes
    -----
    The CRPS is a proper scoring rule that measures the compatibility
    between a probability distribution and an observation. It's defined
    as the integrated squared difference between the CDF of the forecast
    distribution and the empirical CDF of the observation.
    
    For a normal distribution, the CRPS has the closed form:
    CRPS(N(μ, σ), y) = σ * [y_norm * (2*Φ(y_norm) - 1) + 2*φ(y_norm) - 1/√π]
    where y_norm = (y - μ)/σ, Φ is the CDF and φ is the PDF of the
    standard normal.
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


def crps_normal_loss_fct(
    likelihood: _GaussianLikelihoodBase | None = None
) -> Callable:
    """Wrapper to create a CRPS loss function for normal distributions
    that handles missing values and optionally transforms the
    distribution.
    
    Parameters
    ----------
    likelihood : _GaussianLikelihoodBase or None, optional
        A Gaussian likelihood transformation to apply to the
        distribution. If provided, transforms the distribution before
        computing the CRPS.
    
    Returns
    -------
    Callable
        A function that computes the CRPS loss between a multivariate
        Normal distribution and observed values.
    
    Notes
    -----
    The returned loss function handles missing values by masking them
    and treats them specially in the computation. For missing values,
    the parameters are set to produce a neutral contribution to the loss.
    """
    def loss_fct(
        distribution: MultivariateNormal,
        obs: torch.Tensor
    ) -> torch.Tensor:
        mask = torch.isnan(obs)
        obs = torch.where(mask, 0.0, obs)

        if likelihood is not None:
            distribution = cast(MultivariateNormal, likelihood(distribution))

        mu = torch.where(mask, 0.0, distribution.mean)
        sigma = torch.where(
            mask, 1 / torch.sqrt(torch.tensor(torch.pi)), distribution.stddev
        )
        return crps_normal(obs, mu, sigma)
    return loss_fct


def mll_loss_fct(mll: ExactMarginalLogLikelihood):
    """Wrapper to create a negative log-likelihood loss function
    of a multivariate normal distribution, optionally transformed by a
    likelihood function.
    
    Parameters
    ----------
    mll : ExactMarginalLogLikelihood
        The marginal log likelihood object that computes the log
        likelihood of the observations given the distribution.
    
    Returns
    -------
    Callable
        A function that computes the negative log likelihood loss
        between a multivariate normal distribution and observed values.
    
    Notes
    -----
    The returned loss function negates and averages the log likelihood
    to create a loss suitable for minimization in optimization problems.
    
    Raises
    ------
    TypeError
        If the mll doesn't return a torch.Tensor.
    """
    def loss_fct(
        distribution: MultivariateNormal,
        obs: torch.Tensor
    ) -> torch.Tensor:
        log_likelihood = mll(distribution, obs)
        if isinstance(log_likelihood, torch.Tensor):
            return -log_likelihood.mean()

        raise TypeError(
            f"Expected mll to return a torch.Tensor, "
            f"got {type(log_likelihood)}"
        )
    return loss_fct

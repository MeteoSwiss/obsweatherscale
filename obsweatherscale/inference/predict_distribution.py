import torch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import Normal

from obsweatherscale.likelihoods import TransformedGaussianLikelihood


def predict_posterior(
    model,
    likelihood: TransformedGaussianLikelihood,
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    target_x: torch.Tensor,
    noise: bool = True
) -> MultivariateNormal:
    model.eval()
    likelihood.eval()

    model.set_train_data(
        inputs=context_x,
        targets=context_y,
        strict=False
    )
    post_distribution = model(target_x)
    if noise:
        return likelihood(post_distribution)
    return post_distribution


def predict_prior(
    model,
    likelihood: TransformedGaussianLikelihood,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    noise: bool = True
) -> MultivariateNormal:
    model.train()
    likelihood.train()

    model.set_train_data(
        inputs=target_x,
        targets=target_y,
        strict=False
    )
    prior_distribution = model(target_x)
    if noise:
        return likelihood(prior_distribution)
    return prior_distribution


def marginal(distribution: MultivariateNormal) -> Normal:
    return Normal(distribution.mean, distribution.stddev)

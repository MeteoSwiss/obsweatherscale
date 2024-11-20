import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from torch.distributions import Normal

from obsweatherscale.likelihoods import TransformedGaussianLikelihood


def predict_posterior(
    model: ExactGP,
    likelihood: TransformedGaussianLikelihood,
    context_x: torch.Tensor,
    context_y: torch.Tensor,
    target_x: torch.Tensor
) -> MultivariateNormal:
    model.eval()
    likelihood.eval()

    model.set_train_data(inputs=context_x, targets=context_y, strict=False)
    post_distribution = model(target_x)

    return likelihood(post_distribution)


def predict_prior(
    model: ExactGP,
    likelihood: TransformedGaussianLikelihood,
    target_x: torch.Tensor,
    target_y: torch.Tensor
) -> MultivariateNormal:
    model.train()
    likelihood.train()

    model.set_train_data(inputs=target_x, targets=target_y, strict=False)
    prior_distribution = model(target_x)

    return likelihood(prior_distribution)


def marginal(distribution: MultivariateNormal) -> Normal:
    return Normal(distribution.mean, distribution.stddev)

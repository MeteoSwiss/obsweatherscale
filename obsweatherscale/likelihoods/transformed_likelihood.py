import math
from typing import Any, Union

import gpytorch.settings as settings
import torch
from gpytorch.distributions import base_distributions, MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from linear_operator.operators import LinearOperator, MaskedLinearOperator

from .noise_models import TransformedNoise


class TransformedGaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(
        self,
        noise_covar: TransformedNoise,
        **kwargs: Any,
    ) -> None:
        super().__init__(noise_covar=noise_covar)

    def _shaped_noise_covar(
        self,
        y: torch.Tensor,
        *params: Any, **kwargs: Any
    ) -> Union[torch.Tensor, LinearOperator]:
        base_shape = y.shape
        return self.noise_covar(*params, y=y, shape=base_shape, **kwargs)
    
    def expected_log_prob(
        self, target: torch.Tensor,
        input: MultivariateNormal,
        *params: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        noise = self._shaped_noise_covar(input.mean, *params, **kwargs).diagonal(dim1=-1, dim2=-2)
        # Potentially reshape the noise to deal with the multitask case
        noise = noise.view(*noise.shape[:-1], *input.event_shape)

        # Handle NaN values if enabled
        nan_policy = settings.observation_nan_policy.value()
        if nan_policy == "mask":
            observed = settings.observation_nan_policy._get_observed(target, input.event_shape)
            input = MultivariateNormal(
                mean=input.mean[..., observed],
                covariance_matrix=MaskedLinearOperator(
                    input.lazy_covariance_matrix, observed.reshape(-1), observed.reshape(-1)
                ),
            )
            noise = noise[..., observed]
            target = target[..., observed]
        elif nan_policy == "fill":
            missing = torch.isnan(target)
            target = settings.observation_nan_policy._fill_tensor(target)

        mean, variance = input.mean, input.variance
        res = ((target - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)

        if nan_policy == "fill":
            res = res * ~missing

        # Do appropriate summation for multitask Gaussian likelihoods
        num_event_dim = len(input.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))

        return res

    def forward(
        self,
        function_samples: torch.Tensor,
        *params: Any,
        **kwargs: Any
    ) -> torch.distributions.Normal:
        noise = self._shaped_noise_covar(function_samples.mean,
                                         *params, **kwargs).diagonal(dim1=-1, dim2=-2)
        return base_distributions.Normal(function_samples, noise.sqrt())

    def marginal(
        self,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any
    ) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)

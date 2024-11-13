import math
from typing import Any, Union

import torch
from gpytorch import ExactMarginalLogLikelihood, settings
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
        base_shape: torch.Size,
        *params: Any,
        y: torch.Tensor = None,
        **kwargs: Any
    ) -> Union[torch.Tensor, LinearOperator]:
        # If `y` is provided, derive `base_shape` from `y`
        if y is not None:
            base_shape = y.shape
        return self.noise_covar(*params, y=y, shape=base_shape, **kwargs)

    def expected_log_prob(
        self, target: torch.Tensor,
        input: MultivariateNormal,
        *params: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        noise = self._shaped_noise_covar(
            input.mean.shape, *params, y=input.mean, **kwargs
        ).diagonal(dim1=-1, dim2=-2)
        # Potentially reshape the noise to deal with multitask case
        noise = noise.view(*noise.shape[:-1], *input.event_shape)

        # Handle NaN values if enabled
        nan_policy = settings.observation_nan_policy.value()
        if nan_policy == "mask":
            observed = settings.observation_nan_policy._get_observed(
                target, input.event_shape
            )
            input = MultivariateNormal(
                mean=input.mean[..., observed],
                covariance_matrix=MaskedLinearOperator(
                    input.lazy_covariance_matrix,
                    observed.reshape(-1),
                    observed.reshape(-1)
                ),
            )
            noise = noise[..., observed]
            target = target[..., observed]
        elif nan_policy == "fill":
            mask = torch.isnan(target)
            cov = input.covariance_matrix
            cov_masked = torch.where(
                mask.unsqueeze(-1) + mask.unsqueeze(-1).mT, 0.0, cov
            )

            input = MultivariateNormal(
                mean=torch.where(mask, 0.0, input.mean),
                covariance_matrix=torch.where(
                    torch.diag_embed(mask), 1/(2*torch.pi), cov_masked
                )
            )
            noise = torch.where(mask, 0.0, noise)
            target = torch.where(mask, 0.0, target)

        mean, variance = input.mean, input.variance
        res = ((target - mean).square() + variance) \
            / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)

        if nan_policy == "fill":
            res = res * ~mask

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
        noise = self._shaped_noise_covar(
            function_samples.mean.shape,
            *params,
            y=function_samples.mean,
            **kwargs
        ).diagonal(dim1=-1, dim2=-2)
        return base_distributions.Normal(function_samples, noise.sqrt())

    def marginal(
        self,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any
    ) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(
            mean.shape, *params, y=mean, **kwargs
        )
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)


class ExactMarginalLogLikelihoodFill(ExactMarginalLogLikelihood):
    def forward(
        self,
        function_dist: _GaussianLikelihoodBase,
        target: torch.Tensor,
        *params: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        r"""
        Computes the MLL given :math:`p(\mathbf f)`
        and :math:`\mathbf y`.

        :param ~gpytorch.distributions._GaussianLikelihoodBase function_dist:
            :math:`p(\mathbf f)` the outputs of the latent function
            (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape
          of the model/input data.
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError(
                "ExactMarginalLogLikelihoodFill can only "
                "operate on Gaussian random variables"
            )

        # Determine output likelihood
        output = self.likelihood(function_dist, *params, **kwargs)

        # Remove NaN values if enabled
        if settings.observation_nan_policy.value() == "mask":
            observed = settings.observation_nan_policy._get_observed(
                target, output.event_shape
            )
            output = MultivariateNormal(
                mean=output.mean[..., observed],
                covariance_matrix=MaskedLinearOperator(
                    output.lazy_covariance_matrix,
                    observed.reshape(-1),
                    observed.reshape(-1)
                ),
            )
            target = target[..., observed]
        elif settings.observation_nan_policy.value() == "fill":
            mask = torch.isnan(target)
            cov = output.covariance_matrix
            cov_masked = torch.where(
                mask.unsqueeze(-1) + mask.unsqueeze(-1).mT, 0.0, cov
            )

            output = MultivariateNormal(
                mean=torch.where(mask, 0.0, output.mean),
                covariance_matrix=torch.where(
                    torch.diag_embed(mask), 1/(2*torch.pi), cov_masked
                )
            )
            target = torch.where(mask, 0.0, target)

        # Get the log prob of the marginal distribution
        res = output.log_prob(target)
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)

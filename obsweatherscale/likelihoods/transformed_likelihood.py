import math
from typing import Any

import torch
from gpytorch import ExactMarginalLogLikelihood, settings
from gpytorch.distributions import (
    base_distributions, MultivariateNormal
)
from gpytorch.likelihoods import _GaussianLikelihoodBase
from linear_operator.operators import (
    LinearOperator, MaskedLinearOperator
)

from obsweatherscale.likelihoods.noise_models import TransformedNoise


def _handle_nan_policy(
    target: torch.Tensor,
    distr: MultivariateNormal,
) -> tuple[MultivariateNormal, torch.Tensor, torch.Tensor]:
    """Apply the configured NaN handling policy to the target and
    input.

    This method modifies the distribution and target tensor
    based on the active `settings.observation_nan_policy` value.
    Supports masking or filling missing observations.

    Parameters
    ----------
    target : torch.Tensor
        The target values for which NaNs may be present.
    distr : MultivariateNormal
        The predicted distribution corresponding to the target. Its
        covariance and mean may be modified to handle NaNs.

    Returns
    -------
    tuple[MultivariateNormal, torch.Tensor, torch.Tensor]
        - distr: A modified MultivariateNormal distribution with
        NaNs handled according to the policy.
        - target: A tensor with NaNs masked or filled according to
        the policy.
        - mask: A tensor indicating which values to mask in the
        resulting log prob.

    Notes
    -----
    Policies:
    - "mask": Remove (ignore) all observations where `target` is
    NaN. This uses `MaskedLinearOperator` to exclude those entries
    from the covariance.
    - "fill": Replace NaN values with zeros in `target` and adjust
    the covariance accordingly, assigning small variance (1 / 2π) to
    missing entries.

    This method ensures that subsequent log-probability computations
    do not fail due to NaN values in the target tensor.
    """

    nan_policy = settings.observation_nan_policy.value()

    if nan_policy == "mask":
        # pylint: disable=protected-access
        observed = settings.observation_nan_policy._get_observed(
            target, distr.event_shape,
        )
        distr = MultivariateNormal(
            mean=distr.mean[..., observed],
            covariance_matrix=MaskedLinearOperator(
                distr.lazy_covariance_matrix,  # type: ignore
                observed.reshape(-1),
                observed.reshape(-1),
            ),
        )
        target = target[..., observed]
        mask = torch.zeros_like(target)

    elif nan_policy == "fill":
        mask = torch.isnan(target)
        cov = distr.covariance_matrix
        cov_masked = torch.where(
            mask.unsqueeze(-1) + mask.unsqueeze(-1).mT,
            0.0,
            cov,  # type: ignore
        )

        distr = MultivariateNormal(
            mean=torch.where(mask, 0.0, distr.mean),
            covariance_matrix=torch.where(
                torch.diag_embed(mask),
                1 / (2 * torch.pi),
                cov_masked,
            ),
        )
        target = torch.where(mask, 0.0, target)

    return distr, target, mask


class TransformedGaussianLikelihood(_GaussianLikelihoodBase):
    """A Gaussian likelihood with a transformed noise model.

    This likelihood allows custom noise transformations by leveraging
    a `TransformedNoise` object. It supports missing observation
    handling policies, including masking and filling, and is designed
    for compatibility with GPyTorch models using `MultivariateNormal`
    outputs.

    Methods
    -------
    _shaped_noise_covar(base_shape, *params, y=None, **kwargs)
        Returns a noise covariance object of the correct shape,
        optionally based on `y`.

    expected_log_prob(target, input, *params, **kwargs)
        Computes the expected log probability of the target under the
        input distribution, optionally handling NaNs according to the
        active policy.

    forward(function_samples, *params, **kwargs)
        Returns a Normal distribution by applying transformed noise to
        function samples.

    marginal(function_dist, *params, **kwargs)
        Returns the marginal distribution including the transformed
        noise covariance.
    """

    def __init__(
        self,
        noise_covar: TransformedNoise,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        noise_covar : TransformedNoise
            A callable noise covariance module implementing a
            transformation strategy for noise models.
        **kwargs : Any
            Additional keyword arguments passed to the base
            `_GaussianLikelihoodBase`.

        Returns
        -------
        """
        super().__init__(noise_covar=noise_covar)

    def expected_log_prob(
        self,
        target: torch.Tensor,
        input: MultivariateNormal, # pylint: disable=redefined-builtin
        *params: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Expected log probability of target given input distribution.

        Computes the expected log probability of the `target` given the
        `input` distribution, accounting for noise and handling missing
        values according to the configured NaN policy.

        Parameters
        ----------
        target : torch.Tensor
            The target tensor to compute the log probability for.
        input : MultivariateNormal
            The input distribution, typically a MultivariateNormal, from
            which the mean and variance will be used to compute the log
            probability.
        *params : tuple
            Additional parameters passed to the likelihood.
        **kwargs : Any
            Additional keyword arguments passed to the likelihood.

        Returns
        -------
        torch.Tensor
            The expected log probability for each observation, reduced
            along event dimensions for multitask outputs if needed.

        Notes
        -----
        - Missing values in `target` are handled according to the active
        `settings.observation_nan_policy`:
            - "mask": excludes NaNs from computation using a masked covariance.
            - "fill": replaces NaNs with zeros and assigns small variance.
        - Noise variance is computed using the transformed likelihood's
        noise model (`self._shaped_noise_covar`) and scaled according to
        any target transformation.
        - The final log probability is computed as the Gaussian log-likelihood:
            log P(y | μ, σ^2) = -0.5 * [(y - μ)^2 / σ^2 + log σ^2 + log 2π]
        - For multitask outputs, the log probabilities are summed across
        event dimensions.
        """
        noise = self._shaped_noise_covar(
            input.mean.shape, *params, y=input.mean, **kwargs
        ).diagonal(dim1=-1, dim2=-2)

        # Reshape noise to deal with multitask cases
        noise = noise.view(*noise.shape[:-1], *input.event_shape)

        # Handle NaN values according to active policy
        input, target, mask = _handle_nan_policy(target, input)

        mean, variance = input.mean, input.variance

        res = -0.5 * (
            ((target - mean).square() + variance) / noise
            + noise.log()
            + math.log(2 * math.pi)
        )
        res *= ~mask  # type: ignore

        # Do appropriate summation for multitask cases
        num_event_dim = len(input.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))

        return res

    def forward(
        self,
        function_samples: torch.Tensor,
        *params: Any,
        **kwargs: Any,
    ) -> torch.distributions.Normal:
        """Applies noise to given function samples.

        Returns a Normal distribution by applying the transformed noise
        to the given function samples.

        Parameters
        ----------
        function_samples : torch.Tensor
            The samples from the function (typically the mean of a
            MultivariateNormal).
        *params : tuple
            Additional parameters passed to the noise covariance
            function.
        **kwargs : Any
            Additional keyword arguments passed to the likelihood.

        Returns
        -------
        torch.distributions.Normal
            The Normal distribution obtained after adding noise to the
            function samples.
        """
        noise = self._shaped_noise_covar(
            function_samples.mean.shape,  # type: ignore
            *params,
            y=function_samples.mean,  # type: ignore
            **kwargs,
        ).diagonal(dim1=-1, dim2=-2)
        return base_distributions.Normal(function_samples, noise.sqrt())

    def marginal(
        self,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any,
    ) -> MultivariateNormal:
        """Computes the marginal distribution by adding the noise
        covariance to the covariance of the function distribution.

        Parameters
        ----------
        function_dist : MultivariateNormal
            The MultivariateNormal distribution representing the
            function's posterior.
        *params : tuple
            Additional parameters passed to the noise covariance
            function.
        **kwargs : Any
            Additional keyword arguments passed to the likelihood.

        Returns
        -------
        MultivariateNormal
            The marginal distribution, which is the sum of the
            function's covariance and the noise covariance.
        """
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(
            mean.shape, *params, y=mean, **kwargs
        )
        full_covar = covar + noise_covar  # type: ignore
        return function_dist.__class__(mean, full_covar)

    def _shaped_noise_covar(
        self,
        base_shape: torch.Size,
        *params: Any,
        y: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | LinearOperator:
        """
        Returns the noise covariance of the appropriate shape, based on
        the provided `base_shape` and optional target `y`.

        Parameters
        ----------
        base_shape : torch.Size
            The base shape of the noise covariance matrix. If `y` is
            provided, the shape will be inferred from `y`.
        *params : tuple
            Additional parameters passed to noise covariance function.
        y : torch.Tensor, optional
            The target tensor whose shape will be used to infer the
            noise covariance. Default is None.
        **kwargs : Any
            Additional keyword arguments passed to the noise covariance
            function.

        Returns
        -------
        torch.Tensor or LinearOperator
            The noise covariance, either as tensor or LinearOperator.
        """
        if y is not None:
            base_shape = y.shape
        return self.noise_covar(*params, y=y, shape=base_shape, **kwargs)


class ExactMarginalLogLikelihoodFill(ExactMarginalLogLikelihood):
    """Extension of ExactMarginalLogLikelihood with support for handling
    NaN values by filling.

    This class extends the ExactMarginalLogLikelihood class to handle
    missing values (represented as NaNs) through a filling mechanism.
    It computes the exact marginal log likelihood for Gaussian
    Processes, which is crucial for model selection and hyperparameter
    optimization in GP models.

    Parameters
    ----------
    model : gpytorch.models.GP
        The GP model for which to compute the marginal log likelihood.
    likelihood : gpytorch.likelihoods.Likelihood
        The likelihood for the GP model.

    Attributes
    ----------
    model : gpytorch.models.GP
        The GP model.
    likelihood : gpytorch.likelihoods.Likelihood
        The likelihood for the GP model.

    Notes
    -----
    This implementation properly handles NaN values in the target tensor
    based on the observation_nan_policy setting, with special handling
    for the "fill" policy.
    """

    def forward(
        self,
        function_dist: _GaussianLikelihoodBase,
        target: torch.Tensor,
        *params: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Computes the marginal log likelihood given a function
        distribution and target values.

        This method calculates the marginal log likelihood by evaluating
        the probability of the observed data given the model. It
        supports special handling for NaN values in the target tensor
        based on the active `settings.observation_nan_policy` value.

        Parameters
        ----------
        function_dist : gpytorch.distributions._GaussianLikelihoodBase
            The distribution p(f) representing outputs of the latent
            function.
        target : torch.Tensor
            The target values y to condition on.
        *params : Any
            Additional positional arguments to pass to the likelihood.
        **kwargs : Any
            Additional keyword arguments to pass to the likelihood.

        Returns
        -------
        torch.Tensor
            The computed marginal log likelihood divided by the number
            of data points. Output shape corresponds to the batch shape
            of the model/input data.

        Raises
        ------
        RuntimeError
            If function_dist is not a MultivariateNormal distribution.

        Notes
        -----
        The method handles NaN values in the target tensor according to
        the observation_nan_policy setting:
        - "mask": Remove (ignore) all observations where `target` is
        NaN. This uses `MaskedLinearOperator` to exclude those entries
        from the covariance.
        - "fill": Replace NaN values with zeros in `target` and adjust
        the covariance accordingly, assigning small variance (1 / 2π) to
        missing entries.
        """
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError(
                "ExactMarginalLogLikelihoodFill can only "
                "operate on Gaussian random variables"
            )

        # Determine output likelihood
        output = self.likelihood(function_dist, *params, **kwargs)

        # Handle NaN values according to active policy
        output, target, _ = _handle_nan_policy(target, output)

        # Get the log prob of the marginal distribution
        res = output.log_prob(target)
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)

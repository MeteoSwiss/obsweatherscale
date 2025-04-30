import functools
from typing import Union
import warnings

import torch
from gpytorch import settings
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.utils.memoize import cached, clear_cache_hook
from linear_operator import to_linear_operator
from linear_operator.operators import LinearOperator, MaskedLinearOperator
from torch import Tensor


class DefaultPredictionStrategyFill(DefaultPredictionStrategy):

    @cached(name="mean_cache")
    def _mean_cache(self, nan_policy: str) -> Tensor:
        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar = mvn.loc, mvn.lazy_covariance_matrix

        train_labels_offset = (self.train_labels - train_mean).unsqueeze(-1)

        if nan_policy == "ignore":
            mean_cache = train_train_covar.evaluate_kernel().solve(train_labels_offset).squeeze(-1)
        elif nan_policy == "mask":
            # Mask all rows and columns in the kernel matrix
            # corresponding to the missing observations.
            observed = settings.observation_nan_policy._get_observed(self.train_labels,
                                                                     torch.Size((self.train_labels.shape[-1], )))
            mean_cache = torch.full_like(self.train_labels, torch.nan)
            kernel = MaskedLinearOperator(train_train_covar.evaluate_kernel(), observed.reshape(-1),
                                          observed.reshape(-1))
            mean_cache[..., observed] = kernel.solve(train_labels_offset[..., observed, :]).squeeze(-1)
        else:  # 'fill'
            # Fill all rows and columns in the kernel matrix
            # corresponding to the missing observations with 0. Don't
            # touch the corresponding diagonal elements to ensure a
            # unique solution. This ensures that missing data is ignored
            # during solving.
            warnings.warn(
                "Observation NaN policy 'fill' makes the kernel matrix "
                "dense during exact prediction.",
                RuntimeWarning,
            )
            kernel = train_train_covar.evaluate_kernel()
            missing = torch.isnan(self.train_labels)
            kernel_mask = (~missing).to(torch.float)
            kernel_mask = kernel_mask[..., None] * kernel_mask[..., None, :]
            torch.diagonal(kernel_mask, dim1=-2, dim2=-1)[...] = 1
            kernel = kernel * kernel_mask  # makes kernel dense atm :(
            torch.diagonal(kernel_mask, dim1=-2, dim2=-1)[...] = 1 / (2 * torch.pi)  # TODO
            train_labels_offset = settings.observation_nan_policy._fill_tensor(train_labels_offset)  # TODO
            mean_cache = kernel.solve(train_labels_offset).squeeze(-1)
            mean_cache[missing] = torch.nan  # Ensure nobody expects these values to be valid.
        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache

    def exact_predictive_mean(self, test_mean: Tensor,
                              test_train_covar: LinearOperator) -> Union[Tensor, LinearOperator]:
        """
        Computes the posterior predictive covariance of a GP

        :param Tensor test_mean: The test prior mean
        :param linear_operator.operators.LinearOperator test_train_covar:
            Covariance matrix between test and train inputs
        :return: The predictive posterior mean of the test points
        """
        mean_cache = self.mean_cache
        if len(mean_cache.shape) == 4:
            mean_cache = mean_cache.squeeze(1)

        # Handle NaNs
        nan_policy = settings.observation_nan_policy.value()
        if nan_policy == "ignore":
            res = (test_train_covar @ mean_cache.unsqueeze(-1)).squeeze(-1)
        elif nan_policy == "mask":
            # Restrict train dimension to observed values
            observed = settings.observation_nan_policy._get_observed(mean_cache, torch.Size((mean_cache.shape[-1], )))
            full_mask = torch.ones(test_mean.shape[-1], dtype=torch.bool, device=test_mean.device)
            test_train_covar = MaskedLinearOperator(to_linear_operator(test_train_covar), full_mask,
                                                    observed.reshape(-1))
            res = (test_train_covar @ mean_cache[..., observed].unsqueeze(-1)).squeeze(-1)
        else:  # 'fill'
            # Set the columns corresponding to missing observations to 0
            # to ignore them during matmul.
            mask = (~torch.isnan(mean_cache)).to(torch.float)[..., None, :]
            test_train_covar = test_train_covar * mask
            # TODO: gets filled with -999.0 instead of 0
            # mean = settings.observation_nan_policy._fill_tensor(mean_cache)
            mean = torch.nan_to_num(mean_cache, nan=0.0)
            res = (test_train_covar @ mean.unsqueeze(-1)).squeeze(-1)
        res = res + test_mean

        return res

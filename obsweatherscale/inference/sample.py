from typing import Union

import torch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import Normal

Distribution = Union[MultivariateNormal, Normal]

def sample(
    distribution,
    n_samples: int
) -> torch.Tensor:
    if distribution.mean.dim() < 2 or distribution.mean.dim() > 3:
        raise ValueError(
            f"Data should be of shape [batch_size, n_points]"
            f" or [batch_size, n_points, n_vars]"
        )
    sample_size = torch.Size((n_samples,))
    samples = distribution.sample(sample_size)
    # Add "variable" dimension to make it robust for multitask
    # shape is now [n_samples, batch_size, n_points, n_variables]
    if samples.dim() == 3:
        samples = samples.unsqueeze(-1)
    return samples.permute(1, 2, 3, 0)  # samples dimension last

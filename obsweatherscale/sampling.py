from typing import TypeAlias

import torch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import Normal

Distribution: TypeAlias = MultivariateNormal | Normal


def sample(distribution: Distribution, n_samples: int) -> torch.Tensor:
    """Draw samples from a Gaussian process or normal distribution.

    Parameters
    ----------
    distribution : MultivariateNormal or Normal
        The distribution to sample from. Can be either a GPyTorch
        MultivariateNormal (typically for GP outputs) or a standard
        torch Normal distribution.
    n_samples : int
        The number of independent samples to draw from the distribution.

    Returns
    -------
    torch.Tensor
        A tensor of shape [batch_size, n_points, n_variables, n_samples]
        containing the drawn samples. The last dimension corresponds to
        the sampled realizations, while the other dimensions match the
        input distribution’s batch and event shapes. A singleton
        variable dimension is added automatically for univariate cases
        to ensure consistent tensor shape for multitask settings.

    Raises
    ------
    ValueError
        If the input distribution’s mean tensor does not have 2 or 3
        dimensions, which are expected to correspond to
        [batch_size, n_points] or [batch_size, n_points, n_variables].

    Notes
    -----
    - For a MultivariateNormal with shape
      [batch_size, n_points, n_variables], the output tensor shape is
      [batch_size, n_points, n_variables, n_samples].
    - For a univariate distribution (2D mean), a singleton variable
      dimension is added automatically for consistency.
    - The function permutes the dimensions so that the sample axis is
      last, which is often convenient for computing statistics over
      multiple samples without reshaping.

    Examples
    --------
    >>> from gpytorch.distributions import MultivariateNormal
    >>> mvn = MultivariateNormal(
            torch.zeros(5, 10),
            torch.eye(10).expand(5, 10, 10),
        )
    >>> samples = sample(mvn, n_samples=100)
    >>> samples.shape
    torch.Size([5, 10, 1, 100])
    """
    if distribution.mean.dim() < 2 or distribution.mean.dim() > 3:
        raise ValueError(
            "Data should be of shape [batch_size, n_points]"
            " or [batch_size, n_points, n_vars]"
        )

    sample_size = torch.Size([n_samples])
    samples = distribution.sample(sample_size)

    # shape is now [n_samples, batch_size, n_points, n_variables]
    if samples.dim() == 3:
        samples = samples.unsqueeze(-1)

    return samples.permute(1, 2, 3, 0)  # samples dimension last

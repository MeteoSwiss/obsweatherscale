from typing import Any, Optional

import torch
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.priors import Prior


class ScaledRBFKernel(Kernel):
    def __init__(
        self,
        variance: Optional[torch.Tensor] = None,
        lengthscale: Optional[torch.Tensor] = None,
        ard_num_dims: Optional[int] = None,
        batch_shape: Optional[torch.Size] = None,
        active_dims: Optional[list[int]] = None,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[torch.nn.Module] = None,
        outputscale_prior: Optional[Prior] = None,
        outputscale_constraint: Optional[torch.nn.Module] = None,
        train_lengthscale: bool = True,
        train_variance: bool = True,
        eps: float = 1e-06,
        **kwargs: Any
    ) -> None:
        super().__init__()

        if (
            active_dims is not None
            and lengthscale is not None
            and len(lengthscale) != len(active_dims)
        ):
            raise ValueError(
                "The length of 'lengthscale' must match"
                " the length of 'active_dims'"
            )

        if lengthscale is not None:
            ard_num_dims = len(lengthscale)

        if not train_lengthscale and lengthscale is None:
            raise ValueError(
                "A lengthscale value must be provided"
                " if lengthscale is not trainable"
            )

        if not train_variance and variance is None:
            raise ValueError(
                "A variance value must be provided "
                "if variance is not trainable"
            )

        self.rbf_kernel = RBFKernel(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
            eps=eps,
            **kwargs
        )

        # Set lengthscale
        if lengthscale is not None:
            self.rbf_kernel.lengthscale = lengthscale
        self.rbf_kernel.raw_lengthscale.requires_grad = train_lengthscale

        self.kernel = ScaleKernel(
            self.rbf_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint
        )

        # Set variance
        if variance is not None:
            self.kernel.outputscale = variance
        self.kernel.raw_outputscale.requires_grad = train_variance

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *params: Any,
        **kwargs: Any
    ) -> torch.Tensor:
        return self.kernel(x1, x2, *params, **kwargs)

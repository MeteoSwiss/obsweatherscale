from typing import Any, Optional, Union

import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.priors import Prior
from linear_operator.operators import LinearOperator
from torch import nn


class ScaledRBFKernel(Kernel):
    """Scaled RBF Kernel class.

    This class implements a scaled radial basis function (RBF) kernel
    with the option to set the lengthscale and variance parameters.

    """
    def __init__(
        self,
        variance: Optional[torch.Tensor] = None,
        lengthscale: Optional[torch.Tensor] = None,
        ard_num_dims: Optional[int] = None,
        batch_shape: Optional[torch.Size] = None,
        active_dims: Optional[tuple[int, ...]] = None,
        lengthscale_prior: Optional[Prior] = None,
        lengthscale_constraint: Optional[Interval] = None,
        outputscale_prior: Optional[Prior] = None,
        outputscale_constraint: Optional[Interval] = None,
        train_lengthscale: bool = True,
        train_variance: bool = True,
        eps: float = 1e-06,
        **kwargs: Any
    ) -> None:
        """Initialize the ScaledRBFKernel.

        Parameters
        ----------
        variance : torch.Tensor, optional
            The variance parameter for the kernel.
        lengthscale : torch.Tensor, optional
            The lengthscale parameter for the kernel.
        ard_num_dims : int, optional
            The number of dimensions for the lengthscale parameter.
        batch_shape : torch.Size, optional
            The batch shape of the kernel.
        active_dims : tuple[int, ...], optional
            The active dimensions for the kernel.
        lengthscale_prior : Prior, optional
            The prior for the lengthscale parameter.
        lengthscale_constraint : Interval, optional
            The constraint for the lengthscale parameter.
        outputscale_prior : Prior, optional
            The prior for the outputscale parameter.
        outputscale_constraint : Interval, optional
            The constraint for the outputscale parameter.
        train_lengthscale : bool, default=True
            Whether to train the lengthscale parameter.
        train_variance : bool, default=True
            Whether to train the variance parameter.
        eps : float, default=1e-06
            A small value to prevent numerical instability.
        **kwargs : Any
            Additional keyword arguments for the kernel.
        """
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

        rbf_kernel = RBFKernel(
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
            rbf_kernel.raw_lengthscale = nn.Parameter(
                rbf_kernel.raw_lengthscale_constraint.inverse_transform(lengthscale),
                requires_grad=train_lengthscale
            )
        rbf_kernel.raw_lengthscale.requires_grad = train_lengthscale

        self.kernel = ScaleKernel(
            rbf_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint
        )

        # Set variance
        if variance is not None:
            self.kernel.raw_outputscale = nn.Parameter(
                self.kernel.raw_outputscale_constraint.inverse_transform(variance),
                requires_grad=train_variance
            )
        self.kernel.raw_outputscale.requires_grad = train_variance
    
    def extra_repr(self):
        return '\n'.join(
            [
                f"(lengthscale): {self.kernel.base_kernel.lengthscale}",
                f"(variance): {self.kernel.outputscale}",
                f"(num_dims): {self.kernel.base_kernel.ard_num_dims}"
            ]
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *params: Any,
        **kwargs: Any
    ) -> Union[torch.Tensor, LinearOperator]:
        return self.kernel(x1, x2, *params, **kwargs)

import torch
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel

class ScaledRBFKernel(Kernel):
    def __init__(
        self,
        variance=None,
        lengthscale=None,
        ard_num_dims=None,
        batch_shape=None,
        active_dims=None,
        lengthscale_prior=None,
        lengthscale_constraint=None,
        outputscale_prior=None,
        outputscale_constraint=None,
        train_lengthscale: bool = True,
        train_variance: bool = True,
        eps=1e-06,
        **kwargs
    ):
        super().__init__()

        if active_dims is not None and lengthscale is not None and len(lengthscale) != len(active_dims):
            raise ValueError("The length of 'lengthscale' must match the length of 'active_dims'")

        if lengthscale is not None:
            ard_num_dims = len(lengthscale)

        if not train_lengthscale and lengthscale is None:
            raise ValueError("A lengthscale value must be provided if lengthscale is not trainable")

        if not train_variance and variance is None:
            raise ValueError("A variance value must be provided if variance is not trainable")

        self.rbf_kernel = RBFKernel(ard_num_dims=ard_num_dims,
                                    batch_shape=batch_shape,
                                    active_dims=active_dims,
                                    lengthscale_prior=lengthscale_prior,
                                    lengthscale_constraint=lengthscale_constraint,
                                    eps=eps,
                                    **kwargs)
        
        # Set lengthscale
        if lengthscale is not None:
            self.rbf_kernel.lengthscale = lengthscale
        self.rbf_kernel.raw_lengthscale.requires_grad = train_lengthscale

        self.kernel = ScaleKernel(self.rbf_kernel,
                                  outputscale_prior=outputscale_prior,
                                  outputscale_constraint=outputscale_constraint)

        # Set variance
        if variance is not None:
            self.kernel.outputscale = variance
        self.kernel.raw_outputscale.requires_grad = train_variance

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *params,
        **kwargs
    ) -> torch.Tensor:
        return self.kernel(x1, x2, *params, **kwargs)

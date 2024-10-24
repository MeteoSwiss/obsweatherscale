from typing import Any, Optional

import torch
from gpytorch.likelihoods.noise_models import (FixedGaussianNoise,
                                               HeteroskedasticNoise,
                                               HomoskedasticNoise,
                                               Noise)
from linear_operator.operators import (ConstantDiagLinearOperator,
                                       DiagLinearOperator,
                                       ZeroLinearOperator)

from ..transformations.transformer import Transformer


class TransformedNoise(Noise):
    def __init__(
        self,
        transformer: Transformer
    ):
        Noise.__init__(self)
        self.transformer = transformer
    
    def forward(
        self,
        pure_noise_var: torch.Tensor,
        y: torch.Tensor
    ) -> DiagLinearOperator:
        noise_diag = pure_noise_var \
                     * self.transformer.noise_transform(y) ** 2
        return DiagLinearOperator(noise_diag)


class TransformedHomoskedasticNoise(
    TransformedNoise,
    HomoskedasticNoise
):
    def __init__(
        self,
        transformer: Transformer,
        noise_prior=None,
        noise_constraint=None,
        batch_shape=torch.Size()
    ):
        HomoskedasticNoise.__init__(
            self,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape
            )
        TransformedNoise.__init__(self, transformer)

    def forward(
        self,
        *params: Any,
        y: torch.Tensor,
        shape: Optional[torch.Size] = None,
        **kwargs: Any
    ) -> DiagLinearOperator:
        pure_noise_var = super(
            HomoskedasticNoise,
            self
        ).forward(*params, shape=shape, **kwargs).diag
        return super(
            TransformedNoise,
            self
        ).forward(pure_noise_var, y)


class TransformedHeteroskedasticNoise(
    TransformedNoise,
    HeteroskedasticNoise
):
    def __init__(
        self,
        transformer: Transformer,
        noise_model,
        noise_indices=None,
        noise_constraint=None
    ):
        HeteroskedasticNoise.__init__(
            self,
            noise_model=noise_model,
            noise_indices=noise_indices,
            noise_constraint=noise_constraint
        )
        TransformedNoise.__init__(self, transformer)

    def forward(
        self,
        *params: Any,
        y: torch.Tensor,
        batch_shape: Optional[torch.Size] = None,
        shape: Optional[torch.Size] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> DiagLinearOperator:
        pure_noise_var = super(
            HeteroskedasticNoise,
            self
        ).forward(
            *params,
            batch_shape=batch_shape,
            shape=shape,
            noise=noise
        ).diag
        return super(
            TransformedNoise,
            self
        ).forward(pure_noise_var, y)


class TransformedFixedGaussianNoise(
    TransformedNoise,
    FixedGaussianNoise
):
    def __init__(
        self,
        transformer: Transformer,
        obs_noise_var: torch.Tensor | int | float = torch.tensor(1.0)
    ):
        if not torch.is_tensor(obs_noise_var):
            obs_noise_var = torch.tensor(obs_noise_var)
        super().__init__(transformer)
        FixedGaussianNoise.__init__(self, obs_noise_var)

    def forward(
        self,
        *params: Any,
        y: torch.Tensor,
        shape: Optional[torch.Size] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> DiagLinearOperator:
        # Determine shape
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]
        batch_shape, n = shape

        # Get pure noise
        # noise is provided
        if noise is not None:  
            pure_noise_var = DiagLinearOperator(noise).diag()

        # self.noise is a scalar, we need to broadcast
        elif self.noise.numel() == 1:  
            noise_diag = self.noise.expand((batch_shape, 1))
            pure_noise_var = ConstantDiagLinearOperator(
                noise_diag, diag_shape=n
            ).diag()

        # self.noise is same shape as "shape"
        elif shape[-1] == self.noise.shape[-1]:
            pure_noise_var = DiagLinearOperator(self.noise).diag()
            
        # no noise provided AND self.noise has wrong shape: noise is 0
        else:  
            pure_noise_var = ZeroLinearOperator().diag()

        # Transform noise
        return super().forward(pure_noise_var, y)

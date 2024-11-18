import abc
from typing import Any, Optional

import torch
from gpytorch.likelihoods.noise_models import (
    FixedGaussianNoise, HeteroskedasticNoise, HomoskedasticNoise, Noise
)
from gpytorch.priors import Prior
from linear_operator.operators import (
    ConstantDiagLinearOperator, DiagLinearOperator, ZeroLinearOperator
)

from ..transformations.transformer import Transformer


class TransformedNoise(Noise):
    def __init__(
        self,
        transformer: Transformer
    ) -> None:
        Noise.__init__(self)
        self.transformer = transformer

    def transform_noise(
        self,
        pure_noise_var: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> DiagLinearOperator:
        if y is None:
            y = 0
        noise_diag = pure_noise_var * self.transformer.noise_transform(y) ** 2
        return DiagLinearOperator(noise_diag)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class TransformedHomoskedasticNoise(TransformedNoise, HomoskedasticNoise):
    def __init__(
        self,
        transformer: Transformer,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[torch.nn.Module] = None,
        batch_shape: torch.Size = torch.Size()
    ) -> None:
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
        y: Optional[torch.Tensor] = None,
        shape: Optional[torch.Size] = None,
        **kwargs: Any
    ) -> DiagLinearOperator:
        pure_noise_var = super(
            HomoskedasticNoise, self
        ).forward(*params, shape=shape, **kwargs).diagonal()

        return self.transform_noise(pure_noise_var=pure_noise_var, y=y)


class TransformedHeteroskedasticNoise(TransformedNoise, HeteroskedasticNoise):
    def __init__(
        self,
        transformer: Transformer,
        noise_model,
        noise_indices: Optional[list[int]] = None,
        noise_constraint: Optional[torch.nn.Module] = None
    ) -> None:
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
        y: Optional[torch.Tensor] = None,
        batch_shape: Optional[torch.Size] = None,
        shape: Optional[torch.Size] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> DiagLinearOperator:
        pure_noise_var = super(HeteroskedasticNoise, self).forward(
            *params,
            batch_shape=batch_shape,
            shape=shape,
            noise=noise
        ).diag

        return self.transform_noise(pure_noise_var=pure_noise_var, y=y)


class TransformedFixedGaussianNoise(TransformedNoise, FixedGaussianNoise):
    def __init__(
        self,
        transformer: Transformer,
        obs_noise_var: torch.Tensor | int | float = torch.tensor(1.0)
    ) -> None:
        if not torch.is_tensor(obs_noise_var):
            obs_noise_var = torch.tensor(obs_noise_var)
        super().__init__(transformer)
        FixedGaussianNoise.__init__(self, obs_noise_var)

    def forward(
        self,
        *params: Any,
        y: Optional[torch.Tensor] = None,
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
            pure_noise_var = DiagLinearOperator(noise).diagonal()

        # self.noise is a scalar, we need to broadcast
        elif self.noise.numel() == 1:
            noise_diag = self.noise.expand((batch_shape, 1))
            pure_noise_var = ConstantDiagLinearOperator(
                noise_diag, diag_shape=n
            ).diagonal()

        # self.noise is same shape as "shape"
        elif shape[-1] == self.noise.shape[-1]:
            pure_noise_var = DiagLinearOperator(self.noise).diagonal()

        # no noise provided AND self.noise has wrong shape: noise is 0
        else:
            pure_noise_var = ZeroLinearOperator().diagonal()

        # Transform noise
        return self.transform_noise(pure_noise_var=pure_noise_var, y=y)

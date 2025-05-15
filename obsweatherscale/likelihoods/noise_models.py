import abc
from typing import Any

import torch
from gpytorch.likelihoods.noise_models import (
    FixedGaussianNoise,
    HeteroskedasticNoise,
    HomoskedasticNoise,
    Noise,
)
from gpytorch.priors import Prior
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    DiagLinearOperator,
    ZeroLinearOperator,
)

from ..transformations.transformer import Transformer


class TransformedNoise(Noise):
    """Noise model incorporating a transformation of the target data.

    This is useful when the model is trained on transformed targets and
    the noise must be expressed in the transformed space (e.g., log).It
    ensures that the chosen noise model is scaled appropriately under
    the transformation.

    Attributes
    ----------
    transformer : Transformer
        Object representing the data transformation applied to the
        targets.

    Methods
    -------
    transform_noise(pure_noise_var, y=None) -> DiagLinearOperator
        Applies the transformation to the noise variance based on the
        target data.

    forward(*args, **kwargs)
        Abstract method to be implemented by subclasses.
    """

    def __init__(self, transformer: Transformer) -> None:
        """Initialize the TransformedNoise model with data transformer.

        Parameters
        ----------
        transformer : Transformer
            Object representing the transformation to be applied to
            the target data.
        """
        Noise.__init__(self)
        self.transformer = transformer

    def transform_noise(
        self,
        pure_noise_var: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> DiagLinearOperator:
        """Transform noise variance to reflect target data
        transformation.

        Applies the square of the local derivative of the inverse
        transformation to the predicted noise variance. This ensures
        that the noise is properly scaled in the transformed target
        space.

        Parameters
        ----------
        pure_noise_var : torch.Tensor
            Variance of the pure noise before transformation.

        y : torch.Tensor, optional
            Target data to be used in the transformation. If not
            provided, a zero tensor is used, assuming a transformation
            that is independent of the target values.

        Returns
        -------
        DiagLinearOperator
            A diagonal linear operator representing the transformed
            noise variance.

        Raises
        ------
        AttributeError
            If the transformer does not implement `noise_transform`.
        """
        if y is None:
            y = torch.tensor(0)

        noise_diag = pure_noise_var * self.transformer.noise_transform(y) ** 2
        return DiagLinearOperator(noise_diag)

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> DiagLinearOperator:
        """Abstract method for the forward pass.

        Must be implemented in subclasses to define how the noise model
        produces a covariance structure from input data.

        Parameters
        ----------
        *args
            Positional arguments for the forward pass.

        **kwargs
            Keyword arguments for the forward pass.

        Raises
        ------
        NotImplementedError
            To be raised if the method is not implemented in a subclass.
        """


class TransformedHomoskedasticNoise(TransformedNoise, HomoskedasticNoise):
    """A homoskedastic noise model that accounts for target
    transformations.

    This model assumes constant noise across all inputs (homoskedastic),
    but ensures that the noise is properly transformed to reflect the
    transformed target space.

    Homoskedastic noise is appropriate when all observations are assumed
    to have the same level of uncertainty, such as in well-controlled
    experiments or uniformly sampled data. A transformed noise model is
    appropriate when it is trained on transformed outputs, and noise
    must be adjusted accordingly.

    Inherits from:
    - `TransformedNoise`: Applies the appropriate correction for target
    transformations.
    - `HomoskedasticNoise`: Represents fixed noise variance across all
    inputs.

    Attributes
    ----------
    transformer : Transformer
        A transformation object (e.g., log, standard scaler, etc.) that
        defines the transformation applied to the target variable.

    noise_prior : Prior or None
        Optional prior over the noise variance parameter.

    noise_constraint : nn.Module or None
        Optional constraint to ensure valid noise variance
        (e.g., positivity).

    batch_shape : torch.Size
        The batch shape for the noise model parameters.
    """

    def __init__(
        self,
        transformer: Transformer,
        noise_prior: Prior | None = None,
        noise_constraint: torch.nn.Module | None = None,
        batch_shape: torch.Size = torch.Size(),
    ) -> None:
        """Initializes the TransformedHomoskedasticNoise model.

        Sets up both the trainable constant noise module and the
        transformation-aware wrapper that ensures proper scaling in
        transformed target space.

        Parameters
        ----------
        transformer : Transformer
            A callable or object representing the transformation applied
            to the target variable (e.g., logarithmic, standardization).

        noise_prior : Prior, optional
            Prior distribution placed over the constant noise variance.

        noise_constraint : nn.Module, optional
            Optional constraint to ensure valid noise variance
            (e.g., positivity).

        batch_shape : torch.Size, optional
            The shape of batches for which independent noise parameters
            are maintained.
        """
        HomoskedasticNoise.__init__(
            self,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
        )
        TransformedNoise.__init__(self, transformer)

    def forward(
        self,
        *params: Any,
        y: torch.Tensor | None = None,
        shape: torch.Size | None = None,
        **kwargs: Any,
    ) -> DiagLinearOperator:
        """Computes transformed homoskedastic noise as linear operator.

        Produces a constant diagonal noise covariance matrix and applies
        a correction based on the transformation of the target variable.

        Parameters
        ----------
        *params : Any
            Parameters passed to the underlying noise model.

        y : torch.Tensor, optional
            Target values used to transform the noise variance. Required
            if the transformation is data-dependent.

        shape : torch.Size, optional
            The desired shape of the output covariance matrix.

        **kwargs : Any
            Additional keyword arguments passed to internal components.

        Returns
        -------
        DiagLinearOperator
            A diagonal linear operator representing the transformed
            noise covariance matrix. This is useful in Gaussian process
            models where efficient matrix representations are needed.
        """
        pure_noise_var = HomoskedasticNoise.forward(
            *params, shape=shape, **kwargs
        ).diagonal()

        return self.transform_noise(pure_noise_var=pure_noise_var, y=y)


class TransformedHeteroskedasticNoise(TransformedNoise, HeteroskedasticNoise):
    """A heteroskedastic noise model that accounts for target
    transformations.

    Heteroskedastic noise is data-dependent: it allows for a different
    noise variance for each data point. This model combines
    heteroskedastic noise modeling with a transformation-aware
    mechanism, allowing it to model noise in the transformed target
    space rather than the original.

    Heteroskedastic noise is useful when the variability of observations
    depends on the input, such as in sensor readings, financial data, or
    datasets with heterogeneity across the input space. A transformed
    noise model is appropriate when it is trained on transformed
    outputs, and noise must be adjusted accordingly.

    Inherits from both:
    - `TransformedNoise`: Handles transformations on noise in the target
    space.
    - `HeteroskedasticNoise`: Models variance as a function of inputs.

    Attributes
    ----------
    transformer : Transformer
        A transformation object (e.g., log, standard scaler, etc.) that
        defines the transformation applied to the target variable.

    noise_model : nn.Module
        A PyTorch module that outputs noise variances, typically
        conditioned on the input features.

    noise_indices : list[int] or None
        Optional list of indices indicating which input dimensions are
        used to model the noise. If None, all inputs are used.

    noise_constraint : nn.Module or None
        Optional constraint to ensure valid noise variance
        (e.g., positivity).
    """

    def __init__(
        self,
        transformer: Transformer,
        noise_model: torch.nn.Module,
        noise_indices: list[int] | None = None,
        noise_constraint: torch.nn.Module | None = None,
    ) -> None:
        """Initializes the TransformedHeteroskedasticNoise model.

        Sets up both the heteroskedastic noise module and the
        transformation-aware wrapper that ensures proper scaling in
        transformed target space.

        Parameters
        ----------
        transformer : Transformer
            A callable or object representing the transformation applied
            to the target variable (e.g., logarithmic, standardization).

        noise_model : nn.Module
            A module that maps inputs to noise variances.

        noise_indices : list[int], optional
            A list of indices specifying which input dimensions are used
            for modeling the noise. If None, all dimensions are used.

        noise_constraint : nn.Module, optional
            Optional constraint to ensure valid noise variance
            (e.g., positivity).
        """
        HeteroskedasticNoise.__init__(
            self,
            noise_model=noise_model,
            noise_indices=noise_indices,
            noise_constraint=noise_constraint,
        )
        TransformedNoise.__init__(self, transformer)

    def forward(
        self,
        *params: Any,
        y: torch.Tensor | None = None,
        batch_shape: torch.Size | None = None,
        shape: torch.Size | None = None,
        noise: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> DiagLinearOperator:
        """Computes transformed heteroskedastic noise as a linear
        operator.

        Computes input-dependent noise, applies transformation
        correction, and returns result as a diagonal linear operator.

        Parameters
        ----------
        *params : Any
            Parameters passed to the underlying noise model.

        y : torch.Tensor, optional
            Target values used to transform the noise variance. Required
            if the transformation is data-dependent.

        batch_shape : torch.Size, optional
            The desired batch shape of the output operator.

        shape : torch.Size, optional
            The desired shape of the output covariance matrix.

        noise : torch.Tensor, optional
            Optional fixed noise tensor to override the learned model
            output.

        **kwargs : Any
            Additional keyword arguments passed to internal components.

        Returns
        -------
        DiagLinearOperator
            A diagonal linear operator representing the transformed
            noise covariance matrix. This is useful in Gaussian process
            models where efficient matrix representations are needed.
        """
        pure_noise_var = HeteroskedasticNoise.forward(
            *params, batch_shape=batch_shape, shape=shape, noise=noise
        ).diagonal()

        return self.transform_noise(pure_noise_var=pure_noise_var, y=y)


class TransformedFixedGaussianNoise(TransformedNoise, FixedGaussianNoise):
    """A fixed Gaussian noise model that accounts for target
    transformations.

    This model assumes fixed (non-trainable) Gaussian noise across all
    inputs, but ensures that the noise is properly transformed to
    reflect the transformed target space.

    Fixed Gaussian noise is appropriate when the observation noise is
    known ahead of time or externally provided (e.g., sensor precision),
    and does not vary with input. A transformed noise model is useful
    when the model is trained on transformed targets, and noise must be
    adjusted accordingly.

    Inherits from:
    - `TransformedNoise`: Applies the appropriate correction for target
    transformations.
    - `FixedGaussianNoise`: Represents fixed (non-trainable) Gaussian
    noise variance across all inputs.

    Attributes
    ----------
    transformer : Transformer
        A transformation object (e.g., log, standard scaler, etc.) that
        defines the transformation applied to the target variable.
    """

    def __init__(
        self,
        transformer: Transformer,
        obs_noise_var: torch.Tensor | int | float = 1.0,
    ) -> None:
        """Initializes the TransformedFixedGaussianNoise model.

        Sets up both the fixed constant noise module and the
        transformation-aware wrapper that ensures proper scaling in
        transformed target space.

        Parameters
        ----------
        transformer : Transformer
            A callable or object representing the transformation applied
            to the target variable (e.g., logarithmic, standardization).

        obs_noise_var : torch.Tensor or float or int
            The fixed variance of the Gaussian noise. Can be a scalar,
            a tensor of shape (N,), or a tensor matching the shape of
            the input data.
        """
        if isinstance(obs_noise_var, torch.Tensor):
            obs_noise_var = obs_noise_var.clone().detach().float()
        else: # for scalars (int or float)
            obs_noise_var = torch.tensor(obs_noise_var, dtype=torch.float)

        super().__init__(transformer)
        FixedGaussianNoise.__init__(self, obs_noise_var)

    def extra_repr(self) -> str:
        """Returns a string representation of the noise value.

        Returns
        -------
        str
            A formatted string showing the stored fixed noise value.
        """
        return f"\n  (obs_noise_var): {self.noise}"

    def forward(
        self,
        *params: Any,
        y: torch.Tensor | None = None,
        shape: torch.Size | None = None,
        noise: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> DiagLinearOperator:
        """Computes transformed fixed Gaussian noise as a linear
        operator.

        Parameters
        ----------
        *params : Any
            Parameters passed to the underlying noise model.

        y : torch.Tensor, optional
            Target values used to transform the noise variance. Required
            if the transformation is data-dependent.

        shape : torch.Size, optional
            The desired shape of the output covariance matrix.

        noise :

        **kwargs : Any
            Additional keyword arguments passed to internal components.

        Returns
        -------
        DiagLinearOperator
            A diagonal linear operator representing the transformed
            noise covariance matrix. This is useful in Gaussian process
            models where efficient matrix representations are needed.
        """
        # Determine shape
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = (
                torch.Size(p.shape)
                if len(p.shape) == 1
                else torch.Size(p.shape[:-1])
            )
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

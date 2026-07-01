"""ScaledRBFKernel class.

This module provides :class:`ScaledRBFKernel`, a GPyTorch-compatible
kernel that wraps an :class:`~gpytorch.kernels.RBFKernel` inside a
:class:`~gpytorch.kernels.ScaleKernel` to produce a fully parameterised
squared-exponential kernel with controllable lengthscale(s) and output
variance. Both hyperparameters can be initialised to fixed values and
optionally frozen during optimisation.

Classes
-------
ScaledRBFKernel
    A scaled RBF kernel with optional ARD, priors, constraints, and
    parameter freezing.
"""

from typing import Any

import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.priors import Prior
from linear_operator.operators import LinearOperator


class ScaledRBFKernel(Kernel):
    """A scaled radial basis function (RBF) kernel.

    Wraps :class:`~gpytorch.kernels.RBFKernel` inside a
    :class:`~gpytorch.kernels.ScaleKernel` to implement the kernel:

    .. math::
        k(\\mathbf{x}_1, \\mathbf{x}_2) =
            \\sigma^2 \\exp\\!\\left(
                -\\frac{1}{2}
                (\\mathbf{x}_1 - \\mathbf{x}_2)^{\\top}
                \\mathbf{L}^{-2}
                (\\mathbf{x}_1 - \\mathbf{x}_2)
            \\right)

    where :math:`\\sigma^2` is the output variance and :math:`\\mathbf{L}`
    is a diagonal lengthscale matrix (scalar when ARD is disabled).

    Both hyperparameters can be initialised to a fixed value and
    optionally frozen (i.e. made non-trainable) during training, which
    is useful for partially-fixed kernel configurations.

    Parameters
    ----------
    variance : torch.Tensor, optional
        Initial value for the output variance :math:`\\sigma^2`.
        If ``None``, the GPyTorch default initialisation is used.
    lengthscale : torch.Tensor, optional
        Initial value for the lengthscale(s). A scalar tensor sets a
        single shared lengthscale; a 1-D tensor of length ``D``
        activates ARD with one lengthscale per dimension (and
        ``ard_num_dims`` is inferred automatically). If ``None``, the
        GPyTorch default initialisation is used.
    ard_num_dims : int, optional
        Number of ARD lengthscale dimensions. Ignored when *lengthscale*
        is a multi-element tensor, in which case ``ard_num_dims`` is
        inferred from ``lengthscale.numel()``.
    batch_shape : torch.Size, optional
        Batch shape for kernel, enabling independent kernel instances
        across a batch dimension (e.g. for multi-output GPs).
    active_dims : tuple[int, ...], optional
        Indices of the input dimensions this kernel should operate on.
        When provided alongside a multi-element *lengthscale*, the
        tensor length must match ``len(active_dims)``.
    lengthscale_prior : Prior, optional
        GPyTorch prior placed on the lengthscale parameter.
    lengthscale_constraint : Interval, optional
        GPyTorch constraint applied to the lengthscale parameter (e.g.
        :class:`~gpytorch.constraints.Positive`)
    outputscale_prior : Prior, optional
        GPyTorch prior placed on the output variance parameter.
    outputscale_constraint : Interval, optional
        GPyTorch constraint applied to the output variance parameter.
    train_lengthscale : bool, default=True
        If ``False``, the lengthscale is frozen and will not receive
        gradient updates. *lengthscale* must be provided when this is
        ``False``.
    train_variance : bool, default=True
        If ``False``, the output variance is frozen and will not receive
        gradient updates. *variance* must be provided when this is
        ``False``.
    eps : float, default=1e-6
        Numerical jitter added to the diagonal of the kernel matrix for
        stability. Forwarded directly to
        :class:`~gpytorch.kernels.RBFKernel`.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :class:`~gpytorch.kernels.RBFKernel`.

    Attributes
    ----------
    kernel : ScaleKernel
        The composed kernel: a :class:`~gpytorch.kernels.ScaleKernel`
        wrapping an :class:`~gpytorch.kernels.RBFKernel`. Access inner
        RBF kernel via ``self.kernel.base_kernel``.

    Raises
    ------
    ValueError
        If *active_dims* is provided alongside a multi-element
        *lengthscale* whose length does not match ``len(active_dims)``.
    ValueError
        If ``train_lengthscale=False`` but *lengthscale* is ``None`` (no
        initial value to freeze to).
    ValueError
        If ``train_variance=False`` but *variance* is ``None`` (no
        initial value to freeze to).

    Notes
    -----
    Freezing a parameter is implemented by calling
    ``requires_grad_(False)`` on the corresponding raw (unconstrained)
    parameter tensor. This means the parameter is still present in the
    model's ``state_dict`` and can be saved/loaded normally, but it will
    not appear in ``model.parameters()`` for gradient-based optimisers.
    """

    def __init__(
        self,
        variance: torch.Tensor | None = None,
        lengthscale: torch.Tensor | None = None,
        ard_num_dims: int | None = None,
        batch_shape: torch.Size | None = None,
        active_dims: tuple[int, ...] | None = None,
        lengthscale_prior: Prior | None = None,
        lengthscale_constraint: Interval | None = None,
        outputscale_prior: Prior | None = None,
        outputscale_constraint: Interval | None = None,
        train_lengthscale: bool = True,
        train_variance: bool = True,
        eps: float = 1e-06,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if (
            active_dims is not None
            and lengthscale is not None
            and lengthscale.numel() > 1
            and lengthscale.numel() != len(active_dims)
        ):
            raise ValueError(
                "`lengthscale` must be a scalar"
                " or its shape must match the shape of `active_dims`."
            )

        if not train_lengthscale and lengthscale is None:
            raise ValueError(
                "`lengthscale` must be provided "
                "if `train_lengthscale` is False."
            )

        if not train_variance and variance is None:
            raise ValueError(
                "`variance` must be provided if `train_variance` is False."
            )

        if lengthscale is not None and lengthscale.numel() > 1:
            ard_num_dims = lengthscale.numel()

        rbf_kernel = RBFKernel(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            active_dims=active_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
            eps=eps,
            **kwargs,
        )

        # Set lengthscale
        if lengthscale is not None:
            rbf_kernel.initialize(lengthscale=lengthscale)
        rbf_kernel.raw_lengthscale.requires_grad_(train_lengthscale)

        self.kernel = ScaleKernel(
            rbf_kernel,
            outputscale_prior=outputscale_prior,
            outputscale_constraint=outputscale_constraint,
        )

        # Set variance
        if variance is not None:
            self.kernel.initialize(outputscale=variance)
        self.kernel.raw_outputscale.requires_grad_(train_variance)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        *params: Any,
        **kwargs: Any,
    ) -> torch.Tensor | LinearOperator:
        return self.kernel(x1, x2, *params, **kwargs)

    def extra_repr(self) -> str:
        return "\n".join(
            [
                f"(lengthscale): {self.kernel.base_kernel.lengthscale}",
                f"(variance): {self.kernel.outputscale}",
                f"(num_dims): {self.kernel.base_kernel.ard_num_dims}",
            ]
        )

import pytest
import torch
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import LinearMean
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel

import obsweatherscale as ows


def test_gp_model() -> None:

    train_x = torch.randn(20, 3)
    train_y = torch.randn(20)

    likelihood = GaussianLikelihood(NormalPrior(0.0, 0.1))
    mean = LinearMean(3)
    kernel = RBFKernel(3)

    model = ows.GPModel(mean, kernel, likelihood, train_x, train_y)

    train_x_shape = model.train_inputs[0].shape # type: ignore
    train_y_shape = model.train_targets.shape   # type: ignore

    assert isinstance(model, ows.GPModel), "Model is not of type GPModel"
    assert train_x_shape == train_x.shape, "Train x shape mismatch"
    assert train_y_shape == train_y.shape, "Train y shape mismatch"

    # Check if the model can be called
    test_x = torch.randn(5, 3)
    model.eval()
    output = model(test_x)

    mean_shape = output.mean.shape # type: ignore
    cov_shape = output.covariance_matrix.shape # type: ignore

    assert mean_shape == (5,), "Output mean shape mismatch"
    assert cov_shape == (5, 5), "Output covariance shape mismatch"
    assert isinstance(output, torch.distributions.MultivariateNormal), \
        "Output type mismatch"


def test_mlp() -> None:
    # Test the MLP class
    mlp = ows.MLP([2,8,1], active_dims=[0,1])

    # Check the input and output dimensions
    test_input = torch.randn(5, 3)
    output = mlp(test_input)
    assert output.shape == (5, 1), "Output shape mismatch"


def make_nan_gp_model(nan_policy: str) -> ows.GPModel:
    torch.manual_seed(0)
    train_x = torch.randn(20, 3)
    train_y = torch.randn(20)
    return ows.GPModel(
        LinearMean(3),
        RBFKernel(3),
        GaussianLikelihood(),
        train_x,
        train_y,
        nan_policy=nan_policy,
    )


@pytest.mark.parametrize("nan_policy", ["fill", "mask"])
def test_predict_applies_nan_policy(nan_policy: str) -> None:
    model = make_nan_gp_model(nan_policy)
    active_policies = []
    forward = model.forward

    def spy_forward(x: torch.Tensor) -> MultivariateNormal:
        active_policies.append(settings.observation_nan_policy.value())
        return forward(x)

    model.forward = spy_forward  # type: ignore[method-assign]
    default_policy = settings.observation_nan_policy.value()

    x_context = torch.randn(10, 3)
    model.predict_prior(x_context, torch.randn(10))
    model.predict_posterior(x_context, torch.randn(10), torch.randn(5, 3))

    assert active_policies and set(active_policies) == {nan_policy}
    assert settings.observation_nan_policy.value() == default_policy


def test_predict_posterior_mask_ignores_nan_context_in_mean() -> None:
    model = make_nan_gp_model("mask")
    x_context = torch.randn(10, 3)
    y_context = torch.randn(10)
    y_context[[2, 7]] = torch.nan
    observed = ~torch.isnan(y_context)
    x_target = torch.randn(5, 3)

    with torch.no_grad():
        masked = model.predict_posterior(x_context, y_context, x_target)
        dropped = model.predict_posterior(
            x_context[observed], y_context[observed], x_target,
        )

    # Only the mean is compared: GPyTorch applies the "mask" policy to the
    # predictive mean but not to the predictive covariance.
    torch.testing.assert_close(masked.mean, dropped.mean)


@pytest.mark.parametrize("nan_policy", ["fill", "mask"])
def test_predict_posterior_finite_with_nan_context(nan_policy: str) -> None:
    model = make_nan_gp_model(nan_policy)
    y_context = torch.randn(10)
    y_context[[2, 7]] = torch.nan

    with torch.no_grad():
        output = model.predict_posterior(
            torch.randn(10, 3), y_context, torch.randn(5, 3),
        )

    assert torch.isfinite(output.mean).all()
    assert torch.isfinite(output.covariance_matrix).all()

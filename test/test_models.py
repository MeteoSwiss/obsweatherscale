import torch
from gpytorch.means import LinearMean
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel

from obsweatherscale.models import GPModel, MLP

def test_gp_model() -> None:

    train_x = torch.randn(20, 3)
    train_y = torch.randn(20)

    likelihood = GaussianLikelihood(NormalPrior(0.0, 0.1))
    mean = LinearMean(3)
    kernel = RBFKernel(3)

    model = GPModel(train_x, train_y, likelihood, mean, kernel)

    train_x_shape = model.train_inputs[0].shape # type: ignore
    train_y_shape = model.train_targets.shape

    assert isinstance(model, GPModel), "Model is not of type GPModel"
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
    mlp = MLP([2,8,1], active_dims=[0,1])

    # Check the input and output dimensions
    test_input = torch.randn(5, 3)
    output = mlp(test_input)
    assert output.shape == (5, 1), "Output shape mismatch"

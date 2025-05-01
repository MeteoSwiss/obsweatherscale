import torch
from gpytorch.means import LinearMean
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel

from obsweatherscale.models import GPModel, MLP

def test_gp_model():

    train_x = torch.randn(20, 3)
    train_y = torch.randn(20)

    likelihood = GaussianLikelihood(NormalPrior(0.0, 0.1))
    mean = LinearMean(3)
    kernel = RBFKernel(3)

    model = GPModel(train_x, train_y, likelihood, mean, kernel)

    assert isinstance(model, GPModel), "Model is not of type GPModel"
    assert model.train_inputs[0].shape == train_x.shape, "Train x shape mismatch"
    assert model.train_targets.shape == train_y.shape, "Train y shape mismatch"


    # Check if the model can be called
    test_x = torch.randn(5, 3)
    model.eval()
    output = model(test_x)
    assert output.mean.shape == (5,), "Output mean shape mismatch"
    assert output.covariance_matrix.shape == (5, 5), "Output covariance shape mismatch"
    assert isinstance(output, torch.distributions.MultivariateNormal), "Output type mismatch"


def test_mlp():
    # Test the MLP class
    mlp = MLP([2,8,1], active_dims=[0,1])

    # Check the input and output dimensions
    test_input = torch.randn(5, 3)
    output = mlp(test_input)
    assert output.shape == (5, 1), "Output shape mismatch"
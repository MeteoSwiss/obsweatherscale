import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import ExactMarginalLogLikelihood
import gpytorch
from gpytorch.distributions import MultivariateNormal

from obsweatherscale.training.loss_functions import crps_normal, crps_normal_loss_fct, mll_loss_fct

def test_crps_normal():
    # Test the CRPS normal function
    mu = torch.tensor([0.0])
    sigma = torch.tensor([1.0])
    y = torch.tensor([0.5])

    crps_value = crps_normal(y, mu, sigma)

    assert crps_value.shape == (), "CRPS value shape mismatch"
    assert crps_value.item() > 0, "CRPS value should be positive"


def test_crps_normal_loss_fct():

    likelihood = GaussianLikelihood()
    mu = torch.tensor([0.0])
    sigma = torch.tensor([1.0])
    y = torch.tensor([0.5])
    dist = MultivariateNormal(mu, covariance_matrix=torch.diag_embed(sigma**2))
    loss_fct = crps_normal_loss_fct(likelihood)

    loss_value = loss_fct(dist, y)

    assert loss_value.shape == (), "Loss value shape mismatch"
    assert loss_value.item() > 0, "Loss value should be positive"


def test_mll_loss_fct():

    # Prepare training data
    train_x = torch.randn(10, 3)
    train_y = torch.randn(10, 1)

    # Define a simple Exact GP model
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # Instantiate GP model, likelihood and the loss, then wrap it
    likelihood = GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    loss_fct = mll_loss_fct(mll)

    # Generate a distribution to test
    mu = torch.tensor([0.0])
    sigma = torch.tensor([1.0])
    y = torch.tensor([0.5])
    dist = MultivariateNormal(mu, covariance_matrix=torch.diag_embed(sigma**2))

    # Compute the loss and check its properties
    loss_value = loss_fct(dist, y)
    assert loss_value.shape == (), "Loss value shape mismatch"
    assert loss_value.item() > 0, "Loss value should be positive"
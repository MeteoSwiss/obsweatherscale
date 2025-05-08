import torch
from gpytorch.kernels import RBFKernel

from obsweatherscale.kernels import NeuralKernel, ScaledRBFKernel


def test_neural_kernel():
    # Create a simple neural network
    net = torch.nn.Sequential(
        torch.nn.Linear(3, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2)
    )

    # Create the neural kernel
    neural_kernel = NeuralKernel(net, RBFKernel(ard_num_dims=2))

    # Generate some random input data
    x1 = torch.randn(5, 3)
    x2 = torch.randn(5, 3)

    # Compute the kernel output
    output = neural_kernel(x1, x2).evaluate() # type: ignore

    assert output.shape == (5, 5), "Output shape mismatch"
    assert isinstance(output, torch.Tensor), "Output type mismatch"
    assert output.dtype == torch.float32, "Output dtype mismatch"


def test_scaled_rbf_kernel():
    # Create a scaled RBF kernel
    scaled_kernel = ScaledRBFKernel(torch.tensor(1.5), torch.tensor([2., 0.5]))

    # Generate some random input data
    x1 = torch.randn(5, 2)
    x2 = torch.randn(5, 2)

    # Compute the kernel output
    output = scaled_kernel(x1, x2).evaluate() # type: ignore

    assert output.shape == (5, 5), "Output shape mismatch"
    assert isinstance(output, torch.Tensor), "Output type mismatch"
    assert output.dtype == torch.float32, "Output dtype mismatch"

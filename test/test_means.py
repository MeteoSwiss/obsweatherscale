import torch

from obsweatherscale.means import NeuralMean


def test_neural_mean() -> None:
    # Create a simple neural network
    net = torch.nn.Sequential(
        torch.nn.Linear(3, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2)
    )

    # Create the neural mean
    neural_mean = NeuralMean(net)

    # Generate some random input data
    x = torch.randn(5, 3)

    # Compute the mean output
    output = neural_mean(x)

    assert output.shape == (5, 2), "Output shape mismatch"  # type: ignore
    assert isinstance(output, torch.Tensor), "Output type mismatch"
    assert output.dtype == torch.float32, "Output dtype mismatch"

import torch

from obsweatherscale.transformations import (
    Standardizer, QuantileFittedTransformer
)


def test_standardizer():
    # Generate some random input data
    x = torch.randn(5, 3)

    # Create a simple Standardizer
    standardizer = Standardizer(x, 0)

    # Transform the data
    transformed_x = standardizer.transform(x)

    assert transformed_x.shape == x.shape, "Transformed shape mismatch"
    assert isinstance(transformed_x, torch.Tensor), "Transformed type mismatch"
    assert transformed_x.dtype == torch.float32, "Transformed dtype mismatch"

    # Inverse transform the data
    inverse_transformed_x = standardizer.inverse_transform(transformed_x)

    assert torch.allclose(
        x, inverse_transformed_x
    ), "Inverse transform mismatch"
    assert standardizer.mean.shape == (3,), "Mean shape mismatch"
    assert standardizer.std.shape == (3,), "Std shape mismatch"
    assert standardizer.mean.dtype == torch.float32, "Mean dtype mismatch"
    assert standardizer.std.dtype == torch.float32, "Std dtype mismatch"


def test_quantile_fitted_transformer():
    # Generate some random input data (must be positive)
    x = torch.exp(torch.randn(5, 3))

    # Create a simple QuantileFittedTransformer
    quantile_transformer = QuantileFittedTransformer()

    # Transform the data
    transformed_x = quantile_transformer.transform(x)

    assert transformed_x.shape == x.shape, "Transformed shape mismatch"
    assert isinstance(transformed_x, torch.Tensor), "Transformed type mismatch"
    assert transformed_x.dtype == torch.float32, "Transformed dtype mismatch"

    # Inverse transform the data
    inverse_transformed_x = quantile_transformer.inverse_transform(
        transformed_x
    )

    assert torch.allclose(
        x, inverse_transformed_x
    ), "Inverse transform mismatch"

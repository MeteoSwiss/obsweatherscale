import torch


class Standardizer:
    """Standardization transformation class."""

    def __init__(
        self,
        data: torch.Tensor,
        variables: tuple[int, ...] | int | None = None
    ) -> None:
        """Initialize the Standardizer.

        Parameters
        ----------
        data : torch.Tensor
            The data to be standardized.
        variables : tuple[int, ...] or int, optional
            The dimensions to be used for standardization.
            If None, all dimensions will be used.
        """
        self.fit(data, variables)

    @property
    def description(self) -> str:
        """Return a short description of the sandard normalization."""
        return "Standard normalization: f(y) = (y - mean(y) / stddev(y))"

    def fit(
        self,
        data: torch.Tensor,
        variables: tuple[int, ...] | int | None = None
    ) -> None:
        """Fit standardization transformation to input data.

        Parameters
        ----------
        data : torch.Tensor
            The data to be standardized.
        variables : tuple[int, ...] or int, optional
            The dimensions to be used for standardization.
            If None, all dimensions will be used.
        """
        self.mean = data.mean(dim=variables).squeeze()
        self.std = data.std(dim=variables).squeeze()

    def transform(self, y: torch.Tensor, copy: bool = False) -> torch.Tensor:
        """Apply standardization transformation to input data."""
        if copy:
            y = y.detach().clone()
        return (y - self.mean) / self.std

    def inverse_transform(
        self,
        z: torch.Tensor,
        copy: bool = False
    ) -> torch.Tensor:
        """Apply inverse standardization transformation to input data."""
        if copy:
            z = z.detach().clone()
        return z * self.std + self.mean

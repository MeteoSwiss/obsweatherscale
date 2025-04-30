from typing import Union, TypeVar
import torch
import xarray as xr

XRData = TypeVar("XRData", xr.Dataset, xr.DataArray)


def coords_to_station(dataset: XRData) -> XRData:
    """Convert an xarray object with x and y coordinates to a stacked
    representation with a single station dimension.

    This function takes an xarray Dataset or DataArray with separate x
    and y coordinates and stacks them into a single 'station' dimension.
    This can be useful for processing data at specific locations or
    stations rather than working with a full grid.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input xarray object with x and y coordinates that will be
        stacked.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The input dataset with x and y coordinates stacked into a single
        'station' dimension. The returned object is of the same type as
        the input.
    """
    return dataset.stack(station=("x", "y"))


def station_to_coords(dataset: XRData) -> XRData:
    """Convert an xarray object with a station dimension back to
    separate x and y coordinates.

    This function performs the reverse operation of coords_to_station,
    unstacking the 'station' dimension back into separate x and y
    coordinates.

    Parameters
    ----------
    dataset : xr.Dataset or xr.DataArray
        The input xarray object with a 'station' dimension that will be
        unstacked.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The input dataset with the 'station' dimension unstacked into
        separate x and y coordinates. The returned object is of the same
        type as the input.
    
    Notes
    -----
    This function assumes that the 'station' dimension was created by
    stacking x and y coordinates. It will fail if the 'station'
    dimension does not exist or was not created from x and y coordinates.
    """
    return dataset.unstack("station")


def wrap_tensor(
    pred: torch.Tensor,
    dims: tuple[str, ...],
    coords: dict[str, Union[torch.Tensor, xr.DataArray]],
    name: str = "",
    realization_name: str = "realization",
) -> xr.Dataset:
    """Convert a PyTorch tensor into an xarray Dataset with specified
    dimensions and coordinates.

    This function wraps a PyTorch tensor into an xarray Dataset,
    allowing for easy integration of PyTorch model outputs with xarray's
    labeled data structures. If the input tensor has more than 3
    dimensions, an additional dimension (default name 'realization') is
    automatically added.

    Parameters
    ----------
    pred : torch.Tensor
        The PyTorch tensor to be converted into an xarray Dataset.
        The tensor should be detached from the computational graph.

    dims : tuple[str, ...]
        A tuple of dimension names that correspond to the dimensions of
        the input tensor. The number of dimensions should match the
        tensor's shape, or be one less if the tensor has more than 3
        dimensions (in which case a realization dimension is added).

    coords : dict[str, Union[torch.Tensor, xr.DataArray]]
        A dictionary mapping dimension names to their corresponding
        coordinate values.
        Coordinates can be either PyTorch tensors or xarray DataArrays.

    name : str, optional
        The name to assign to the resulting DataArray within the Dataset.
        Defaults to an empty string.

    realization_name : str, optional
        The name to use for the realization dimension if the tensor has
        more than 3 dimensions.
        Defaults to "realization".

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the input tensor data with the
        specified dimensions, coordinates, and variable name.
    """
    # Create dimensions
    if len(pred.shape) > 3:
        dims += (realization_name,)

    # Transform back to DataArray
    pred_da = xr.DataArray(pred.detach(), coords, dims, name=name)

    return pred_da.to_dataset()

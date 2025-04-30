from typing import Optional
import xarray as xr


def to_dataarray(
    dataset: xr.Dataset,
    variable_name: Optional[str] = None,
) -> xr.DataArray:
    if variable_name is None:
        return dataset.to_dataarray("").squeeze("")
    return dataset.to_array(variable_name)
